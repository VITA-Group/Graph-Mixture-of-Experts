import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

import math

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
full_bond_feature_dims = get_bond_feature_dims()
class TwoHopBondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(TwoHopBondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims+full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, hop=1):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if hop==1:
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif hop==2:
            self.bond_encoder = TwoHopBondEncoder(emb_dim = emb_dim)
        else:
            raise Exception('Unimplemented hop %d' % hop)  

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, hop=1):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if hop==1:
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif hop==2:
            self.bond_encoder = TwoHopBondEncoder(emb_dim = emb_dim)
        else:
            raise Exception('Unimplemented hop %d' % hop)  

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', hop=1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, hop=hop))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GNN_MoE_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_experts=3, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', num_experts_1hop=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
        '''

        super(GNN_MoE_node, self).__init__()
        self.num_layer = num_layer
        self.num_experts = num_experts
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            convs_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for expert_idx in range(num_experts):
                if gnn_type == 'gin':
                    convs_list.append(GINConv(emb_dim))
                elif gnn_type == 'gcn':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GCNConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GCNConv(emb_dim, hop=2))  
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                
                bn_list.append(torch.nn.BatchNorm1d(emb_dim))

            self.convs.append(convs_list)
            self.batch_norms.append(bn_list)

        # self.mix_fn = lambda h_expert_list: torch.mean(torch.stack(h_expert_list, dim=0), dim=0)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        # edge_index: shape=(2, N_batch)
        # edge_attr: shape=(N_batch, d_attr)

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h_expert_list = []
            for expert in range(self.num_experts):

                h = self.convs[layer][expert](h_list[layer], edge_index, edge_attr) # TODO: use different edge_index and edge_attr for each expert
                h = self.batch_norms[layer][expert](h)
                h_expert_list.append(h)

            h = torch.stack(h_expert_list, dim=0) # shape=[num_experts, num_nodes, d_features]         
            h = torch.mean(h, dim=0) # shape=[num_nodes, d_features]

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

from moe import MoE
class GNN_SpMoE_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, num_experts=3, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gcn', k=1, coef=1e-2, num_experts_1hop=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            JK: Jumping knowledge refers to "Representation Learning on Graphs with Jumping Knowledge Networks"
            k: k value for top-k sparse gating. 
            num_experts: total number of experts in each layer. 
            num_experts_1hop: number of hop-1 experts in each layer. The first num_experts_1hop are hop-1 experts. The rest num_experts-num_experts_1hop are hop-2 experts.
        '''

        super(GNN_SpMoE_node, self).__init__()
        self.num_layer = num_layer
        self.num_experts = num_experts
        self.k = k
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if not num_experts_1hop:
            self.num_experts_1hop = num_experts # by default, all experts are hop-1 experts.
        else:
            assert num_experts_1hop <= num_experts
            self.num_experts_1hop = num_experts_1hop

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.ffns = torch.nn.ModuleList()

        for layer in range(num_layer):
            convs_list = torch.nn.ModuleList()
            bn_list = torch.nn.ModuleList()
            for expert_idx in range(num_experts):
                if gnn_type == 'gin':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GINConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GINConv(emb_dim, hop=2)) 
                elif gnn_type == 'gcn':
                    if expert_idx < self.num_experts_1hop:
                        convs_list.append(GCNConv(emb_dim, hop=1))
                    else:
                        convs_list.append(GCNConv(emb_dim, hop=2))  
                else:
                    raise ValueError('Undefined GNN type called {}'.format(gnn_type))
                
                bn_list.append(torch.nn.BatchNorm1d(emb_dim))
                
            ffn = MoE(input_size=emb_dim, output_size=emb_dim, num_experts=num_experts, experts_conv=convs_list, experts_bn=bn_list, 
                    k=k, coef=coef, num_experts_1hop=self.num_experts_1hop)

            self.ffns.append(ffn)

        # self.mix_fn = lambda h_expert_list: torch.mean(torch.stack(h_expert_list, dim=0), dim=0)

    def forward(self, batched_data, batched_data_2hop=None):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        if batched_data_2hop:
            x_2hop, edge_index_2hop, edge_attr_2hop, batch_2hop = batched_data_2hop.x, batched_data_2hop.edge_index, batched_data_2hop.edge_attr, batched_data_2hop.batch

        ### computing input node embedding

        h_list = [self.atom_encoder(x)]
        self.load_balance_loss = 0 # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for layer in range(self.num_layer):

            if batched_data_2hop:
                h, _layer_load_balance_loss = self.ffns[layer](h_list[layer], edge_index, edge_attr, edge_index_2hop, edge_attr_2hop) 
            else:
                h, _layer_load_balance_loss = self.ffns[layer](h_list[layer], edge_index, edge_attr, None, None) 
            self.load_balance_loss += _layer_load_balance_loss

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)
        
        self.load_balance_loss /= self.num_layer

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):

        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
