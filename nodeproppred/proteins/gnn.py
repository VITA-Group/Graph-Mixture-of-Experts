import argparse, os, math

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from moe import MoE
import numpy as np 


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class GCN_SpMoE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, k=1, coef=1e-2):
        super(GCN_SpMoE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for layer_idx in range(num_layers - 2):
            if layer_idx % 2 == 0:
                ffn = MoE(input_size=hidden_channels, output_size=hidden_channels, num_experts=num_experts, k=k, coef=coef)
                self.convs.append(ffn)
            else:
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        self.load_balance_loss = 0 # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, MoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        self.load_balance_loss /= math.ceil((self.num_layers-2)/2)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class SAGE_SpMoE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_experts=4, k=1, coef=1e-2):
        super(SAGE_SpMoE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for layer_idx in range(num_layers - 2):
            if layer_idx % 2 == 0:
                ffn = MoE(input_size=hidden_channels, output_size=hidden_channels, num_experts=num_experts, k=k, coef=coef, sage=True)
                self.convs.append(ffn)
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        self.load_balance_loss = 0 # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, MoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        self.load_balance_loss /= math.ceil((self.num_layers-2)/2)
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    if isinstance(model, GCN_SpMoE):
        loss += model.load_balance_loss
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--gnn', default='gcn-spmoe', choices=['gcn', 'sage', 'gcn-spmoe', 'sage-spmoe'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', '-d', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--num_experts', '-n', type=int, default=8,
                        help='total number of experts in GCN-MoE')
    parser.add_argument('-k', type=int, default=4,
                        help='selected number of experts in GCN-MoE')
    parser.add_argument('--coef', type=float, default=1,
                        help='loss coefficient for load balancing loss in sparse MoE training')   

    args = parser.parse_args()
    print(args)

    exp_str = '%s-dropout%s-lr%s-e%d' % (args.gnn, args.dropout, args.lr, args.epochs)
    if 'spmoe' in args.gnn:
        exp_str += '-d%d-n%d-k%d-coef%s' % (args.hidden_channels, args.num_experts, args.k, args.coef)

    from datetime import datetime
    current_date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_str += '-%s' % current_date_and_time

    save_dir = os.path.join('results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.gnn == 'sage':
        model = SAGE(data.num_features, args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    elif args.gnn == 'gcn':
        model = GCN(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout).to(device)
    elif args.gnn == 'gcn-spmoe':
        model = GCN_SpMoE(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout, 
                    num_experts=args.num_experts, k=args.k, coef=args.coef).to(device)
    elif args.gnn == 'sage-spmoe':
        model = SAGE_SpMoE(data.num_features, args.hidden_channels, 112,
                    args.num_layers, args.dropout, 
                    num_experts=args.num_experts, k=args.k, coef=args.coef).to(device)

    if args.gnn in ['gcn', 'gcn-spmoe']:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    valid_curve = []
    test_curve = []
    train_curve = []
    for epoch in range(1, 1 + args.epochs):
        loss = train(model, data, train_idx, optimizer)

        if epoch % args.eval_steps == 0:
            result = test(model, data, split_idx, evaluator)
            train_rocauc, valid_rocauc, test_rocauc = result
            valid_curve.append(valid_rocauc)
            test_curve.append(test_rocauc)

            if epoch % args.log_steps == 0:
                log_str = f'Epoch: {epoch:02d}, '\
                        f'Loss: {loss:.4f}, '\
                        f'Train: {100 * train_rocauc:.2f}%, '\
                        f'Valid: {100 * valid_rocauc:.2f}% '\
                        f'Test: {100 * test_rocauc:.2f}%'
                print(log_str)

                with open(os.path.join(save_dir, '%s.txt' % exp_str), 'a+') as fp:
                    fp.write(log_str)
                    fp.write('\n')
                    fp.flush()
                    fp.close()

    best_val_epoch = np.argmax(np.array(valid_curve))
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    with open(os.path.join(save_dir, '%s.txt' % exp_str), 'a+') as fp:
        fp.write('Best validation score: {}\n'.format(valid_curve[best_val_epoch]))
        fp.write('Test score: {}\n'.format(test_curve[best_val_epoch]))
        fp.flush()
        fp.close()
        
    filename = os.path.join(save_dir, '%s.pth' % exp_str)
    torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch]}, filename)


if __name__ == "__main__":
    main()
