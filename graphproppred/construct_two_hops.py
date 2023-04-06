import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import pickle

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv", choices=['ogbg-molhiv', 'ogbg-molpcba', 'ogbg-molmuv'],
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    args = parser.parse_args()

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    import copy
    two_hop_dataset = copy.deepcopy(dataset)
    two_hop_edge_index = []
    two_hop_edge_attr = []
    two_hop_edge_slices = [0]
    for i in tqdm(range(len(dataset))): # loop though each graph
        graph = dataset[i]
        edge_index, edge_attr = graph.edge_index, graph.edge_attr # shape=(2, num_edges), shape=(num_edges,3)
        num_edges = graph.num_edges
        num_nodes = graph.num_nodes
        # construct hash table:
        hash_table = {}
        for j in range(num_edges):
            start_node_idx = edge_index[0,j].item()
            if start_node_idx in hash_table:
                hash_table[start_node_idx].append(j)
            else:
                hash_table[start_node_idx] = [j]
        for node_idx in range(num_nodes):
            if node_idx not in hash_table: # this is weird but some graphs has isolated nodes.
                continue
            for first_edge_idx in hash_table[node_idx]:
                first_edge = edge_index[:,first_edge_idx]
                first_edge_attr = edge_attr[first_edge_idx,:]
                hop_node_idx = first_edge[1].item()
                for second_edge_idx in hash_table[hop_node_idx]:
                    second_edge = edge_index[:,second_edge_idx]
                    if second_edge[1].item() == first_edge[0].item(): 
                        continue # we don't consider 1->2 and 2->1 as 1--two-hop-->1
                    if second_edge[1].item() in hash_table[first_edge[0].item()]: # note: first_edge[0].item() == node_idx
                        continue # we don't consider 1->2 as a two-hop path if there is a one-hop path between 1 & 2.
                    second_edge_attr = edge_attr[second_edge_idx,:]
                    two_hop_edge = [first_edge[0].item(), second_edge[1].item()]
                    two_hope_edge_attr = torch.cat([first_edge_attr, second_edge_attr], dim=-1)
                    two_hop_edge_index.append(two_hop_edge)
                    two_hop_edge_attr.append(two_hope_edge_attr)

        two_hop_edge_slices.append(len(two_hop_edge_index))

    two_hop_edge_index = torch.Tensor(two_hop_edge_index).T.long()
    two_hop_edge_attr = torch.stack(two_hop_edge_attr, dim=0)
    two_hop_edge_slices = torch.Tensor(two_hop_edge_slices)
    two_hop_dataset.edge_index, two_hop_dataset.edge_attr = two_hop_edge_index, two_hop_edge_attr
    two_hop_dataset.data['edge_index'], two_hop_dataset.data['edge_attr'] = two_hop_edge_index, two_hop_edge_attr
    two_hop_dataset.slices['edge_index'] = two_hop_edge_slices.long()
    two_hop_dataset.slices['edge_attr'] = two_hop_edge_slices.long()
    # two_hop_dataset.num_edge_features *= 2 # 3 -> 6, concatenating features of two edges.

    pickle.dump(two_hop_dataset, open('/home/haotao/GNN-MoE/mol/two_hop_%s_dataset.pkl' % args.dataset, 'wb'))

if __name__ == "__main__":
    main()
