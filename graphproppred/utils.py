import random
from typing import Tuple
import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_node, subgraph


@functional_transform('random_node_drop')
class RandomDropNode(BaseTransform):
    r"""Randomly drop nodes from a graph with ratio p for graph prediction task only.

    Args:
        p (float): randomly drop nodes with probability p.
    """
    def __init__(self, p: float):
        assert isinstance(p, float) and 0 <= p <=1
        self.p = p

    def __call__(self, data: Data) -> Data:
        if data.x.size(0) < 5:
            if 'num_nodes' not in data.keys:
                import numpy as np 
                print('found bad data')
                np.save('bad_data.npy', data)
            return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y)
        # edge_index, edge_mask, node_mask = dropout_node(data.edge_index, p=self.p, num_nodes=data.num_nodes)
        node_mask = (torch.empty(data.x.size(0)).uniform_(0, 1) > self.p)
        # node_mask = (torch.rand(1).item() > self.p)
        new_edge_index, new_edge_attr = subgraph(node_mask, data.edge_index, data.edge_attr, relabel_nodes =True)
        new_data = Data(x=data.x[node_mask], edge_index=new_edge_index, edge_attr=new_edge_attr, y=data.y) # TODO: Now for graph prediction task only. 
        if torch.sum(node_mask)==0:
            import numpy as np 
            print('found bad data')
            np.save('bad_data.npy', data)
            return Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y)
        else:
            return new_data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.p})'