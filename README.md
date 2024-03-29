# Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling

Official code for "Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling" in NeurIPS 2023. 

## Introduction

In this work, we propose the Graph Mixture of Experts (GMoE) model structure to enhance the ability of GNNs to accommodate the diversity of training graph structures, without incurring computational overheads at inference.

## How to run the code

To train the GMoE model, run

```
python main_pyg.py --dataset $dataset -n $total_number_of_experts --n1 $number_of_one_hop_experts -k $number_of_selected_experts -d $feature_dimension --device 0 --gnn gcn-spmoe --coef 1
```

For example, on ogbg-molhiv dataset, run 

```
python main_pyg.py --dataset ogbg-molhiv -n 8 --n1 4 -k 4 -d 150 --device 0 --gnn gcn-spmoe --coef 1
```

The test results for the best performing model on validation set will be recorded in the output files generated by the training code.


## Acknowledgement

Our code is built upon the [official OGB code](https://github.com/snap-stanford/ogb/tree/master/examples).

## Citation

```
@inproceedings{wang2023gmoe,
author = {Wang, Haotao and Jiang, Ziyu and You, Yuning and Han, Yan and Liu, Gaowen and Srinivasa, Jayanth and Kompella, Ramana Rao  and Wang, Zhangyang},
title = {Graph Mixture of Experts: Learning on Large-Scale Graphs with Explicit Diversity Modeling},
booktitle = {NeurIPS}, 
year = {2023}
}
```
