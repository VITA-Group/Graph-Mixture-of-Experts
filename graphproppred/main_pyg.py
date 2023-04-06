import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from gnn import GNN
from conv import GNN_SpMoE_node

from tqdm import tqdm
import argparse
import time
import numpy as np
import os
from utils import RandomDropNode

import pickle 

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                utility_loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                utility_loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            if isinstance(model.gnn_node, GNN_SpMoE_node):
                load_balance_loss = model.gnn_node.load_balance_loss
                loss = utility_loss + load_balance_loss
            else:
                loss = utility_loss
            loss.backward()
            optimizer.step()
    
    if isinstance(model.gnn_node, GNN_SpMoE_node):
        loss_str = 'loss: %.4f (utility: %.4f, load balance: %.4f)' % (loss.item(), utility_loss.item(), load_balance_loss.item())
        print(loss_str)

def train_mixed(model, device, loader, loader_2hop, optimizer, task_type):
    '''
    Deprecated
    '''
    model.train()

    for step, (batch, batch_2hop) in enumerate(tqdm(zip(loader, loader_2hop), desc="Iteration")):
        batch = batch.to(device)
        batch_2hop = batch_2hop.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch, batch_2hop)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                utility_loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                utility_loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            if isinstance(model.gnn_node, GNN_SpMoE_node):
                load_balance_loss = model.gnn_node.load_balance_loss
                loss = utility_loss + load_balance_loss
            else:
                loss = utility_loss
            loss.backward()
            optimizer.step()
    
    if isinstance(model.gnn_node, GNN_SpMoE_node):
        loss_str = 'loss: %.4f (utility: %.4f, load balance: %.4f)' % (loss.item(), utility_loss.item(), load_balance_loss.item())
        print(loss_str)

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def eval_mixed(model, device, loader, loader_2hop, evaluator):
    '''
    Deprecated
    '''
    model.eval()
    y_true = []
    y_pred = []

    for step, (batch, batch_2hop) in enumerate(tqdm(zip(loader, loader_2hop), desc="Iteration")):
        batch = batch.to(device)
        batch_2hop = batch_2hop.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, batch_2hop)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn-spmoe',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', '-d', type=int, default=150,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-moltox21", 
                        choices=['ogbg-molhiv', 'ogbg-molmuv', 'ogbg-molpcba', 
                        'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-molclintox',
                        'ogbg-molsider', 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molesol', 
                        'ogbg-molfreesolv', 'ogbg-mollipo'], 
                        help='dataset name')

    parser.add_argument('--drop_node_ratio', type=float, default=0.2,
                        help='randomly drop node with a ratio')

    parser.add_argument('--num_experts', '-n', type=int, default=8,
                        help='total number of experts in GCN-MoE')
    parser.add_argument('-k', type=int, default=4,
                        help='selected number of experts in GCN-MoE')
    parser.add_argument('--hop', type=int, default=1,
                        help='number of GCN hops')
    parser.add_argument('--num_experts_1hop', '--n1', type=int, default=8,
                        help='number of hop-1 experts in GCN-MoE. Only used when --hop>1.')
    parser.add_argument('--coef', type=float, default=1,
                        help='loss coefficient for load balancing loss in sparse MoE training')   
    parser.add_argument('--pretrain', default='',
                        help='pretrained ckpt file name')                    
    args = parser.parse_args()

    exp_str = '%s-%s-dropout%s-lr%s-wd%s' % (args.dataset, args.gnn, args.drop_ratio, args.lr, args.weight_decay)
    # exp_str = '%s-%s' % (args.dataset, args.gnn)
    if 'spmoe' in args.gnn:
        exp_str += '-d%d-n%d-k%d-coef%s' % (args.emb_dim, args.num_experts, args.k, args.coef)
    elif 'moe' in args.gnn:
        exp_str += '-d%d-n%d' % (args.emb_dim, args.num_experts)    
    if 'moe' in args.gnn:
        args.num_experts_1hop = args.num_experts if args.num_experts_1hop < 0 else args.num_experts_1hop
        exp_str += '-split-%d-%d' % (args.num_experts_1hop, args.num_experts-args.num_experts_1hop) # e.g., gcn-spmoe-split-2-2
    elif args.hop > 1:
        exp_str += '-hop%d' % args.hop # e.g., gcn-hop2 
    if args.pretrain:
        exp_str += '-pretrained'
    print('Saving as %s' % exp_str)
    
    from datetime import datetime
    current_date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_str += '-%s' % current_date_and_time

    save_dir = os.path.join('results', args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    train_transform = RandomDropNode(p=args.drop_node_ratio)
    transformed_dataset = PygGraphPropPredDataset(name = args.dataset, transform=train_transform)
    dataset = PygGraphPropPredDataset(name = args.dataset)
    split_idx = dataset.get_idx_split()
    train_set = transformed_dataset[split_idx["train"]]
    valid_set, test_set = dataset[split_idx["valid"]], dataset[split_idx["test"]]
    # try:
    #     dataset_2hop = pickle.load(open('/home/haotao/GNN-MoE/mol/two_hop_%s_dataset.pkl' % args.dataset, 'rb'))
    # except:
    #     dataset_2hop = None

    def get_loaders(train_set, valid_set, test_set, train_sampler):
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers = args.num_workers)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

        return train_loader, valid_loader, test_loader

    # if dataset_2hop:
    #     train_set_2hop, valid_set_2hop, test_set_2hop = split_dataset(dataset_2hop)

    # seed = np.random.randint(0,1000)
    # seed=0
    # _train_generator = torch.Generator()
    # _train_generator.manual_seed(seed)
    # torch.manual_seed(seed)
    # train_sampler = RandomSampler(train_set, generator=_train_generator) # use the same random sampler to sync the two datasets.
    train_sampler = SequentialSampler(train_set)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-spmoe':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, 
                    moe='sparse', num_experts=args.num_experts, k=args.k, coef=args.coef, num_experts_1hop=args.num_experts_1hop).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, 
                    hop=args.hop).to(device)
    elif args.gnn == 'gcn-moe':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, 
                    moe='dense', num_experts=args.num_experts, num_experts_1hop=args.num_experts_1hop).to(device)
    elif args.gnn == 'gcn-spmoe':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, 
                    moe='sparse', num_experts=args.num_experts, k=args.k, coef=args.coef, num_experts_1hop=args.num_experts_1hop).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    if args.pretrain:
        ckpt = torch.load(os.path.join('results/ogbg-molpcba', args.pretrain))
        pretrained_state_dict = ckpt['model']
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'graph_pred_linear' not in k}
        model.state_dict().update(pretrained_state_dict)
        # model.load_state_dict(ckpt['model'])
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))

        random_shuffled_idx = np.random.permutation(len(train_set))
        train_set = train_set[random_shuffled_idx]
        train_loader, valid_loader, test_loader = \
            get_loaders(train_set, valid_set, test_set, train_sampler)
        # if dataset_2hop:
        #     train_set_2hop = train_set_2hop[random_shuffled_idx]
        #     train_loader_2hop, valid_loader_2hop, test_loader_2hop = \
        #         get_loaders(train_set_2hop, valid_set_2hop, test_set_2hop, train_sampler)
        
        if 'moe' in args.gnn:
            if 'moe' in args.gnn and args.num_experts_1hop == 0:
                raise Exception("No longer support two hop datasets after applying random node drop!")
                print('Training...')
                train(model, device, train_loader_2hop, optimizer, dataset.task_type)
                print('Evaluating...')
                train_perf = eval(model, device, train_loader_2hop, evaluator)
                valid_perf = eval(model, device, valid_loader_2hop, evaluator)
                test_perf = eval(model, device, test_loader_2hop, evaluator)
            elif 'moe' in args.gnn and args.num_experts_1hop == args.num_experts:
                print('Training...')
                train(model, device, train_loader, optimizer, dataset.task_type)
                print('Evaluating...')
                # train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)
            else:
                raise Exception("Now using drop node, which doesn't support mixed training/eval yet!")
                print('Training mixed...')
                train_mixed(model, device, train_loader, train_loader_2hop, optimizer, dataset.task_type)
                print('Evaluating mixed...')
                train_perf = eval_mixed(model, device, train_loader, train_loader_2hop, evaluator)
                valid_perf = eval_mixed(model, device, valid_loader, valid_loader_2hop, evaluator)
                test_perf = eval_mixed(model, device, test_loader, test_loader_2hop, evaluator)
        else:
            if args.hop == 1:
                print('Training...')
                train(model, device, train_loader, optimizer, dataset.task_type)
                print('Evaluating...')
                # train_perf = eval(model, device, train_loader, evaluator)
                valid_perf = eval(model, device, valid_loader, evaluator)
                test_perf = eval(model, device, test_loader, evaluator)
            elif args.hop == 2:
                raise Exception("No longer support two hop datasets after applying random node drop!")
                print('Training...')
                train(model, device, train_loader_2hop, optimizer, dataset.task_type)
                print('Evaluating...')
                train_perf = eval(model, device, train_loader_2hop, evaluator)
                valid_perf = eval(model, device, valid_loader_2hop, evaluator)
                test_perf = eval(model, device, test_loader_2hop, evaluator)

        print({'Validation': valid_perf, 'Test': test_perf})
        with open(os.path.join(save_dir, '%s.txt' % exp_str), 'a+') as fp:
            # fp.write(str({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf}))
            fp.write(str({'Validation': valid_perf, 'Test': test_perf}))
            fp.write('\n')
            fp.flush()
            fp.close()
        # eval_str = 'Train: %.4f | Validation: %.4f | Test: %.4f' % (train_perf, valid_perf, test_perf)
        # print(eval_str)
        # with open('%s.txt' % exp_str) as fp:
        #     fp.write(eval_str+'\n')
        #     fp.flush()
        #     fp.close()

        # train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        # best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        # best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))
    with open(os.path.join(save_dir, '%s.txt' % exp_str), 'a+') as fp:
        fp.write('Best validation score: {}\n'.format(valid_curve[best_val_epoch]))
        fp.write('Test score: {}\n'.format(test_curve[best_val_epoch]))
        fp.flush()
        fp.close()

    filename = os.path.join(save_dir, '%s.pth' % exp_str)
    # torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, filename)
    if args.dataset == 'ogbg-molpcba':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'model': model.state_dict()}, filename)
    else:
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch]}, filename)


if __name__ == "__main__":
    main()
