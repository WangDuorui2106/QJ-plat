from torchtools import *
from data import MiniImagenetLoader, TieredImagenetLoader
from model import TRPN
from backbone.conv4 import EmbeddingImagenet
import shutil
import os
import random

tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
# replace dataset_root with your own
tt.arg.dataset_root = '/home/jovyan/16061175/dataset/'
tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
tt.arg.meta_batch_size = 20 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
tt.arg.transductive = True if tt.arg.transductive is None else tt.arg.transductive
tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

tt.arg.num_ways_train = tt.arg.num_ways
tt.arg.num_ways_test = tt.arg.num_ways

tt.arg.num_shots_train = tt.arg.num_shots
tt.arg.num_shots_test = tt.arg.num_shots

tt.arg.train_transductive = tt.arg.transductive
tt.arg.test_transductive = tt.arg.transductive

# model parameter related
tt.arg.emb_size = 128

# train, test parameters
tt.arg.train_iteration = 150000 if tt.arg.dataset == 'mini' else 200000
tt.arg.test_iteration = 10000
tt.arg.test_interval = 5000 if tt.arg.test_interval is None else tt.arg.test_interval
tt.arg.test_batch_size = 10
tt.arg.log_step = 100 if tt.arg.log_step is None else tt.arg.log_step

tt.arg.lr = 1e-3
tt.arg.grad_clip = 5
tt.arg.weight_decay = 1e-6
tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

def one_hot_encode(num_classes, class_idx):
    return torch.eye(num_classes)[class_idx]

def label2edge(label):
    # get size
    num_samples = label.size(1)

    # reshape
    label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
    label_j = label_i.transpose(1, 2)

    # compute edge
    edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

    # expand
    edge = edge.unsqueeze(1)
    edge = torch.cat([edge, 1 - edge], 1)
    return edge

def eval(partition='test', log_flag=True, global_step=0, enc_module=None, gcn_module=None, data_loader=None):
    best_acc = 0
    # set edge mask (to distinguish support and query edges)
    num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
    num_queries = tt.arg.num_ways_test * 1
    num_samples = num_supports + num_queries

    support_edge_mask = torch.zeros(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    evaluation_mask = torch.ones(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)
    # for semi-supervised setting, ignore unlabeled support sets for evaluation
    for c in range(tt.arg.num_ways_test):
        evaluation_mask[:,
        ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test,
        :num_supports] = 0
        evaluation_mask[:, :num_supports,
        ((c + 1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c + 1) * tt.arg.num_shots_test] = 0

    query_node_accrs1 = []

    # for each iteration
    for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):
        # load task data list
        [support_data,
         support_label,
         query_data,
         query_label] = data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                   num_ways=tt.arg.num_ways_test,
                                                                   num_shots=tt.arg.num_shots_test,
                                                                   seed=iter)
        # set as single data
        full_data = torch.cat([support_data, query_data], 1)
        full_label = torch.cat([support_label, query_label], 1)
        full_edge = label2edge(full_label)

        # set init edge
        init_edge = full_edge.clone()
        init_edge[:, :, num_supports:, :] = 0.5
        init_edge[:, :, :, num_supports:] = 0.5
        for i in range(num_queries):
            init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
            init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

        # for semi-supervised setting,
        for c in range(tt.arg.num_ways_test):
            init_edge[:, :, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test, :num_supports] = 0.5
            init_edge[:, :, :num_supports, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test] = 0.5

        # set as train mode
        enc_module.eval()
        gcn_module.eval()

        # (1) encode data
        full_data = [enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
        full_data = torch.stack(full_data, dim=1)

        # num_tasks x num_quries x num_ways
        query_score_list, learned_score_list = gcn_module(node_feat=full_data, adj=init_edge[:, 0, :num_supports, :num_supports])

        # compute node accuracy: num_tasks x num_quries x num_ways == {num_tasks x num_quries x num_supports} * {num_tasks x num_supports x num_ways}
        # query_score_list = query_score_list * evaluation_mask[:,num_supports:]
        query_node_pred1 = torch.bmm(query_score_list[:, :, :num_supports],
                                     one_hot_encode(tt.arg.num_ways_test, support_label.long()))
        query_node_accr1 = torch.eq(torch.max(query_node_pred1, -1)[1], query_label.long()).float().mean()
        query_node_accrs1 += [query_node_accr1.item()]

    # logging
    if log_flag:
        tt.log('---------------------------')
        tt.log_scalar('{}/node_accr1'.format(partition), np.array(query_node_accrs1).mean(), global_step)

        tt.log('evaluation: total_count=%d, accuracy1: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
               (iter,
                np.array(query_node_accrs1).mean() * 100,
                np.array(query_node_accrs1).std() * 100,
                1.96 * np.array(query_node_accrs1).std() / np.sqrt(float(len(np.array(query_node_accrs1)))) * 100))
        tt.log('---------------------------')