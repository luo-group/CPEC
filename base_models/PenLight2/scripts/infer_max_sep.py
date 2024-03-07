import sys 
sys.path.append('.')
import torch
import pandas as pd
import numpy as np
import os, json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse


def n_smallest(data, n):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    smallest_values, indices = torch.topk(-data, k=n)
    smallest_values = -smallest_values
    
    return smallest_values, indices        

def maximum_separation(dist_lst, first_grad=True, use_max_grad=False):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        if not len(large_grads[-1]) == 0:
            max_sep_i = large_grads[-1][opt]
        else:
            max_sep_i = 0
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def batch_max_sep(smallest_k_dist, smallest_k_indices, ecs_lookup):
    n_query = len(smallest_k_dist)
    pred_ecs = []
    distances = []
    for i in range(n_query):
        ecs = []
        dist = []
        dist_lst = smallest_k_dist[i]
        if not isinstance(dist_lst, list):
            dist_lst = dist_lst.tolist()
        max_sep_i = maximum_separation(dist_lst)
        for j in range(max_sep_i + 1):
            # print(smallest_k_indices[i][j], len(ecs_lookup))
            EC_j = ecs_lookup[smallest_k_indices[i][j]]
            dist_j = dist_lst[j]
            ecs.append(EC_j)
            dist.append(dist_j)
        pred_ecs.append(ecs)
        distances.append(dist)
    
    return pred_ecs, distances

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--test_data', type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    # log_dir = f'logs_cath_all_splits/train_EC_cath_{seed}'
    log_dir = args.model_dir
    dist_mat = torch.load(os.path.join(log_dir, 'distance_test_ec_cluster.pt'))
    with open(os.path.join(log_dir, 'ec_cluster_list.json')) as f:
        ec_cluster_list = json.load(f)
    with open(os.path.join(log_dir, 'test_ids.json')) as f:
        test_ids = json.load(f)
    smallest_k_dist, smallest_k_indices = n_smallest(dist_mat, n=10)
    pred_ecs, distances = batch_max_sep(smallest_k_dist, smallest_k_indices, ec_cluster_list)
    print(pred_ecs[:5], len(pred_ecs))
    with open(args.test_data) as f:
        test_data = json.load(f)
    test_labels = [d['ec'] for d in test_data.values()]
    print(test_labels[:5], len(test_labels))
    print(ec_cluster_list)
    # input()
    mlb = MultiLabelBinarizer()
    mlb.fit([ec_cluster_list])
    gt = mlb.transform(test_labels)
    pred = mlb.transform(pred_ecs)
    print(f'f1: {f1_score(gt, pred, average="micro")}, precision: {precision_score(gt, pred, average="micro")}, recall: {recall_score(gt, pred, average="micro")}, accuracy: {accuracy_score(gt, pred)}')
    input()
    with open(os.path.join(log_dir, 'max_sep_pred.csv'), 'w') as f:
        for i in range(len(test_ids)):
            f.write(f'{test_ids[i]},')
            for j in range(len(pred_ecs[i])):
                f.write(f'EC:{pred_ecs[i][j]}/{distances[i][j]:4f},')
            f.write('\n')
    
if __name__ == '__main__':
    main()
    # for i in range(5):
    #     print(f'Running seed {i}')
    #     main(i)
