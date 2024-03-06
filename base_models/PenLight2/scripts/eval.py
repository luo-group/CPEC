import sys
sys.path.append('.')
import torch
import torch_geometric
from torch.utils.tensorboard import SummaryWriter
import time, argparse, datetime, yaml, logging, json, os, shutil
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from models.gat import GATModel
from utils import commons
from utils.losses import TripletLoss
from datasets.triplet_dataset import TripletMultilabelDataset, ProteinGraphDataset

torch.set_num_threads(1)
node_in_dim = (6, 3)
edge_in_dim = (32, 1)

def calculate_multilabel_f1_score(true_labels, predicted_labels):
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit the binarizer and transform the true labels
    y_true = mlb.fit_transform(true_labels.values())

    # Transform the predicted labels using the same binarizer
    y_pred = mlb.transform(predicted_labels.values())

    # Calculate F1 score, with average='macro' or 'micro' based on your preference
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def get_leveled_id2label(id2label, level):
    leveled_id2label = {}
    for k, v in id2label.items():
        leveled_v = ['.'.join(_v.split('.')[:level]) for _v in v]
        leveled_id2label[k] = leveled_v
    
    return leveled_id2label

def accs2str(acc):
    return '; '.join([f'f1-{i}: {acc[i]:.4f}' for i in range(len(acc))])    

def predict(model, best_ckpt, lookupset, valset, testset, calibset, exp_dir, logger, args):
    model.load_state_dict(best_ckpt)
    model.eval()
    with torch.no_grad():
        lookup_emb, test_emb, calib_emb = [], [], []
        lookup_loader = torch_geometric.loader.DataLoader(lookupset, batch_size=16, shuffle=False, drop_last=False)
        test_loader = torch_geometric.loader.DataLoader(testset, batch_size=16, shuffle=False, drop_last=False)
        calib_loader = torch_geometric.loader.DataLoader(calibset, batch_size=16, shuffle=False, drop_last=False)
        val_loader = torch_geometric.loader.DataLoader(valset, batch_size=16, shuffle=False, drop_last=False)
        for idx, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(args.device)
            emb = model.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name, batch=test_batch.batch)
            test_emb.append(emb.cpu().detach())
        for idx, lookup_batch in enumerate(lookup_loader):
            lookup_batch = lookup_batch.to(args.device)
            emb = model.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr, name=lookup_batch.name, batch=lookup_batch.batch)
            lookup_emb.append(emb.cpu().detach())
        for idx, calib_batch in enumerate(calib_loader):
            calib_batch = calib_batch.to(args.device)
            emb = model.single_pass(calib_batch.x, calib_batch.edge_index, calib_batch.edge_attr, name=calib_batch.name, batch=calib_batch.batch)
            calib_emb.append(emb.cpu().detach())
        
        lookup_emb = torch.cat(lookup_emb, dim=0)
        test_emb = torch.cat(test_emb, dim=0)
        calib_emb = torch.cat(calib_emb, dim=0)
        
    lookup_id2label = lookupset.get_id2label()
    test_id2label = testset.get_id2label()
    calib_id2label = calibset.get_id2label()
    lookup_labels = list(lookup_id2label.values())
    test_labels = list(test_id2label.values())
    calib_labels = list(calib_id2label.values())
    
    
    distance_test_lookup = torch.cdist(test_emb, lookup_emb)
    test_id2pred = {}
    for idx, test_id in enumerate(test_id2label):
        test_id2pred[test_id] = lookup_labels[torch.argmin(distance_test_lookup[idx])]
    f1_scores_leveled = []
    for level in range(1, 5):
        leveled_test_id2label = get_leveled_id2label(test_id2label, level)
        leveled_test_id2pred = get_leveled_id2label(test_id2pred, level)
        f1_scores_leveled.append(calculate_multilabel_f1_score(leveled_test_id2label, leveled_test_id2pred))
    logger.info(f'f1 scores: {accs2str(f1_scores_leveled)}')
    torch.save(distance_test_lookup, os.path.join(exp_dir, 'distance_test_lookup.pt'))
    with open(os.path.join(exp_dir, 'test_id2pred.json'), 'w') as f:
        json.dump(test_id2pred, f)
    with open(os.path.join(exp_dir, 'lookup_ids.json'), 'w') as f:
        json.dump(list(lookup_id2label.keys()), f)
    with open(os.path.join(exp_dir, 'test_ids.json'), 'w') as f:
        json.dump(list(test_id2label.keys()), f)
    
    distance_calib_lookup = torch.cdist(calib_emb, lookup_emb)
    calib_id2pred = {}
    for idx, calib_id in enumerate(calib_id2label):
        calib_id2pred[calib_id] = lookup_labels[torch.argmin(distance_calib_lookup[idx])]
    f1_scores_leveled = []
    for level in range(1, 5):
        leveled_calib_id2label = get_leveled_id2label(calib_id2label, level)
        leveled_calib_id2pred = get_leveled_id2label(calib_id2pred, level)
        f1_scores_leveled.append(calculate_multilabel_f1_score(leveled_calib_id2label, leveled_calib_id2pred))
    logger.info(f'f1 scores: {accs2str(f1_scores_leveled)}')
    torch.save(distance_calib_lookup, os.path.join(exp_dir, 'distance_calib_lookup.pt'))
    with open(os.path.join(exp_dir, 'calib_id2pred.json'), 'w') as f:
        json.dump(calib_id2pred, f)
    with open(os.path.join(exp_dir, 'calib_ids.json'), 'w') as f:
        json.dump(list(calib_id2label.keys()), f)
    
    lookup_label2pid = {}
    for pid, labels in lookup_id2label.items():
        for label in labels:
            if label not in lookup_label2pid:
                lookup_label2pid[label] = []
            lookup_label2pid[label].append(pid)
    lookup_label2emb = {}
    lookup_id2emb = {}
    for i, pid in enumerate(lookup_id2label):
        lookup_id2emb[pid] = lookup_emb[i]
    for label, pids in lookup_label2pid.items():
        for pid in pids:
            if label not in lookup_label2emb:
                lookup_label2emb[label] = []
            lookup_label2emb[label].append(lookup_id2emb[pid])
    for label, embs in lookup_label2emb.items():
        lookup_label2emb[label] = torch.vstack(embs).mean(dim=0)
    ec_list = list(lookup_label2emb.keys())
    ec_cluster_emb = torch.vstack(list(lookup_label2emb.values()))
    distance_test_ec_cluster = torch.cdist(test_emb, ec_cluster_emb)
    distance_calib_ec_cluster = torch.cdist(calib_emb, ec_cluster_emb)
    distance_lookup_ec_cluster = torch.cdist(lookup_emb, ec_cluster_emb)
    with open(os.path.join(exp_dir, 'ec_cluster_list.json'), 'w') as f:
        json.dump(ec_list, f)
    torch.save(distance_test_ec_cluster, os.path.join(exp_dir, 'distance_test_ec_cluster.pt'))
    torch.save(distance_calib_ec_cluster, os.path.join(exp_dir, 'distance_calib_ec_cluster.pt'))
    torch.save(distance_lookup_ec_cluster, os.path.join(exp_dir, 'distance_lookup_ec_cluster.pt'))

def get_args():
    parser = argparse.ArgumentParser(description='Train GAT model')
    
    parser.add_argument('config', type=str, default='configs/train_EC.yml')
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--val_data_file', type=str, default=None)
    parser.add_argument('--test_data_file', type=str, default=None)
    parser.add_argument('--calib_data_file', type=str, default=None)
    parser.add_argument('--ckpt', type=str)
    
    args = parser.parse_args()
    
    return args
        
def main():
    args = get_args()
    
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.train.seed = config.train.seed if args.seed is None else args.seed
    commons.seed_all(config.train.seed)
    config.data.data_file = args.data_file if args.data_file is not None else config.data.data_file
    
    # logger
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=True)
    logger = commons.get_logger('train')
    
    # dataset
    lookupset = ProteinGraphDataset(config.data.data_file, config.data, logger)
    testset = ProteinGraphDataset(args.test_data_file, config.data, logger)
    calibset = ProteinGraphDataset(args.calib_data_file, config.data, logger)
    valset = ProteinGraphDataset(args.val_data_file, config.data, logger)
    
    # load model
    config.model.in_channels = 1280 if config.data.seq_emb == 'esm' else 1024
    if config.model.append_scalar_features:
        config.model.in_channels += node_in_dim[0]
    config.model.edge_dim=32 if config.data.scalar_only else 35
    
    model = GATModel(config.model)
    logger.info(model)
    model.to(args.device)
    
    # checkpoint
    ckpt = torch.load(args.ckpt)
    
    # predict
    predict(model, ckpt, lookupset, valset, testset, calibset, log_dir, logger, args)
    
    
if __name__ == '__main__':
    main()