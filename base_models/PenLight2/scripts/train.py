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

torch.set_num_threads(4)
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

def calculate_accuracy(true_labels, predicted_labels):
    # print('true_labels', true_labels.values())
    # print('predicted_labels', predicted_labels.values())
    # input()
    y_true = list(true_labels.values())
    y_pred = list(predicted_labels.values())
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    return correct / len(y_true)

def get_leveled_id2label(id2label, level):
    leveled_id2label = {}
    for k, v in id2label.items():
        leveled_v = ['.'.join(_v.split('.')[:level]) for _v in v]
        leveled_id2label[k] = leveled_v
    
    return leveled_id2label

def predict(model, best_ckpt, lookupset, testset, calibset, exp_dir, logger, args):
    model.load_state_dict(best_ckpt)
    model.eval()
    with torch.no_grad():
        lookup_emb, test_emb, calib_emb = [], [], []
        lookup_loader = torch_geometric.loader.DataLoader(lookupset, batch_size=16, shuffle=False, drop_last=False)
        test_loader = torch_geometric.loader.DataLoader(testset, batch_size=16, shuffle=False, drop_last=False)
        calib_loader = torch_geometric.loader.DataLoader(calibset, batch_size=16, shuffle=False, drop_last=False)
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

def evaluation(model, lookupset, testset, logger, args):
    model.eval()
    with torch.no_grad():
        lookup_emb, test_emb = [], []
        lookup_loader = torch_geometric.loader.DataLoader(lookupset, batch_size=16, shuffle=False, drop_last=False)
        test_loader = torch_geometric.loader.DataLoader(testset, batch_size=16, shuffle=False, drop_last=False)
        for idx, test_batch in enumerate(test_loader):
            test_batch = test_batch.to(args.device)
            emb = model.single_pass(test_batch.x, test_batch.edge_index, test_batch.edge_attr, name=test_batch.name, batch=test_batch.batch)
            test_emb.append(emb.cpu().detach())
        for idx, lookup_batch in enumerate(lookup_loader):
            lookup_batch = lookup_batch.to(args.device)
            emb = model.single_pass(lookup_batch.x, lookup_batch.edge_index, lookup_batch.edge_attr, name=lookup_batch.name, batch=lookup_batch.batch)
            lookup_emb.append(emb.cpu().detach())
        lookup_emb = torch.cat(lookup_emb, dim=0)
        test_emb = torch.cat(test_emb, dim=0)
        
    lookup_id2label = lookupset.get_id2label()
    test_id2label = testset.get_id2label()
    lookup_labels = list(lookup_id2label.values())
    test_labels = list(test_id2label.values())
    
    distance = torch.cdist(test_emb, lookup_emb)
    test_id2pred = {}
    for idx, test_id in enumerate(test_id2label):
        test_id2pred[test_id] = lookup_labels[torch.argmin(distance[idx])]
    # print(f'test_id2pred: {test_id2pred}')
    # input()
    # print(f'test_id2label: {test_id2label}')
    # input()
    # TODO: implement multi-level accuracy
    # correct = [0]
    # for test_id, predictions in test_id2pred.items():
    #     hit = 0
    #     for label in predictions:
    #         for gt in test_id2label[test_id]:
    #             if label == gt:
    #                 hit = 1
    #     correct[0] += hit
    # acc = [correct[i] / len(test_id2label) for i in range(len(correct))]
    f1_scores_leveled = []
    for level in range(1, 5):
        leveled_test_id2label = get_leveled_id2label(test_id2label, level)
        leveled_test_id2pred = get_leveled_id2label(test_id2pred, level)
        f1_scores_leveled.append(calculate_accuracy(leveled_test_id2label, leveled_test_id2pred))
        
    model.train()
    
    return f1_scores_leveled

def accs2str(acc):
    return '; '.join([f'acc-{i}: {acc[i]:.4f}' for i in range(len(acc))])    

def train(model, trainloader, lookupset, valset, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args, config):
    model.train()
    
    n_bad = 0
    best_acc = 0
    all_loss = []
    best_ckpt = None
    for epoch in range(num_epochs):
        start = time.time()
        acc = evaluation(model, lookupset, valset, logger, args)
        end_test = time.time()
        if acc[-1] < best_acc:
            n_bad += 1
            if n_bad >= config.early_stop_threshold:
                logger.info(f'No performance improvement for {config.early_stop_threshold} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! acc-{len(acc)-1}={acc[-1]:.4f}')
            n_bad = 0
            best_acc = acc[-1]
            # TODO: save checkpoint
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_checkpoint.pt'))
            best_ckpt = model.state_dict()
        losses = []
        for train_idx, data in enumerate(trainloader):
            data = data.to(device)
            anchor, pos, neg = model(data)
            loss = criterion(anchor, pos, neg, data.label)
            losses.append(commons.toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), os.path.join(exp_dir, 'last_checkpoint.pt'))
        end_epoch = time.time()
        
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; {accs2str(acc)}; train time: {commons.sec2min_sec(end_epoch - end_test)}')
        
    return all_loss, best_ckpt

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
    parser.add_argument('--use_timestamp', action='store_true')
    
    args = parser.parse_args()
    
    return args

def main():
    start_overall = time.time()
    args = get_args()
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.train.seed = config.train.seed if args.seed is None else args.seed
    commons.seed_all(config.train.seed)
    config.data.data_file = args.data_file if args.data_file is not None else config.data.data_file
    config.data.val_data_file = args.val_data_file if args.val_data_file is not None else config.data.val_data_file
    config.data.test_data_file = args.test_data_file if args.test_data_file is not None else config.data.test_data_file
    config.data.calib_data_file = args.calib_data_file if args.calib_data_file is not None else config.data.calib_data_file

    # Logging
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=args.use_timestamp)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = commons.get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    # shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)
    
    # load dataset
    trainset = TripletMultilabelDataset(config.data, logger=logger)
    lookupset = ProteinGraphDataset(config.data.data_file, config.data, logger)
    valset = ProteinGraphDataset(config.data.val_data_file, config.data, logger)
    # val_id2label = valset.get_id2label()
    # print(val_id2label)
    # input()
    testset = ProteinGraphDataset(args.test_data_file, config.data, logger)
    calibset = ProteinGraphDataset(args.calib_data_file, config.data, logger)
    train_loader = torch_geometric.loader.DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True, num_workers=1, follow_batch=['x_anchor', 'x_pos', 'x_neg'])
    
    # load model
    config.model.in_channels = 1280 if config.data.seq_emb == 'esm' else 1024
    if config.model.append_scalar_features:
        config.model.in_channels += node_in_dim[0]
    config.model.edge_dim=32 if config.data.scalar_only else 35
    
    model = GATModel(config.model)
    logger.info(model)
    model.to(args.device)
    
    criterion = TripletLoss(exclude_easy=config.train.exclude_easy, batch_hard=config.train.batch_hard, margin=config.train.margin, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.train.patience, verbose=True)
    
    # training
    _, best_ckpt = train(model, train_loader, lookupset, valset, config.train.num_epochs, optimizer, scheduler, criterion, args.device, ckpt_dir, logger, args, config.train)
    
    # save config
    with open(os.path.join(log_dir, os.path.basename(args.config)), 'w') as f:
        yaml.dump(dict(config), f)
        
    # predict
    # predict(model, best_ckpt, lookupset, testset, log_dir, logger, args)
    predict(model=model, best_ckpt=best_ckpt, lookupset=lookupset, testset=testset, calibset=calibset, exp_dir=log_dir, logger=logger, args=args)
    
if __name__ == '__main__':
    main()
    
    