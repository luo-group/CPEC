import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from datasets.sequence_dataset import ESMDataset
from models.mlp import MLPModel
from utils import commons

torch.set_num_threads(1)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    all_loss = []
    all_output = []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            # print(output, output.shape)
            # print(label, label.shape)
            # input()
            loss = criterion(output, label)
            all_loss.append(commons.toCPU(loss).item())
            all_output.append(commons.toCPU(output))
        all_loss = torch.tensor(all_loss)
    model.train()
    
    return all_loss.mean().item()

def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, logger, config):
    model.train()
    n_bad = 0
    all_loss = []
    all_val_loss = []
    best_val_loss = 1.e10
    epsilon = 1e-4
    for epoch in range(config.num_epochs):
        start_epoch = time.time()
        val_loss = evaluate(model, val_loader, criterion, device)
        if val_loss > best_val_loss - epsilon:
            n_bad += 1
            if n_bad > config.patience:
                logger.info(f'No performance improvement for {config.patience} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! val_loss={val_loss:.4f}')
            n_bad = 0
            best_val_loss = val_loss
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(config.ckpt_dir, 'best_checkpoints.pt'))
        all_val_loss.append(val_loss)
        losses = []
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.append(commons.toCPU(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = torch.tensor(losses).mean().item()
        all_loss.append(mean_loss)
        lr_scheduler.step(mean_loss)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(config.ckpt_dir, 'last_checkpoints.pt'))
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}]: loss: {mean_loss:.4f}; val_loss: {val_loss:.4f}; train time: {commons.sec2min_sec(end_epoch - start_epoch)}')
        
    return all_loss, all_val_loss
        
def predict(model, test_loader, device, log_dir, logger):
    model.eval()
    all_output = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            all_output.append(output.detach().cpu())
    all_output = torch.cat(all_output, dim=0)
    model.train()
    torch.save(all_output, os.path.join(log_dir, 'logits_test.pt'))
    logger.info(f'Predictions saved to {os.path.join(log_dir, "logits_test.pt")}')
    

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('config', type=str, default='configs/train_mlp.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs_mlp')
    parser.add_argument('--tag', type=str, default='')
    
    args = parser.parse_args()
    return args


def main():
    start_overall = time.time()
    args = get_args()
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.train.seed = args.seed if args.seed is not None else config.train.seed
    commons.seed_all(config.train.seed)

    # Logging
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=False)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = commons.get_logger('train_mlp', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    # Load dataset
    trainset = ESMDataset(config.data.train_data_file, config.data.label_file)
    validset = ESMDataset(config.data.valid_data_file, config.data.label_file)
    testset = ESMDataset(config.data.test_data_file, config.data.label_file)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=config.train.batch_size, shuffle=False)
    config.model.out_dim = trainset.num_labels
    logger.info(f'Trainset size: {len(trainset)}; Validset size: {len(validset)}; Testset size: {len(testset)}')
    logger.info(f'Number of labels: {trainset.num_labels}')
    
    # Load model
    model = MLPModel(config.model)
    model.to(args.device)
    logger.info(model)
    logger.info(f'Trainable parameters: {commons.count_parameters(model)}')
    
    # Train
    criterion = globals()[config.train.loss]()
    optimizer = globals()[config.train.optimizer](model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.patience-5, verbose=True)
    train(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device, logger=logger, config=config.train)
    
    # Test
    best_ckpt = torch.load(os.path.join(config.train.ckpt_dir, 'best_checkpoints.pt'))
    model.load_state_dict(best_ckpt)
    predict(model=model, test_loader=test_loader, device=args.device, log_dir=log_dir, logger=logger)
    
if __name__ == '__main__':
    main()
