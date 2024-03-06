import sys
sys.path.append('.')
import torch
from models.mlp import MLPModel
from utils import commons
import os, json
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, average_precision_score, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

device = 'cuda:0'
def evaluation(log_dir):
    config = commons.load_config('configs/train_mlp.yml')
    test_data = torch.load('data/mlp_data/test95_data.pt')
    val_data = torch.load('data/mlp_data/valid_data.pt')
    with open('data/mlp_data/all_labels.json', 'r') as f:
        all_labels = json.load(f)
    with open('logs_mlp/train_mlp_CE_seed_0/ec_list.json', 'r') as f:
        ec_list = json.load(f)
    # print(all_labels == ec_list)
    # input()
    print(len(all_labels))
    config.model.out_dim = len(all_labels)
    model = MLPModel(config.model)
    ckpt = torch.load(os.path.join(log_dir, 'checkpoints/last_checkpoints.pt'))
    model.load_state_dict(ckpt)
    # print(model)
    model.to(device)
    model.eval()
    
    # val_output = []
    # val_ids =  []
    # for pid, data in tqdm(list(val_data.items())):
    #     emb = data['embedding']
    #     labels = data['label']
    #     val_ids.append(pid)
    #     emb = emb.to(device)
    #     with torch.no_grad():
    #         output = model(emb).detach().cpu()
    #     val_output.append(output)
    # val_output = torch.stack(val_output)
    # torch.save(val_output, os.path.join(log_dir, 'valid_logits.pt'))
    # with open(os.path.join(log_dir, 'valid_ids.json'), 'w') as f:
    #     json.dump(val_ids, f)
    
    correct = 0
    predictions = []
    ground_truth = []
    all_output = []
    for pid, data in tqdm(list(test_data.items())):
        emb = data['embedding']
        labels = data['label']
        ground_truth.append(labels)
        emb = emb.to(device)
        with torch.no_grad():
            output = model(emb).detach().cpu()
            all_output.append(output)
            # output_list = []
            # for i in range(100):
            #     output = model(emb).detach().cpu()
            #     output_list.append(output)
            # output = torch.stack(output_list).mean(dim=0)
        pred_idx = torch.argmax(output).item()
        predictions.append(all_labels[pred_idx])
        if all_labels[pred_idx] in labels:
            correct += 1
    print(f'Accuracy: {correct/len(test_data)}')
    all_output = torch.vstack(all_output).numpy()
    # all_output = torch.load('logs_mlp/train_mlp_CE_seed_0/RED_calibrated_test_logits.pt')
    # all_output = torch.load('logs_mlp/train_mlp_CE_seed_1/logits_test.pt')
    # all_output = all_output.numpy()
    print(f'all_output: {all_output.shape}')
    predictions = [[pred] for pred in predictions]
    print(f'Predicted labels: {predictions[:10]}')
    print(f'ground truth: {ground_truth[:10]}')
    f1, precision, recall, accuracy = get_f1_score(predictions, ground_truth, all_labels)
    print(f'F1: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}')
    get_precision_recall_curve_1D(all_output, ground_truth, all_labels)
    
def get_f1_score(predictions, ground_truth, all_labels):
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])
    pred_labels = mlb.transform(predictions)
    true_labels = mlb.transform(ground_truth)
    f1 = f1_score(true_labels, pred_labels, average='micro')
    precision = precision_score(true_labels, pred_labels, average='micro')
    recall = recall_score(true_labels, pred_labels, average='micro')
    accuracy = accuracy_score(true_labels, pred_labels)
    return f1, precision, recall, accuracy

def get_precision_recall_curve_1D(test_logits, ground_truth, all_labels):
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])
    true_labels = mlb.transform(ground_truth)
    precision, recall, _ = precision_recall_curve(true_labels.ravel(), test_logits.ravel())
    
    plt.plot(recall, precision, lw=2, label='Precision-Recall curve')
    plt.savefig('pr_curve.png')

def get_precision_recall_curve(test_logits, ground_truth, all_labels):
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])
    true_labels = mlb.transform(ground_truth)
    # For each class
    n_classes = len(all_labels)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], test_logits[:, i])
        average_precision[i] = average_precision_score(true_labels[:, i], test_logits[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        true_labels.ravel(), test_logits.ravel()
    )
    average_precision["micro"] = average_precision_score(true_labels, test_logits, average="micro")
    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
        prevalence_pos_label=Counter(true_labels.ravel())[1] / true_labels.size,
    )
    display.plot(plot_chance_level=True)
    _ = display.ax_.set_title("Micro-averaged over all classes")
    plt.savefig('pr_curve.png')

if __name__ == '__main__':
    for i in range(1):
        evaluation(f'logs_mlp/train_mlp_CE_seed_{i}')
