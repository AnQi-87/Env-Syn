import os
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, confusion_matrix, r2_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.metrics import f1_score

from utils_test import TestbedDataset, save_AUCs
from torch_geometric.loader import DataLoader
from creat_data_DC import creat_data
from model import EnvSyn
import logging

logging.basicConfig(
    filename='Env-Syn.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='w'
)
logger = logging.getLogger(__name__)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def train_one_epoch(model, device, loader1, loader2, optimizer, task):
    model.train()
    total_loss = 0.0
    for (data1, data2) in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)

        y = data1.y.view(-1).to(device) 
        optimizer.zero_grad()
        pred = model(data1, data2, sup_labels=None)

        loss_task = F.binary_cross_entropy(pred, y)
        
        loss = loss_task
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader1))

@torch.no_grad()
def evaluate(model, device, loader1, loader2, task):
    model.eval()
    all_labels = []
    all_scores = []
    all_preds = []
    for (data1, data2) in zip(loader1, loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data1.y.view(-1).to(device)
        out = model(data1, data2, sup_labels=None)
        
        score = out.detach().cpu().numpy()          
        pred_label = (out.detach().cpu().numpy() > 0.5).astype(int)
        all_scores.extend(score.tolist())
        all_preds.extend(pred_label.tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        

    
    T = np.array(all_labels).flatten()
    S = np.array(all_scores).flatten()
    Y = np.array(all_preds).flatten()
    AUC = roc_auc_score(T, S)
    precision, recall, _ = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    F1 = f1_score(T, Y, zero_division=0)

    PREC = precision_score(T, Y, zero_division=0)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    RECALL = recall_score(T, Y)
    return {
        'AUC': round(AUC, 4),
        'PR_AUC': round(PR_AUC, 4),
        'ACC': round(ACC, 4),
        'BACC': round(BACC, 4),
        'PREC': round(PREC, 4),
        'F1': round(F1, 4),
        'KAPPA': round(KAPPA, 4),
        'RECALL': round(RECALL, 4)
    }, (T, S, Y)
    
import re
import numpy as np
import os
import logging

def summarize_metrics(metric_files, folder_path, result_name, dataset):
    rows_per_fold = []
    for mf in metric_files:
        try:
            with open(mf, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logging.warning(f'Could not read {mf}: {e}')
            continue

        last_row = None
        for line in reversed(lines):
            line = line.strip()
            if line == '':
                continue
            if re.match(r'^\d+\t', line):
                last_row = line
                break
        if last_row is None:
            logging.warning(f'No numeric epoch row found in {mf}. Skipping.')
            continue

        parts = last_row.split('\t')
        try:
            vals = [float(x) for x in parts[1:]] 
            rows_per_fold.append(vals)
        except Exception as e:
            logging.warning(f'Failed to parse numeric values in {mf}: {last_row} -- {e}')
            continue

    if len(rows_per_fold) == 0:
        logging.warning('No valid numeric rows found. Skip summary.')
        return

    arr = np.array(rows_per_fold) 
    means = arr.mean(axis=0)
    stds  = arr.std(axis=0, ddof=0) if arr.shape[0] > 1 else np.zeros_like(means)

    
    with open(metric_files[0], 'r') as f:
        header_line = f.readline().strip()
    metric_names = header_line.split('\t')[1:]

    summary_lines = []
    summary_lines.append('Summary (last-row of each fold):')
    for name, m, s in zip(metric_names, means, stds):
        summary_lines.append(f'{name}: {m:.4f} ± {s:.4f}')

    summary_text = '\n'.join(summary_lines)
    print(summary_text)
    logging.info(summary_text)

    
    summary_file = os.path.join(folder_path, f'{result_name}--summary_lastrow_{dataset}.txt')
    with open(summary_file, 'w') as f:
        f.write(summary_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cellfile', type=str, default='data/cell_features_954.csv')
    parser.add_argument('--smiles', type=str, default='data/smiles.csv')
    parser.add_argument('--datafile', type=str, default='data/new_labels_0_10.csv')
    parser.add_argument('--dataset_name', type=str, default='new_labels_0_10')
    parser.add_argument('--MoLFormer', type=str, default='data/processed/MoLFormer.npy', help='Precomputed MoLFormer features file')

    parser.add_argument('--task', type=str, choices=['cls','reg'], default='cls', help='Classification or regression')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--precompute_mol', action='store_true', help='Precompute MoLFormer features if not available')

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--mol_dim', type=int, default=768)
    parser.add_argument('--cell_dim', type=int, default=954)
    parser.add_argument('--node_feat_dim', type=int, default=78)
    args = parser.parse_args()
    logger.info(f'args:{args}')
    
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    logger.info('Processing source files...')
    drug1, drug2, cell, label, cell_features, mol_dict, drug2id = creat_data(
        args.datafile, args.smiles, args.cellfile, 
        MoLFormer_file=args.MoLFormer,
        precompute_mol=args.precompute_mol
    )
    logger.info('Successfully extracted features from source files!')

    logger.info('Loading data...')
    dataset = args.dataset_name
    drug1_data = TestbedDataset(dataset=dataset + '_drug1', xd=drug1, xt=cell, y=label,
                               xt_featrue=cell_features,
                              mol_dict=mol_dict, mol_dim=args.mol_dim, drug2id=drug2id)
    drug2_data = TestbedDataset(dataset=dataset + '_drug2', xd=drug2, xt=cell, y=label,
                               xt_featrue=cell_features,
                              mol_dict=mol_dict, mol_dim=args.mol_dim, drug2id=drug2id)
    logger.info('Data loading complete')

    lenth = len(drug1_data)
    pot = int(lenth / 5)
    logger.info(f'lenth: {lenth}')
    logger.info(f'pot, {pot}')

    random_idx = list(range(lenth))
    random.shuffle(random_idx)

    result_name = f'Env-Syn_{args.task}'
    folder_path = './result/' + result_name
    os.makedirs(folder_path, exist_ok=True)

    best_scores = []
    metric_files = []

    for i in range(5):
        test_num = random_idx[pot * i:pot * (i + 1)]
        train_num = random_idx[:pot * i] + random_idx[pot * (i + 1):]

        drug1_train = drug1_data[train_num]
        drug1_test  = drug1_data[test_num]
        drug2_train = drug2_data[train_num]
        drug2_test  = drug2_data[test_num]

        loader1_train = DataLoader(drug1_train, batch_size=args.batch_size, shuffle=None)
        loader1_test  = DataLoader(drug1_test,  batch_size=args.batch_size, shuffle=None)
        loader2_train = DataLoader(drug2_train, batch_size=args.batch_size, shuffle=None)
        loader2_test  = DataLoader(drug2_test,  batch_size=args.batch_size, shuffle=None)

        model = EnvSyn(
            num_node_features=args.node_feat_dim,
            mol_dim=args.mol_dim,
            hidden_dim=args.hidden_dim,
            cell_dim=args.cell_dim,
            task=args.task, 
            semantic_tokens=8
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        metric_file = os.path.join(folder_path, f'{result_name}_{i}--metrics--{dataset}_{now}.txt')
        metric_files.append(metric_file)
        header = 'Epoch\tAUC\tPR_AUC\tACC\tBACC\tPREC\tF1\tKAPPA\tRECALL'
        with open(metric_file, 'w') as f:
            f.write(header + '\n')

        best_key = 'AUC'
        best_val = 0

        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, device, loader1_train, loader2_train, optimizer, args.task)
            metrics_dict, (T, S, Y) = evaluate(model, device, loader1_test, loader2_test, args.task)

            row = [epoch, metrics_dict['AUC'], metrics_dict['PR_AUC'], metrics_dict['ACC'],
                    metrics_dict['BACC'], metrics_dict['PREC'], metrics_dict['F1'],
                    metrics_dict['KAPPA'], metrics_dict['RECALL']]

            if metrics_dict['AUC'] > best_val:
                logger.info(f'Epoch {epoch}:{metrics_dict}')
                best_val = metrics_dict['AUC']
                save_AUCs(row, metric_file)
            else:
                if epoch % 10 == 0 or epoch == 1:
                    logger.info(f'Epoch {epoch}:{metrics_dict}')

        best_scores.append(best_val)
        save_AUCs(f'best_{best_key}: {best_val}', metric_file)

    logger.info('Cross-validated best scores: %s', best_scores)
    logger.info('Mean %s: %.4f ± %.4f', best_key, np.mean(best_scores), np.std(best_scores))

    summarize_metrics(metric_files, folder_path, result_name, dataset)

if __name__ == '__main__':
    main()
