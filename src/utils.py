import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score, precision_score, recall_score


def dist2prob(train_dist, dist):
    '''
    Transform contrastive learning distance into probability value between 0 and 1
    '''

    max_dist = train_dist.max()
    prob = (max_dist - dist)/max_dist

    return prob


def calibrate_fdr(alpha, delta, cal_prob, cal_labels, N=100):
    '''
    FDR control: code from the below paper:
    A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification 
    https://arxiv.org/abs/2107.07511
    '''

    # calibratation set size
    n = cal_prob.shape[0]

    # calculate risks and HB p-value
    lambdas = torch.linspace(0,1,N)
    losses = torch.zeros((n,N))
    for i in range(n):
        sigmoids = cal_prob[i]
        for j in range(N):
            T = sigmoids > lambdas[j] # prediction set induced by lambda_j
            set_size = T.float().sum()
            if set_size != 0:
                losses[i,j] = 1 - (T[cal_labels[i,:]] == True).float().sum()/set_size # false discovery rate (FDR)
    risk = losses.mean(dim=0)
    pvals = torch.exp(-2*n*(torch.relu(alpha-risk)**2))

    # Fixed-sequence test starting at from lambda[-1] to lambda[0]
    below_delta = (pvals <= delta).float()
    valid = torch.tensor([(below_delta[j:].mean() == 1) for j in range(N)])
    lambda_hats = lambdas[valid] # rejection set
  
    return lambda_hats


def predict(prob, labels, lambda_hat, idx2prot, idx2ec):
    '''
    Make predictions using the calibrated model parameter lambda_hat
    '''

    n = prob.shape[0]
    T = prob >= lambda_hat # prediction set
    
    set_sizes = T.float().sum(axis=1)
    # average_set_size = set_sizes.mean()
    # coverage = (set_sizes > 0).sum() / set_sizes.shape[0]
    
    risks = []
    for i in range(n):
        if set_sizes[i] != 0:
            risk = 1 - (T[i][labels[i,:]] == True).float().sum()/set_sizes[i]
        else:
            risk = torch.tensor(0)    
        risks.append(risk)

    average_risk = torch.mean(torch.tensor(risks)).item()

    conf_prob = prob.clone()
    conf_prob[conf_prob<lambda_hat]=0
    
    ndcg = ndcg_score(labels.numpy(), conf_prob.numpy())
    
    conf_prob[conf_prob>0]=1
    precision = precision_score(labels.numpy(), conf_prob.numpy(), average='samples', zero_division=1)
    recall = recall_score(labels.numpy(), conf_prob.numpy(), average='samples')

    # visualize predictions and ground truth labels
    df = pd.DataFrame([], columns=['PDB-chain', 'Ground truth', 'CPEC prediction'])
    for i in range(n):
        name = idx2prot[i]
        gt = [idx2ec[j] for j in np.where(labels[i])[0]]
        pred = [idx2ec[j] for j in np.where(T[i])[0]]

        tmp_df = pd.DataFrame({
            'PDB-chain':[name],
            'Ground truth':[gt],
            'CPEC prediction':[pred]
        })
        df = pd.concat([df, tmp_df], ignore_index=True)

    return average_risk, precision, recall, ndcg, df
