import pandas as pd
import torch
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn as nn

MEMBER_SEEDS = (8, 1008, 2008)
SAMPLE_SEEDS = (11008, 12008, 13008)

class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)


    def Adj_Generate(self,TF_set,direction=False, loop=False):

        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)


        for pos in self.train_set:

            tf = int(pos[0])  # 确保tf为整数
            target = int(pos[1])  # 确保target为整数

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()


        return adj



class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor





def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()


    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm




def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr



def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'hTFTarget':
        dic = {'CD14500':0.255,'CD4500':0.216,'CD8500':0.400,'B500':0.232,
               'AD_MG500':0.0194,'breast_spa500': 0.066,'IDC500': 0.066,'DCIS500': 0.061}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'CorrTFTarget':
        dic = {'Normal_T500': 0.150, 'Tumor_T500': 0.150, 'Tumor_B500': 0.150}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'RegNetwork':
        dic = {'Microglia500': 0.017}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError


def adjacency_from_directed(edges, num_nodes):
    positions = set()
    for tf, target in edges:
        positions.add((int(tf), int(target)))
        positions.add((int(target), int(tf)))
    if not positions:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    rows = np.fromiter((row for row, _ in positions), dtype=np.int64)
    cols = np.fromiter((column for _, column in positions), dtype=np.int64)
    data = np.ones(len(positions), dtype=np.float32)
    return sp.coo_matrix(
        (data, (rows, cols)), shape=(num_nodes, num_nodes)
    ).tocsr()


def support_mask_from_directed(edges, num_nodes):
    positions = {(int(target), int(tf)) for tf, target in edges}
    if not positions:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    rows = np.fromiter((row for row, _ in positions), dtype=np.int64)
    cols = np.fromiter((column for _, column in positions), dtype=np.int64)
    data = np.ones(len(positions), dtype=np.float32)
    return sp.coo_matrix(
        (data, (rows, cols)), shape=(num_nodes, num_nodes)
    ).tocsr()


def pair_set(frame):
    return set(map(tuple, frame[['TF', 'Target']].astype(int).to_numpy()))


def _negative_budget(frame):
    return {
        int(key): int(value)
        for key, value in frame[frame.Label == 0].groupby('TF').size().items()
    }


def _positive_budget(frame):
    return {
        int(key): int(value)
        for key, value in frame[frame.Label == 1].groupby('TF').size().items()
    }


def sample_uniform_training_sets(train, evaluation_pairs,
                                 all_known_positives, targets):
    if pair_set(train) & evaluation_pairs:
        raise AssertionError('UPNE requires zero Train/evaluation pair overlap')
    if _positive_budget(train) != _negative_budget(train):
        raise AssertionError('UPNE requires sparse per-TF 1:1 Train data')

    positives = train.loc[train.Label == 1, ['TF', 'Target']].to_numpy(np.int64)
    budgets = _negative_budget(train)
    target_values = tuple(sorted(map(int, targets)))
    outputs = []

    for sample_seed in SAMPLE_SEEDS:
        rng = np.random.RandomState(sample_seed)
        selected = []
        for tf in sorted(budgets):
            candidates = np.asarray([
                (tf, target)
                for target in target_values
                if tf != target
                and (tf, target) not in all_known_positives
                and (tf, target) not in evaluation_pairs
            ], dtype=np.int64)
            budget = budgets[tf]
            if len(candidates) < budget:
                raise RuntimeError('Insufficient legal U candidates for TF %d' % tf)
            selected.append(
                candidates[rng.choice(len(candidates), size=budget, replace=False)]
            )

        negatives = (np.concatenate(selected) if selected
                     else np.empty((0, 2), dtype=np.int64))
        outputs.append(np.concatenate([
            np.column_stack([positives, np.ones(len(positives), dtype=np.int64)]),
            np.column_stack([negatives, np.zeros(len(negatives), dtype=np.int64)]),
        ]))
    return outputs






























