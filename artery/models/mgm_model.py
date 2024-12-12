import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import math

from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.ilp import ILP_solver
from src.gconv import Siamese_Gconv, Siamese_SelfAttention
from torch.nn.parameter import Parameter
from itertools import combinations




def data_to_cuda(inputs, device="cuda:0"):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
    device = torch.device(device)
    if type(inputs) is list:
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is tuple:
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            inputs[i] = data_to_cuda(x)
    elif type(inputs) is dict:
        for key in inputs:
            inputs[key] = data_to_cuda(inputs[key])
    elif type(inputs) in [str, int, float, nx.Graph, np.str_]:
        inputs = inputs
    elif type(inputs) in [torch.Tensor]:
        inputs = inputs.to(device)
    elif type(inputs) in [np.ndarray]:
        inputs = torch.tensor(inputs).to(device)
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs


class MGM_Model(nn.Module):

    def __init__(self, pca_params, gnn_params, mgm_params, device):
        super(MGM_Model, self).__init__()
        self.pca_params = pca_params
        self.gnn_params = gnn_params
        self.mgm_params = mgm_params
        self.device = device

        self.sinkhorn = Sinkhorn(max_iter=pca_params['SK_ITER_NUM'], epsilon=pca_params['SK_EPSILON'], tau=pca_params['SK_TAU'])
        self.batch_size = 1
        self.pca_gm = PCA_GM(self.pca_params, self.gnn_params)


    def forward(self, data_dict, **kwargs):
        n_nodes, n_graphs = data_dict['ns'][0], len(data_dict['ns'])
        total_n_nodes = data_dict['ns'][0]*n_graphs

        joint_S = torch.zeros(self.batch_size, total_n_nodes, total_n_nodes, device=self.device)
        joint_S_diag = torch.diagonal(joint_S, dim1=1, dim2=2)
        joint_S_diag += 1

        for src_idx, tgt_idx in combinations(list(range(n_graphs)), 2):
            src, tgt = data_dict['pos_features'][src_idx], data_dict['pos_features'][tgt_idx]
            ns_src, ns_tgt = data_dict['ns'][src_idx], data_dict['ns'][tgt_idx]
            A_src, A_tgt = data_dict['As'][src_idx], data_dict['As'][tgt_idx]
            data_dict_pca = {'pos_features': (src, tgt), 'ns': (ns_src, ns_tgt), 'As': (A_src, A_tgt)}
            data_dict_return, nan_encountered = self.pca_gm(data_dict_pca)
            s, perm_mat = data_dict_return['ds_mat'], data_dict_return['perm_mat']

            if src_idx > tgt_idx:
                joint_S[:, tgt_idx*n_nodes:(tgt_idx+1)*n_nodes, src_idx*n_nodes:(src_idx+1)*n_nodes] += s.transpose(1, 2)
            else:
                joint_S[:, src_idx*n_nodes:(src_idx+1)*n_nodes, tgt_idx*n_nodes:(tgt_idx+1)*n_nodes] += s

            if nan_encountered:
                break

        matching_s = []
        for b in range(self.batch_size):
            # L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L')
            e, v = torch.symeig(joint_S[b], eigenvectors=True) # L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')
            topargs = torch.argsort(torch.abs(e), descending=True)[:n_nodes]
            diff = e[topargs[:-1]] - e[topargs[1:]]
            if torch.min(torch.abs(diff)) > 1e-4:
                matching_s.append(n_graphs * torch.mm(v[:, topargs], v[:, topargs].transpose(0, 1)))
            else:
                matching_s.append(joint_S[b])

        matching_s = torch.stack(matching_s, dim=0)

        pred_s = {}
        pred_x = {}

        for idx1, idx2 in combinations(range(n_graphs), 2):
            s = matching_s[:, idx1*n_nodes:(idx1+1)*n_nodes, idx2*n_nodes:(idx2+1)*n_nodes]
            s = self.sinkhorn(s)

            pred_s[(idx1, idx2)] = s
            pred_x[(idx1, idx2)] = hungarian(s)

        data_dict.update({'ds_mat_list': pred_s, 'perm_mat_list': pred_x})
        return data_dict, nan_encountered



class PCA_GM(nn.Module):
    def __init__(self, pca_params, gnn_params):
        super(PCA_GM, self).__init__()
        self.pca_params = pca_params
        self.gnn_params = gnn_params

        self.sinkhorn = Sinkhorn(max_iter=pca_params['SK_ITER_NUM'], epsilon=pca_params['SK_EPSILON'], tau=pca_params['SK_TAU'])
        self.gnn_layer = gnn_params['GNN_LAYER']

        for i in range(self.gnn_layer):
            if i == 0:
                gnn_layer = Siamese_Gconv(pca_params['FEATURE_CHANNEL'], gnn_params['GNN_FEAT'][i])
            else:
                gnn_layer = Siamese_Gconv(gnn_params['GNN_FEAT'][i-1], gnn_params['GNN_FEAT'][i])
                    
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(gnn_params['GNN_FEAT'][i]))
            if i == self.gnn_layer - 2:  # only second last layer will have cross-graph module
                self.add_module('cross_graph_{}'.format(i), nn.Linear(gnn_params['GNN_FEAT'][i] * 2, gnn_params['GNN_FEAT'][i]))

        self.cross_iter = pca_params['CROSS_ITER']
        self.cross_iter_num = pca_params['CROSS_ITER_NUM']
        
    def forward(self, data_dict, **kwargs):
        # synthetic data
        src, tgt = data_dict['pos_features']
        ns_src, ns_tgt = data_dict['ns']
        A_src, A_tgt = data_dict['As']

        emb1 = src # concat([batch, feat_dim, n_node], [batch, feat_dim, n_node])
        emb2 = tgt
        ss = []
        if not self.cross_iter:
            # Vanilla PCA-GM
            for i in range(self.gnn_layer):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2]) 
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2) # s [batch, n_node, n_node]
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)

                ss.append(s)

                if i == self.gnn_layer - 2:
                    cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                    new_emb1 = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                    new_emb2 = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                    emb1 = new_emb1
                    emb2 = new_emb2
        else:
            # IPCA-GM
            for i in range(self.gnn_layer - 1):
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])

            emb1_0, emb2_0 = emb1, emb2
            s = torch.zeros(emb1.shape[0], emb1.shape[1], emb2.shape[1], device=emb1.device)

            for x in range(self.cross_iter_num):
                i = self.gnn_layer - 2
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1 = cross_graph(torch.cat((emb1_0, torch.bmm(s, emb2_0)), dim=-1))
                emb2 = cross_graph(torch.cat((emb2_0, torch.bmm(s.transpose(1, 2), emb1_0)), dim=-1))

                i = self.gnn_layer - 1
                gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
                emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
                affinity = getattr(self, 'affinity_{}'.format(i))
                s = affinity(emb1, emb2)
                s = self.sinkhorn(s, ns_src, ns_tgt, dummy_row=True)
                ss.append(s)

        if torch.sum(torch.isnan(ss[-1])) > 0:
            # print(ss[-1])
            print("[!] NAN encountered")
            data_dict.update({'ds_mat': torch.nan_to_num(ss[-1], 0.), 'perm_mat': hungarian(torch.nan_to_num(ss[-1], 0), ns_src, ns_tgt)})
            return data_dict, True
        else:
            data_dict.update({'ds_mat': ss[-1], 'perm_mat': hungarian(ss[-1], ns_src, ns_tgt)})

            return data_dict, False



class Affinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix from feature space.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, self.A)
        #M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M