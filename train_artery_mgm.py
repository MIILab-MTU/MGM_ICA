import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pickle

from src.loss_func import *
from artery.dataset import ArteryDatasetMGM
from data.artery_utils import *

from artery.dataset import collate_fn
from artery.models.mgm_model import MGM_Model, data_to_cuda

class MGM_PCA_Trainer(object):
    def __init__(self, params, device):
        self.params = params
        self.rand = np.random.RandomState(seed=params.seed)
        self.device = device
        self.__init_model__()
    
    def __init_model__(self):
        pca_params = {"FEATURE_CHANNEL": self.params.feature_channel, 
                      "SK_ITER_NUM": self.params.sk_iter_num, "SK_EPSILON": self.params.sk_epsilon, "SK_TAU": self.params.sk_tau,
                      "CROSS_ITER": self.params.cross_iter, "CROSS_ITER_NUM": self.params.cross_iter_num}

        gnn_params = {"GNN_FEAT": [self.params.gnn_feat]*self.params.gnn_layers, "GNN_LAYER": self.params.gnn_layers}
        mgm_params = {"NUM_GRAPHS": self.params.num_graphs}

        train_params = {"MOMENTUM": self.params.momentum, "OPTIMIZER": "adam", "EPOCH_ITERS": self.params.n_iters,
                        "LR_DECAY": 0.1, "LR_STEP": [50, 100, 200], 'LR': self.params.lr, "LOSS_FUNC": "ce"}
        self.pca_params = pca_params
        self.gnn_params = gnn_params
        self.train_params = train_params
        
        self.model = MGM_Model(pca_params, gnn_params, mgm_params, self.device).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=train_params['LR'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_params['LR_STEP'], gamma=train_params['LR_DECAY'])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CrossEntropyLoss()

    def test(self):
        self.model.eval()

        g0 = pickle.load(open("data/artery_with_feature_mgm2/1.pkl", "rb"))
        g1 = pickle.load(open("data/artery_with_feature_mgm2/2.pkl", "rb"))
        g2 = pickle.load(open("data/artery_with_feature_mgm2/3.pkl", "rb"))
        gs = [g0, g1, g2]
        inputs = ArteryDatasetMGM.generate_pair(gs, self.params.num_graphs, ["1", "2", "3"], False, "")
        inputs = data_to_cuda(inputs)
        inputs = collate_fn([inputs])
        outputs, _ = self.model(inputs)
        # iterate each pair in MGM
        for j in range(1, self.params.num_graphs):
            mappings = {}
            perm_mat = outputs['perm_mat_list'][(0, j)][0].detach().cpu().numpy()
            gt_perm_mat = outputs['gt_perm_mat'][(0, j)][0].detach().cpu().numpy()
            # iterate each node in paired graphs
            for k in range(perm_mat.shape[0]):
                gr_idx = np.where(perm_mat[k]==1)[0][0]
                mappings[g0.nodes[k]['data'].vessel_class] = gs[j].nodes[gr_idx]['data'].vessel_class

            data_row = {"test_sample": "0", "template_sample": str(j), "category": "LAO_CRA", "n": g0.number_of_nodes()}

            matched, unmatched = 0, 0
            for key in mappings:
                data_row[key] = mappings[key]
                if key == mappings[key]:
                    matched += 1
                else:
                    unmatched += 1
            data_row["matched"] = matched
            data_row["unmatched"] = unmatched
            print(data_row)
                        
    def __restore__(self):
        print("MGM_Model_Trainer.__restore__")
        self.model.load_state_dict(torch.load(f"{self.params.exp}/saved_models/model.pt"))
        print(f"MGM_Model_Trainer.__restore__, model at {self.params.exp}/saved_models/model.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--feature_channel', type=int, default=121)
    parser.add_argument('--sk_iter_num', type=int, default=10)
    parser.add_argument('--sk_epsilon', type=float, default=1e-10)
    parser.add_argument('--sk_tau', type=float, default=0.05)
    parser.add_argument('--sk_emb', type=bool, default=True)
    parser.add_argument('--cross_iter', type=bool, default=True)
    parser.add_argument('--cross_iter_num', type=int, default=3)
    parser.add_argument('--gnn_feat', type=int, default=256)
    parser.add_argument('--gnn_layers', type=int, default=3)
    parser.add_argument('--num_graphs', type=int, default=3)


    # data parameters
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_path', type=str, default='data/artery_with_feature_mgm')
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--cv_max', type=int, default=5)
    parser.add_argument('--template_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=6)
    parser.add_argument('--model_path', type=str, default='')

    # training parameters
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_iters', type=int, default=1001)
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1.0e-5)
    parser.add_argument('--loss_func', type=str, default='ce')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--acceptance_rate', type=float, default=0.9)
    parser.add_argument('--tolerance', type=int, default=1000)
    

    # exp 
    parser.add_argument('--exp', type=str, default="experiments_mgm/NG3_G256_GL3/CV1")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    trainer = MGM_PCA_Trainer(args, device)
    trainer.__restore__()
    trainer.test()
    