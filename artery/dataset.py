import os
import pickle
import networkx as nx
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """
    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            #pad_pattern = torch.from_numpy(np.asfortranarray(pad_pattern))
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, 'constant', 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Keys mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        # elif type(inp[0]) == pyg.data.Data:
        #     ret = pyg.data.Batch.from_data_list(inp)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == tuple:
            ret = inp
        elif type(inp[0]) == nx.Graph:
            ret = inp
        elif type(inp[0]) == np.str_:
            ret = inp
        else:
            raise ValueError('Cannot handle type {}'.format(type(inp[0])))
        return ret

    ret = stack(data)

    ret['batch_size'] = len(data)

    for v in ret.values():
        if type(v) is list:
            ret['num_graphs'] = len(v)
            break

    return ret


def data_to_cuda(inputs):
    """
    Call cuda() on all tensor elements in inputs
    :param inputs: input list/dictionary
    :return: identical to inputs while all its elements are on cuda
    """
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
        inputs = inputs.cuda()
    else:
        raise TypeError('Unknown type of inputs: {}'.format(type(inputs)))
    return inputs


class ArteryDatasetMGM(Dataset):
    def __init__(self, dataset, samples, rand, num_graphs, cache=False, cache_path=".cache"):
        self.dataset = dataset
        self.samples = samples
        self.rand = rand
        self.cache = cache
        self.cache_path = cache_path
        self.num_graphs = num_graphs

    def __len__(self):
        return len(self.samples)

    def __build_graphs__(self, g: nx.Graph):
        A = nx.adjacency_matrix(g).todense()
        return np.array(A, dtype=np.float32)

    @staticmethod
    def generate_pair(gs, num_graphs, sample_ids, cache, cache_path):
        assert len(gs) == num_graphs

        if cache:
            if os.path.isfile(f"{cache_path}/{num_graphs}/{''.join(sample_ids)}.pkl"):
                ret_dict = pickle.load(open(f"{cache_path}/{num_graphs}/{''.join(sample_ids)}.pkl", "rb"))
                return ret_dict

        n0 = gs[0].number_of_nodes()
        As = []
        ns = []
        for i in range(num_graphs):
            A = np.array(nx.adjacency_matrix(gs[i]).todense(), dtype=np.float32)
            As.append(A)
            ns.append(n0)

        perm_mat_dict = {}
        for i in range(num_graphs):
            for j in range(num_graphs):
                if i<j:
                    perm_mat = np.zeros((n0, n0))
                    g_left, g_right = gs[i], gs[j]
                    for ii in range(n0):
                        for jj in range(n0):
                            if g_left.nodes()[ii]['data'].vessel_class == g_right.nodes()[jj]['data'].vessel_class:
                                perm_mat[ii, jj] = 1.0
                    perm_mat_dict[(i,j)] = perm_mat

        ret_dict = {'ns': [torch.tensor(x) for x in ns],
                    'gt_perm_mat': perm_mat_dict,
                    'As': [torch.tensor(x) for x in As],
                    'id_list': sample_ids}

        feats = []
        for i in range(num_graphs):
            n = gs[i].number_of_nodes()
            feat = np.stack([np.array(gs[i].nodes()[j]['data'].features, dtype=np.float32) for j in range(n)], axis=-1).T
            feats.append(feat)

        ret_dict['pos_features'] = [torch.tensor(x) for x in feats]

        if cache:
            pickle.dump(ret_dict, open(f"{cache_path}/{num_graphs}/{''.join(sample_ids)}.pkl", "wb"))

        return ret_dict

    def __getitem__(self, index):
        """
        return a set of graph with same number of nodes
        """
        self.rand = np.random.RandomState() # change random state everytime
        category_id = self.rand.randint(0, 6)
        all_sample_list = list(self.dataset.keys())
        sample_list = []
        for sample_name in all_sample_list:
            if sample_name.rfind(Artery.ARTERY_CATEGORY[category_id]) != -1:
                sample_list.append(sample_name)

        first_sample_idx = self.rand.randint(0, len(sample_list))
        g0 = self.dataset[sample_list[first_sample_idx]]['g']
        n0 = g0.number_of_nodes()

        # get the rest graph:
        eligible_idx = []
        eligible_idx.append(first_sample_idx)
        for i in range(len(sample_list)):
            if i!= first_sample_idx:
                gt = self.dataset[sample_list[i]]['g']
                nt = gt.number_of_nodes()
                # if g0_category == g1_category and n0 <= n1 and nx.diameter(g0) == nx.diameter(g1):
                if nt == n0 and nx.diameter(g0) == nx.diameter(gt):
                    eligible_idx.append(i)
        
        while len(eligible_idx) < self.num_graphs:
            last_element = eligible_idx[-1]
            eligible_idx.append(last_element)
        
        random_idx = self.rand.choice(len(eligible_idx), size=self.num_graphs, replace=False)
        sample_idx = [eligible_idx[i] for i in random_idx]
        random.shuffle(sample_idx)
        
        if self.cache:
            file_name = "".join([sample_list[sample_idx[k]] for k in range(self.num_graphs)])
            if os.path.isfile(f"{self.cache_path}/{self.num_graphs}/{file_name}.pkl"):
                ret_dict = pickle.load(open(f"{self.cache_path}/{self.num_graphs}/{file_name}.pkl", "rb"))
                return ret_dict
        
        ###################################################################
        gs = []
        ns = []
        As = []
        for i in range(self.num_graphs):
            g = self.dataset[sample_list[sample_idx[i]]]['g']
            A = self.__build_graphs__(g)
            n = g.number_of_nodes()
            As.append(A)
            gs.append(g)
            ns.append(n)

        perm_mat_dict = {}
        for i in range(self.num_graphs):
            for j in range(self.num_graphs):
                if i<j:
                    perm_mat = np.zeros((n0, n0))
                    g_left = self.dataset[sample_list[sample_idx[i]]]['g']
                    g_right = self.dataset[sample_list[sample_idx[j]]]['g']
                    for ii in range(n0):
                        for jj in range(n0):
                            if g_left.nodes()[ii]['data'].vessel_class == g_right.nodes()[jj]['data'].vessel_class:
                                perm_mat[ii, jj] = 1.0
                    perm_mat_dict[(i,j)] = perm_mat

        ret_dict = {'ns': [torch.tensor(x) for x in ns],
                    'gt_perm_mat': perm_mat_dict,
                    'As': [torch.tensor(x) for x in As],
                    'id_list': [sample_list[sample_idx[k]] for k in range(self.num_graphs)],}

        feats = []
        for i in range(self.num_graphs):
            n = gs[i].number_of_nodes()
            feat = np.stack([np.array(gs[i].nodes()[j]['data'].features, dtype=np.float32) for j in range(n)], axis=-1).T
            feats.append(feat)

        ret_dict['pos_features'] = [torch.tensor(x) for x in feats]

        if self.cache:
            file_name = "".join([sample_list[sample_idx[k]] for k in range(self.num_graphs)])
            pickle.dump(ret_dict, open(f"{self.cache_path}/{self.num_graphs}/{file_name}.pkl", "wb"))

        return ret_dict
    