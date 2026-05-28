import os
import sys
import time
import json
import pickle
import torch
from torch_geometric.transforms import OneHotDegree, LocalDegreeProfile
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from concurrent.futures import ThreadPoolExecutor

class LoadData(object):
    def __init__(self, root='./dataset', dataset='DD', size=16, hop=3, device='cpu'):
        if dataset not in ["DD", "NCI1", "PROTEINS_full", "ENZYMES", "MUTAG", "IMDB-BINARY", "IMDB-MULTI", "REDDIT-BINARY", "COLLAB"]:
            sys.exit()
        self.root = root
        self.dataset = dataset
        self.size = size
        self.hop = hop
        self.device = device

    def load_dataset(self):
        folder = os.path.join(self.root, self.dataset, 'subgraph')
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, 'hop_{}_size_{}.pkl'.format(self.hop, self.size))
        if os.path.exists(file):
            with open(file, 'rb') as f:
                dict = pickle.load(f)
            dim = dict['dim']
            classes = dict['classes']
            adjs = [torch.tensor(arr) for arr in dict['adjs']]
            features = [torch.tensor(arr) for arr in dict['features']]
            indexs = [torch.tensor(arr) for arr in dict['indexs']]
            labels = [torch.tensor(arr) for arr in dict['labels']]
        else:
            if self.dataset in ["ENZYMES", "PROTEINS_full", "DD", "NCI1", "MUTAG"]:
                graphs = TUDataset(self.root, name=self.dataset, use_node_attr=True)
            elif self.dataset == "IMDB-BINARY":
                graphs = TUDataset(self.root, name=self.dataset, transform=OneHotDegree(135))
            elif self.dataset == "IMDB-MULTI":
                graphs = TUDataset(self.root, name=self.dataset, transform=OneHotDegree(89))
            else:
                graphs = TUDataset(self.root, name=self.dataset, transform=LocalDegreeProfile())
            dim, classes = graphs.num_node_features, graphs.num_classes
            start_time = time.time()
            adjs, features, indexs, labels, indicators = [], [], [], [], []
            for graph in tqdm(graphs, desc='graphs'):
                adj, idx = self.extract_subgraph(graph)
                adjs.append(adj)
                indexs.append(idx)
                labels.append(graph.y)
                features.append(graph.x)
            stop_time = time.time()
            print(f"Time of Subgraph Extraction: {stop_time - start_time:.4f} s")
            dict = {'dim': graphs.num_node_features,
                    'classes': graphs.num_classes,
                    'adjs': [tensor.numpy() for tensor in adjs],
                    'features': [tensor.numpy() for tensor in features],
                    'indexs': [tensor.numpy() for tensor in indexs],
                    'labels': [tensor.numpy() for tensor in labels]}
            with open(file, 'wb') as f:
                pickle.dump(dict, f)
        return dim, classes, adjs, features, labels, indexs

    def extract_subgraph(self, graph):
        def multi_hop_subgraph(node, graph, k):
            row, col = graph.edge_index
            node_mask = row.new_empty(graph.num_nodes, dtype=torch.bool)
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            subsets = [torch.tensor(data=[node], device=row.device)]
            for _ in range(k):
                node_mask.fill_(False)
                node_mask[subsets[-1]] = True
                torch.index_select(input=node_mask, dim=0, index=row, out=edge_mask)
                subsets.append(col[edge_mask])
            subsets = torch.cat(subsets)
            seen, idx = set(), []
            for element in subsets:
                value = element.item()
                if value not in seen:
                    seen.add(value)
                    idx.append(value)
                if len(idx) >= self.size:
                    break
            idx = torch.tensor(idx, dtype=subsets.dtype)
            if len(idx) < self.size:
                padding_size = self.size - len(idx)
                idx = torch.cat([idx, torch.full(size=(padding_size,), fill_value=-1)])
            else:
                idx = idx[:self.size]
            return idx

        def multi_hop_subgraph_parallel(graph, k=3, num_workers=8):
            idxs = []
            num_nodes = graph.num_nodes
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for node in range(num_nodes):
                    futures.append(executor.submit(multi_hop_subgraph, node, graph, k))

                for future in futures:
                    idx = future.result()
                    idxs.append(idx)
            return idxs
        adj = torch.sparse_coo_tensor(indices=graph.edge_index,
                                      values=torch.ones(graph.edge_index.shape[1]),
                                      size=(graph.num_nodes, graph.num_nodes)).to_dense()
        adj = torch.cat((adj, torch.zeros(size=(1, adj.shape[1]))), dim=0)
        adj = torch.cat((adj, torch.zeros(size=(adj.shape[0], 1))), dim=1)
        idxs = torch.stack(multi_hop_subgraph_parallel(graph=graph, k=self.hop, num_workers=6))
        row, col = idxs.unsqueeze(-1), idxs.unsqueeze(-2)
        adjs = adj[row, col]
        return adjs, idxs

    def graph_split(self, fold=10):
        folder = os.path.join(self.root, self.dataset, 'splits')
        file = os.path.join(folder, '{}_splits.json'.format(self.dataset))
        with open(file, 'rb') as f:
            splits = json.load(f)
            idx_train = splits[fold]['model_selection'][0]['train']
            idx_eval = splits[fold]['model_selection'][0]['validation']
            idx_test = splits[fold]['test']
        idx_dict = {'train': idx_train, 'eval': idx_eval, 'test': idx_test}
        return idx_dict

    def get_batchs(self, A_all, F_all, Y_all, index, idx_dict, batch_size):
        dict = {}
        for mode in ['train', 'eval', 'test']:
            idx = idx_dict[mode]
            A_batchs, F_batchs, Y_batchs, index_batchs, indicator_batchs, pos = [], [], [], [], [], 0
            while pos < len(idx):
                idx_batch = idx[pos: pos + batch_size]
                pos += len(idx_batch)
                A_batch, F_batch, Y_batch, index_batch, indicator_batch, total = [], [], [], [], [], 0
                for i, j in enumerate(idx_batch):
                    A_batch.append(A_all[j])
                    F_batch.append(F_all[j])
                    Y_batch.append(Y_all[j])
                    index_batch.append(torch.where(index[j] != -1, index[j] + total, index[j]))
                    indicator_batch.append(torch.full(size=(A_all[j].shape[0],), fill_value=i))
                    total += A_all[j].shape[0]
                A_batchs.append(torch.vstack(A_batch).to(self.device))
                F_batchs.append(torch.vstack(F_batch).to(self.device))
                Y_batchs.append(torch.hstack(Y_batch).to(self.device))
                index_batchs.append(torch.vstack(index_batch).to(self.device))
                indicator_batchs.append(torch.hstack(indicator_batch).to(self.device))
            dict[mode] = {"adj": A_batchs,
                          "feature": F_batchs,
                          "label": Y_batchs,
                          "index": index_batchs,
                          "indicator": indicator_batchs}
        return dict