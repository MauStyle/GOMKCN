import os
import time
import json
import torch

from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from concurrent.futures import ThreadPoolExecutor

class LoadData(object):
    def __init__(self, root='./dataset', dataset='ENZYMES', s_subgraph=16, k_hop=3, device='cpu'):
        self.root = root
        self.dataset = dataset
        self.s_subgraph = s_subgraph
        self.k_hop = k_hop
        self.device = device

    def load_dataset(self):
        graphs = TUDataset(self.root, name=self.dataset, use_node_attr=True)

        start_time = time.time()
        adjs, features, indexs, labels, indicators = [], [], [], [], []
        for graph in tqdm(graphs, desc='graphs'):
            adj, idx = self.extract_subgraph(graph)
            adjs.append(adj)
            indexs.append(idx)
            labels.append(graph.y.to(self.device))
            features.append(graph.x.to(self.device))
        stop_time = time.time()
        print(f"Time_Subgraph_Extraction: {stop_time - start_time:.4f} s")

        return adjs, features, indexs, labels

    def extract_subgraph(self, graph):
        def k_hop_subgraph(node, graph, k):
            row, col = graph.edge_index

            node_mask = row.new_empty(graph.num_nodes, dtype=torch.bool)
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

            subsets = [torch.tensor([node], device=row.device)]
            for _ in range(k):
                node_mask.fill_(False)
                node_mask[subsets[-1]] = True
                torch.index_select(node_mask, 0, row, out=edge_mask)
                subsets.append(col[edge_mask])
            subsets = torch.cat(subsets)
            seen = set()
            idx = []
            for element in subsets:
                value = element.item()
                if value not in seen:
                    seen.add(value)
                    idx.append(value)
            idx = torch.tensor(idx, dtype=subsets.dtype)

            if len(idx) < self.s_subgraph:
                padding_size = self.s_subgraph - len(idx)
                idx = torch.cat([idx, torch.full(size=(padding_size,), fill_value=-1)])
            else:
                idx = idx[:self.s_subgraph]

            return idx

        def k_hop_subgraph_parallel(graph, k=3, num_workers=8):
            idxs = []
            num_nodes = graph.num_nodes
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for node in range(num_nodes):
                    futures.append(executor.submit(k_hop_subgraph, node, graph, k))

                for future in futures:
                    idx = future.result()
                    idxs.append(idx)
            return idxs

        adj = torch.sparse_coo_tensor(indices=graph.edge_index,
                                      values=torch.ones(graph.edge_index.shape[1]),
                                      size=(graph.num_nodes, graph.num_nodes)).to_dense()

        adj = torch.cat((adj, torch.zeros(size=(1, adj.shape[1]))), dim=0)
        adj = torch.cat((adj, torch.zeros(size=(adj.shape[0], 1))), dim=1)
        idxs = torch.stack(k_hop_subgraph_parallel(graph=graph, k=self.k_hop, num_workers=6))
        row, col = idxs.unsqueeze(-1), idxs.unsqueeze(-2)
        adjs = adj[row, col]
        return adjs.to(self.device), idxs.to(self.device)

    def graph_splits(self, fold=10):
        folder = os.path.join(self.root, self.dataset, 'splits')
        file = os.path.join(folder, '{}_splits.json'.format(self.dataset))
        with open(file, 'rb') as f:
            splits = json.load(f)
            idx_train = splits[fold]['model_selection'][0]['train']
            idx_eval = splits[fold]['model_selection'][0]['validation']
            idx_test = splits[fold]['test']

        return idx_train, idx_eval, idx_test