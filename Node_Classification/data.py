import os
import sys
import time
import torch
import pickle
import torch_geometric.transforms as T
from tqdm import tqdm
from torch_geometric.datasets import Actor, Planetoid, WikipediaNetwork
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class LoadData(object):
    def __init__(self, root='./dataset', dataset='Cora', hop=3, size=8):
        self.root = root
        self.dataset = dataset
        self.size = size
        self.hop = hop

        transform = T.Compose([T.ToUndirected(), T.RemoveSelfLoops()])
        if self.dataset in ["Cora", "CiteSeer", "PubMed"]:
            graph = Planetoid(root=self.root, name=self.dataset, transform=transform)[0]
        elif self.dataset in ["chameleon", "squirrel"]:
            graph = WikipediaNetwork(root=self.root, name=self.dataset, transform=transform)[0]
        elif self.dataset in ["actor"]:
            graph = Actor(root=os.path.join(self.root, self.dataset), transform=transform)[0]
        else:
            sys.exit()

        self.graph = graph
        self.n_nodes = graph.num_nodes
        self.dim = graph.x.shape[1]
        self.feature = graph.x
        self.label = graph.y

    def extract_subgraph(self, graph):
        def multi_hop_subgraph(node, graph, k):
            row, col = graph.edge_index
            node_mask = row.new_empty(graph.num_nodes, dtype=torch.bool)
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            subsets = [torch.tensor([node])]
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

            if len(idx) < self.size:
                padding_size = self.size - len(idx)
                idx = torch.cat([idx, torch.full(size=(padding_size,), fill_value=graph.num_nodes)])
            else:
                idx = idx[:self.size]

            return idx

        def multi_hop_subgraph_parallel(graph, k=3, num_workers=8):
            idxs = []
            num_nodes = graph.num_nodes
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for node in range(num_nodes):
                    futures.append(executor.submit(multi_hop_subgraph, node, graph, k))
                for future in tqdm(futures, desc="Processing nodes", total=num_nodes):
                    idx = future.result()
                    idxs.append(idx)
            stop_time = time.time()
            print(f"Time of Subgraph Extraction: {stop_time - start_time:.4f} s")
            return idxs

        adj = torch.sparse_coo_tensor(indices=graph.edge_index,
                                      values=torch.ones(graph.edge_index.shape[1]),
                                      size=(self.n_nodes, self.n_nodes)).to_dense()
        adj = torch.cat((adj, torch.zeros(size=(1, adj.shape[1]))), dim=0)
        adj = torch.cat((adj, torch.zeros(size=(adj.shape[0], 1))), dim=1)
        idxs = torch.stack(multi_hop_subgraph_parallel(self.graph, k=self.hop, num_workers=6))
        row, col = idxs.unsqueeze(-1), idxs.unsqueeze(-2)
        adjs = adj[row, col]
        return idxs, adjs

    def get_data(self, graph):
        folder = os.path.join(self.root, self.dataset, 'subgraph')
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, f'hop_{self.hop}_size_{self.size}.pkl')
        if os.path.exists(file):
            with open(file, 'rb') as f:
                dict = pickle.load(f)
            idxs, adjs = dict['idxs'], dict['adjs']
        else:
            idxs, adjs = self.extract_subgraph(graph)
            dict = {'idxs': idxs, 'adjs': adjs}
            with open(file, 'wb') as f:
                pickle.dump(dict, f)
        return self.dim, idxs, adjs, self.feature, self.label

    def get_split(self, label, seed):
        train_eval_idx, test_idx = train_test_split(range(len(label)),
                                                    test_size=0.2,
                                                    stratify=label,
                                                    random_state=seed)
        train_idx, eval_idx = train_test_split(train_eval_idx,
                                               test_size=0.25,
                                               stratify=label[train_eval_idx],
                                               random_state=seed)
        return train_idx, eval_idx, test_idx



