import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class KC_layer(nn.Module):
    def __init__(self, n_filter=8, s_subgraph=5, d_input=64, d_output=0, k_step=3, tao=0.05, device='cpu'):
        super(KC_layer, self).__init__()

        self.n_filter = n_filter
        self.s_subgraph = s_subgraph
        self.d_input = d_input
        self.d_output = d_output
        self.k_step = k_step
        self.tao = tao
        self.device = device

        self.adjs_hidden = Parameter(torch.FloatTensor(n_filter, (s_subgraph*(s_subgraph-1))//2))
        if d_output == 0:
            self.features_hidden = Parameter(torch.FloatTensor(n_filter, s_subgraph, d_input))
        else:
            self.features_hidden = Parameter(torch.FloatTensor(n_filter, s_subgraph, d_output))
            self.fc_in = nn.Sequential(nn.Linear(d_input, d_output), nn.BatchNorm1d(d_output), nn.Sigmoid())
        self.adjs_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(-1, 1)

    def forward(self, adjs, feature, idxs):
        adjs_hidden = torch.zeros(self.n_filter, self.s_subgraph, self.s_subgraph).to(self.device)
        idx = torch.triu_indices(self.s_subgraph, self.s_subgraph, offset=1)
        adjs_hidden[:, idx[0], idx[1]] = adjs_hidden[:, idx[1], idx[0]] = F.sigmoid(self.adjs_hidden)
        features_hidden = self.features_hidden
        if self.d_output != 0:
            feature = self.fc_in(feature)
        feature = torch.cat([feature, torch.zeros(1, feature.shape[1]).to(self.device)])
        features = feature[idxs]
        total = torch.zeros((self.k_step, features.shape[0], features_hidden.shape[0], features.shape[1], features_hidden.shape[1]), dtype=torch.float).to(self.device)
        for hop in range(self.k_step):
            if hop != 0:
                features, features_hidden = adjs @ features, adjs_hidden @ features_hidden
            features_square = torch.einsum("ace,bde->abcd", (features * features, torch.ones_like(features_hidden)))
            features_hidden_square = torch.einsum("ace,bde->abcd", (torch.ones_like(features), features_hidden * features_hidden))
            features_features_hidden = torch.einsum("ace,bde->abcd", (features, features_hidden))
            total[hop] = torch.exp(-(features_square + features_hidden_square - 2 * features_features_hidden) / self.features_hidden.shape[2] / self.tao)
        total = total.sum(dim=0)
        similarity = total.clone()
        mask = torch.zeros((features.shape[0], features_hidden.shape[0], features.shape[1], features_hidden.shape[1]), dtype=torch.float).to(self.device)
        mask[:,:,0,0] = 1.0
        similarity[:,:,:,0] = -1.0
        for i in range(1, similarity.shape[2]):
            idx = torch.argmax(input=similarity[:, :, i, :], dim=2, keepdim=True)
            mask[:, :, i, :].scatter_(dim=2, index=idx, src=torch.ones_like(idx, dtype=torch.float))
            idx_fix = idx[:, :, None, :].repeat(1, 1, similarity.shape[2], 1)
            similarity.scatter_(dim=3, index=idx_fix, src=-torch.ones_like(idx_fix, dtype=torch.float))
        out = torch.sum(total * mask, dim=(2, 3))

        return out

class GOMKCN(nn.Module):
    def __init__(self, d_input=None, d_output=None, s_subgraph=8, n_filter=[[64, 7]], k_step=3, tao=0.05, norm=True, relu=True, pool='mean', device='cpu'):
        super(GOMKCN, self).__init__()
        self.device = device
        self.norm = norm
        self.relu = relu
        self.pool = pool

        self.ker_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.leaky_relus = nn.ModuleList()
        self.fc_out = torch.nn.ModuleList()

        self.fc_out.append(nn.Sequential(nn.Linear(d_input, d_output)))
        for i in range(len(n_filter)):
            if i == 0:
                self.ker_convs.append(KC_layer(d_input=d_input, d_output=n_filter[i][0], n_filter=n_filter[i][1], s_subgraph=s_subgraph, k_step=k_step, tao=tao, device=device))
            else:
                self.ker_convs.append(KC_layer(d_input=n_filter[i - 1][1], d_output=n_filter[i][0], n_filter=n_filter[i][1], s_subgraph=s_subgraph, k_step=k_step, tao=tao, device=device))
            if self.norm:
                self.batch_norms.append(nn.BatchNorm1d(n_filter[i][1]))
            else:
                self.batch_norms.append(nn.Identity())
            if self.relu:
                # self.leaky_relus.append(nn.LeakyReLU())
                self.leaky_relus.append(nn.Sigmoid())
            else:
                self.leaky_relus.append(nn.Identity())
            self.fc_out.append(nn.Sequential(nn.Linear(n_filter[i][1], 16), nn.ReLU(), nn.Linear(16, d_output)))

    def forward(self, adjs, feature, idxs, indicator):
        reps = [feature]

        for ker_conv, batch_norm, leaky_relu in zip(self.ker_convs, self.batch_norms, self.leaky_relus):
            feature = ker_conv(adjs, feature, idxs)
            feature = batch_norm(feature)
            feature = leaky_relu(feature)
            reps.append(feature)

        score_over_layer = 0
        for i, rep in enumerate(reps):
            if self.pool == 'add':
                rep = global_add_pool(rep, indicator)
            elif self.pool == 'mean':
                rep = global_mean_pool(rep, indicator)
            elif self.pool == 'max':
                rep = global_max_pool(rep, indicator)
            else:
                return
            score_over_layer += self.fc_out[i](rep)

        return score_over_layer