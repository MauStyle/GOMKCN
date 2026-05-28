import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_linear_assignment import batch_linear_assignment


class KC_layer(nn.Module):
    def __init__(self, num=8, size=5, dim=64, step=3, device='cpu'):
        super(KC_layer, self).__init__()
        self.num = num
        self.size = size
        self.step = step
        self.device = device
        self.A_f = Parameter(torch.FloatTensor(num, (size*(size-1))//2))
        self.F_f = Parameter(torch.FloatTensor(num, size, dim))
        self.A_f.data.uniform_(-1, 1)
        self.F_f.data.uniform_(-1, 1)

    def get_filter(self):
        A_f = torch.zeros(self.num, self.size, self.size).to(self.device)
        idx = torch.triu_indices(self.size, self.size, offset=1)
        A_f[:, idx[0], idx[1]] = A_f[:, idx[1], idx[0]] = F.sigmoid(self.A_f)
        F_f = self.F_f
        return A_f, F_f

    def forward(self, A_g, F_g, idxs):
        A_f, F_f = self.get_filter()
        F_g = torch.cat([F_g, torch.zeros(1, F_g.shape[1]).to(self.device)])[idxs]
        sim = torch.zeros((self.step, F_g.shape[0], F_f.shape[0], F_g.shape[1], F_f.shape[1]), dtype=torch.float).to(self.device)
        for hop in range(self.step):
            if hop != 0:
                F_g, F_f = A_g @ F_g, A_f @ F_f
            XY = torch.einsum("ace,bde->abcd", (F_g, F_f))
            X_2 = (F_g ** 2).sum(dim=-1)[:, None, :, None]
            Y_2 = (F_f ** 2).sum(dim=-1)[None, :, None, :]
            sim[hop] = 2 * XY / (X_2 + Y_2 + 1e-8) + 1
        sim = torch.sum(sim, dim=0)
        zero_positions = (idxs == A_g.shape[0])[:, None, :, None]
        sim = torch.where(zero_positions, torch.tensor(data=0.0, device=sim.device), sim)
        assignment = batch_linear_assignment(cost=-sim.reshape(F_g.shape[0] * F_f.shape[0], F_g.shape[1], F_f.shape[1]))
        assignment = assignment.reshape(F_g.shape[0], F_f.shape[0], F_f.shape[1])
        out = sim.gather(dim=3, index=assignment.unsqueeze(3)).squeeze(3).sum(2)
        return out

class GOMKCN(nn.Module):
    def __init__(self, d_in=None, d_out=None, size=8, dim=(64, 7), step=3, norm=True, actv=True, pool='mean', hidden=16, device='cpu'):
        super(GOMKCN, self).__init__()
        self.pool = pool
        self.device = device
        self.fc_in = nn.Sequential(nn.Linear(in_features=d_in, out_features=dim[0]),
                                    nn.BatchNorm1d(dim[0]),
                                    nn.Sigmoid())
        self.convs, self.norms, self.actvs, self.fc_out = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.fc_out.append(nn.Sequential(nn.Linear(in_features=dim[0], out_features=d_out)))
        for i in range(1, len(dim)):
            self.convs.append(KC_layer(num=dim[i], size=size, dim=dim[i-1], step=step, device=device))
            self.norms.append(nn.BatchNorm1d(dim[i]) if norm else nn.Identity())
            self.actvs.append(nn.ReLU() if actv else nn.Identity())
            self.fc_out.append(nn.Sequential(nn.Linear(in_features=dim[i], out_features=hidden),
                                             nn.ReLU(),
                                             nn.Linear(in_features=hidden, out_features=d_out)))

    def forward(self, A_g, F_g, idxs, indicator):
        F_g = self.fc_in(F_g)
        reps = [F_g]
        for conv, norm, actv in zip(self.convs, self.norms, self.actvs):
            F_g = actv(norm(conv(A_g, F_g, idxs)))
            reps.append(F_g)
        scores = 0
        for i, rep in enumerate(reps):
            if self.pool == 'add':
                rep = global_add_pool(rep, indicator)
            elif self.pool == 'mean':
                rep = global_mean_pool(rep, indicator)
            elif self.pool == 'max':
                rep = global_max_pool(rep, indicator)
            else:
                return
            scores += self.fc_out[i](rep)

        return scores