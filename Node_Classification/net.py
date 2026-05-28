import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_linear_assignment import batch_linear_assignment


class KC_layer(nn.Module):
    def __init__(self, num=8, size=5, d_in=64, d_out=0, step=3, norm=True, actv='None', device='cpu'):
        super(KC_layer, self).__init__()
        self.num = num
        self.size = size
        self.step = step
        self.device = device
        self.A_f = Parameter(torch.FloatTensor(num, (size*(size-1))//2))
        self.F_f = Parameter(torch.FloatTensor(num, size, d_in if d_out == 0 else d_out))
        self.fc_in = nn.Sequential(
            nn.Identity() if d_out == 0 else nn.Linear(d_in, d_out),
            nn.BatchNorm1d(d_in if d_out == 0 else d_out) if norm else nn.Identity(),
            nn.Sigmoid() if actv == 'Sigmoid' else (nn.ReLU() if actv == 'ReLU' else nn.Identity())
        )
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
        F_g = self.fc_in(F_g)
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
        assignment = batch_linear_assignment(cost=-sim[:,:,1:,1:].reshape(F_g.shape[0] * F_f.shape[0], F_g.shape[1]-1, F_f.shape[1]-1))
        assignment = assignment.reshape(F_g.shape[0], F_f.shape[0], F_f.shape[1]-1)
        out = sim[:,:,0,0] + sim[:,:,1:,1:].gather(dim=3, index=assignment.unsqueeze(3)).squeeze(3).sum(2)
        return out


class GOMKCN(nn.Module):
    def __init__(self, dim=None, size=8, num=((64, 7),), step=3, norm=True, actv='None', device='cpu'):
        super(GOMKCN, self).__init__()
        self.device = device
        num = ((0, dim),) + num
        self.ker_convs = nn.ModuleList()
        for i in range(1, len(num)):
            self.ker_convs.append(
                KC_layer(d_in=num[i-1][1], d_out=num[i][0],  num=num[i][1], size=size,
                         step=step, norm=norm, actv=actv, device=device))

    def forward(self, A_g, F_g, idxs):
        for ker_conv in self.ker_convs:
            F_g = ker_conv(A_g, F_g, idxs)

        return F_g