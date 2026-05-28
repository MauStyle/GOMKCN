import ast
import os
import time
import uuid
import torch
import random
import shutil
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from net import GOMKCN
from data import LoadData, accuracy


def get_args():
    parser = argparse.ArgumentParser(description='Args for Node Classification')
    parser.add_argument('-seed', type=int, default=0, help='seed')
    parser.add_argument('-epochs', type=int, default=200, help='epochs')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-wd', type=float, default=1e-3, help='weight decay')
    parser.add_argument('-result', type=str, default='result', help='result folder')
    parser.add_argument('-root', type=str, default='dataset', help='dataset folder')
    parser.add_argument('-dataset', default='Cora', choices=['Cora', 'CiteSeer', 'PubMed', 'chameleon', 'squirrel', 'actor'], help='dataset')
    parser.add_argument('-hop', type=int, default=1, help='neighbor hop')
    parser.add_argument('-size', type=int, default=6, help='rate of subgraph')
    parser.add_argument('-num', type=ast.literal_eval, default=((16,7),), help='number of filter')
    parser.add_argument('-step', type=int, default=1, help='neighbor step')
    parser.add_argument('-norm', action='store_true', help='norm in MLP')
    parser.add_argument('-actv', type=str, default='None', choices=['Sigmoid','ReLU', 'None'], help='actv')
    parser.add_argument('-device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1'], help='device')
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    print('device: %s' % device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    save_folder = os.path.join(args.result, args.dataset, str(args.hop) + '-' + str(args.size) + '-' + str(args.num) + '-' + str(args.step) + '-' + time.strftime("%Y%m%d%H%M%S") + '-' + str(uuid.uuid4())[:8])
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    with open(os.path.join(str(save_folder), 'selected_model_accuracy.txt'), 'a') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}:{v}\n")
    dataset = LoadData(root=args.root, dataset=args.dataset, hop=args.hop, size=args.size)
    dim, idxs, A_g, F_g, Y_g = dataset.get_data(dataset.graph)
    train_idx, eval_idx, test_idx = dataset.get_split(label=Y_g, seed=args.seed)
    idxs, A_g, F_g, Y_g = idxs.to(device), A_g.to(device), F_g.to(device), Y_g.to(device)
    time_start = time.time()
    epoch_iters, accuracy_iters = np.zeros(shape=10, dtype=np.int32), np.zeros(shape=10, dtype=np.float32)
    for iter in range(10):
        model = GOMKCN(dim=dim, size=args.size, num=args.num, step=args.step,
                       norm=args.norm, actv=args.actv, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best_eval_acc = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            output = model(A_g, F_g, idxs)
            train_loss = F.cross_entropy(output[train_idx], Y_g[train_idx])
            train_loss.backward()
            optimizer.step()
            model.eval()
            output = model(A_g, F_g, idxs)
            train_acc = accuracy(output[train_idx], Y_g[train_idx])
            eval_acc = accuracy(output[eval_idx], Y_g[eval_idx])
            test_acc = accuracy(output[test_idx], Y_g[test_idx])
            if eval_acc.item() > best_eval_acc:
                best_eval_acc = eval_acc.item()
                epoch_iters[iter], accuracy_iters[iter] = epoch, test_acc.item()
                model_path = os.path.join(str(save_folder), 'model_iter_{}.pth'.format(iter))
                torch.save(model, model_path)
            log_str = (f"Iter:{iter:d} epoch:{epoch:03d} | "
                       f"Train_loss:{train_loss.item():.6f} | "
                       f"Train_acc:{train_acc.item():.6f} | "
                       f"Eval_acc:{eval_acc.item():.6f} | "
                       f"Test_acc:{test_acc.item():.6f} | "
                       f"Target_acc:{accuracy_iters[iter]:.6f}\n")
            with open(os.path.join(str(save_folder), 'accuracy.txt'), 'a') as f:
                f.write(log_str)
            print(log_str, end='')
        with open(os.path.join(str(save_folder), 'selected_model_accuracy.txt'), 'a') as f:
            f.write(f"iter:{iter:d} epoch:{epoch_iters[iter]:03d} test_acc:{accuracy_iters[iter]:.6f}\n")
    time_end = time.time()
    with open(os.path.join(str(save_folder), 'selected_model_accuracy.txt'), 'a') as f:
        f.write(f"acc_mean:{accuracy_iters.mean():.6f} "
                f"acc_std:{accuracy_iters.std():.6f} "
                f"run time:{time_end-time_start:.6f}\n")
    os.rename(src=os.path.join(str(save_folder), 'selected_model_accuracy.txt'),
              dst=os.path.join(str(save_folder), f'{accuracy_iters.mean():.6f}.txt') )
    print("done!")


if __name__ == "__main__":
    main()
