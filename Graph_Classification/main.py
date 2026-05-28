import ast
import os
import time
import torch
import random
import shutil
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from data import LoadData
from net import GOMKCN
from utils import accuracy, AverageMeter


def get_args():
    parser = argparse.ArgumentParser(description='Args for Graph Predition')
    parser.add_argument('-seed', type=int, default=0, help='seed')
    parser.add_argument('-epoch', type=int, default=200, help='epoch')
    parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('-wd', type=float, default=0, help='weight decay')
    parser.add_argument('-batch', type=int, default=128, help='batch')
    parser.add_argument('-result', type=str, default='result', help='result folder')
    parser.add_argument('-root', type=str, default='dataset', help='dataset folder')
    parser.add_argument('-dataset', default='REDDIT-BINARY', help='dataset folder')
    parser.add_argument('-fold', type=int, choices=[0,1,2,3,4,5,6,7,8,9], default=0, help='split fold')
    parser.add_argument('-hop', type=int, default=1, help='neighbor nodes')
    parser.add_argument('-size', type=int, default=4, help='size of subgraph')
    parser.add_argument('-dim', type=ast.literal_eval, default=(64,8), help='number of filter')
    parser.add_argument('-step', type=int, choices=[1,2,3,4,5], default=2, help='neighbor step')
    parser.add_argument('-hidden', type=int, default=16, help='hidden in MPL')
    parser.add_argument('-norm', action='store_true', help='norm')
    parser.add_argument('-actv', action='store_true', help='actv')
    parser.add_argument('-pool', type=str, choices=['add','mean','max'], default='mean', help='pooling method')
    parser.add_argument('-device', type=str, choices=['cpu','cuda:0','cuda:1'], default='cuda:0', help='device')
    return parser.parse_args()

def run_model(model, optimizer, G_dict, mode='train'):
    loss_Meter, acc_Meter = AverageMeter(), AverageMeter()
    G_split = G_dict[mode]
    model.train() if mode == 'train' else model.eval()
    zipped_data = zip(G_split.get('adj'),
                      G_split.get('feature'),
                      G_split.get('label'),
                      G_split.get('index'),
                      G_split.get('indicator'))
    for A_batch, F_batch, Y_batch, index_batch, indicator_batch in zipped_data:
        if mode == 'train':
            optimizer.zero_grad()
        output = model(A_batch, F_batch, index_batch, indicator_batch)
        loss = F.cross_entropy(output, Y_batch)
        if mode == 'train':
            loss.backward()
            optimizer.step()
        loss_Meter.update(loss.item(), len(Y_batch))
        acc_Meter.update(accuracy(output, Y_batch), len(Y_batch))
    return loss_Meter, acc_Meter

def main():
    args = get_args()
    device = torch.device(args.device)
    print('device: %s' % device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_folder = os.path.join("result", args.dataset, str(args.fold) + '-' + time.strftime("%Y%m%d%H%M%S"))
    if os.path.exists(save_folder) : shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'selected_model_accuracy.txt'), 'a') as f:
        for k, v in vars(args).items():
            f.write(f"{k}:{v}\n")
    time_1 = time.time()
    dataset = LoadData(root=args.root, dataset=args.dataset, hop=args.hop, size=args.size, device=device)
    dim, classes, A_all, F_all, Y_all, index_all = dataset.load_dataset()
    idx_dict = dataset.graph_split(fold=args.fold)
    G_dict = dataset.get_batchs(A_all, F_all, Y_all, index_all, idx_dict, args.batch)
    time_2 = time.time()
    model = GOMKCN(d_in=dim, d_out=classes, size=args.size, dim=args.dim, step=args.step, norm=args.norm,
                   actv=args.actv, pool=args.pool, hidden=args.hidden, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_eval_acc, best_epoch, target_test_acc = 0.0, 0, 0.0
    for epoch in range(1, args.epoch+1):
        train_loss, train_acc = run_model(model, optimizer, G_dict, mode='train')
        eval_loss, eval_acc = run_model(model, optimizer, G_dict, mode='eval')
        test_loss, test_acc = run_model(model, optimizer, G_dict, mode='test')
        log_str = (f"Fold:{args.fold:d} | "
                   f"Epoch:{epoch:03d} | "
                   f"Train Loss:{train_loss.avg:.6f} "
                   f"Train Acc:{train_acc.avg:.6f} | "
                   f"Eval Loss:{eval_loss.avg:.6f} "
                   f"Eval Acc:{eval_acc.avg:.6f} | "
                   f"Test Acc:{test_acc.avg:.6f} "
                   f"Target Acc:{target_test_acc:.6f}\n")
        with open(os.path.join(save_folder, 'accuracy.txt'), 'a') as f:
            f.write(log_str)
        print(log_str, end='')
        if eval_acc.avg > best_eval_acc:
            best_eval_acc, best_epoch, target_test_acc = eval_acc.avg, epoch, test_acc.avg
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(save_folder, 'model.tar'))
    time_3 = time.time()
    with open(os.path.join(save_folder, 'selected_model_accuracy.txt'), 'a') as f:
        f.write(f"Epoch:{best_epoch:03d} Test Acc:{target_test_acc:.6f} Data Time:{time_2-time_1:.6f}  Model Time:{time_3-time_2:.6f}\n")
    os.rename(src=os.path.join(save_folder, 'selected_model_accuracy.txt'),
              dst=os.path.join(save_folder, f'{target_test_acc:.6f}.txt'))
    print("Best Epoch={:03d} ".format(best_epoch) + "Accuracy={:.6f} ".format(target_test_acc))
    print("done!")


if __name__ == "__main__":
    main()
