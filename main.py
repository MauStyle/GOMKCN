import ast
import os
import sys
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
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('-seed', type=int, default=0, help='seed')
    parser.add_argument('-epochs', type=int, default=200, help='epochs')
    parser.add_argument('-batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('-dataset', default='PROTEINS_full', help='dataset folder name')
    parser.add_argument('-fold', type=int, default=4, help='split fold')
    parser.add_argument('-k_hop', type=int, default=2, help='neighbor hop')
    parser.add_argument('-s_subgraph', type=int, default=4, help='size of s_subgraph')
    parser.add_argument('-n_filter', type=ast.literal_eval, default=[[64, 7]], help='number of filter')
    parser.add_argument('-k_step', type=int, default=1, help='neighbor step')
    parser.add_argument('-tao', type=float, default=0.05, help='neighbor step')
    parser.add_argument('-norm', type=bool, default=True, help='norm')
    parser.add_argument('-relu', type=bool, default=True, help='relu')

    parser.add_argument('-pool', type=str, default='mean', help='pool of graph')
    return parser.parse_args()

def get_batchs(adjs, features, indexs, labels, idx, batch_size, device):
    adjs_batchs, features_batchs, labels_batchs, indexs_batchs, indicators_batchs, pos = [], [], [], [], [], 0
    while pos < len(idx):
        idx_batch = idx[pos : pos+batch_size]
        pos += len(idx_batch)
        adjs_batch, features_batch, labels_batch, indexs_batch, indicators_batch, total = [], [], [], [], [], 0
        for i, j in enumerate(idx_batch):
            adjs_batch.append(adjs[j])
            features_batch.append(features[j])
            labels_batch.append(labels[j])
            indexs_batch.append(torch.where(indexs[j]!=-1, indexs[j]+total, indexs[j]))
            indicators_batch.append(torch.full(size=(adjs[j].shape[0],), fill_value=i, device=device))
            total += adjs[j].shape[0]
        adjs_batchs.append(torch.vstack(adjs_batch))
        features_batchs.append(torch.vstack(features_batch))
        labels_batchs.append(torch.hstack(labels_batch))
        indexs_batchs.append(torch.vstack(indexs_batch))
        indicators_batchs.append(torch.hstack(indicators_batch))

    dict = {"adjs": adjs_batchs,
            "features": features_batchs,
            "labels": labels_batchs,
            "indexs": indexs_batchs,
            "indicators": indicators_batchs}
    return dict


def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    data_dict = {"DD": [89, 2], "NCI1": [37, 2], "PROTEINS_full": [32, 2], "ENZYMES": [21, 6], "MUTAG": [7, 2],
                 "IMDB-BINARY": [5, 2], "IMDB-MULTI": [5, 3], "REDDIT-BINARY": [5, 2], "COLLAB": [5, 3]}
    dims = data_dict.get(args.dataset)
    if dims is None: sys.exit()
    else: d_input, d_output = dims[0], dims[1]

    time_1 = time.time()

    dataset = LoadData(root='./dataset', dataset=args.dataset, k_hop=args.k_hop, s_subgraph=args.s_subgraph, device=device)
    adjs, features, indexs, labels = dataset.load_dataset()
    idx_train, idx_eval, idx_test = dataset.graph_splits(fold=args.fold)
    dict_train = get_batchs(adjs, features, indexs, labels, idx_train, args.batch_size, device)
    dict_eval = get_batchs(adjs, features, indexs, labels, idx_eval, args.batch_size, device)
    dict_test = get_batchs(adjs, features, indexs, labels, idx_test, args.batch_size, device)

    time_2 = time.time()

    model = GOMKCN(d_input=d_input, d_output=d_output, s_subgraph=args.s_subgraph, n_filter=args.n_filter,
                   k_step=args.k_step, tao=args.tao, norm=args.norm,
                   relu=args.relu, pool=args.pool, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    best_eval_acc, best_epoch, target_test_acc = 0.0, 0, 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_acc = AverageMeter(), AverageMeter()
        zipped_train = zip(dict_train.get('adjs'),
                           dict_train.get('features'),
                           dict_train.get('labels'),
                           dict_train.get('indexs'),
                           dict_train.get('indicators'))

        for adjs_batch, features_batch, labels_batch, indexs_batch, indicators_batch in zipped_train:
            # 训练
            optimizer.zero_grad()
            output = model(adjs_batch, features_batch, indexs_batch, indicators_batch)
            loss = F.cross_entropy(output, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), len(labels_batch))
            train_acc.update(accuracy(output, labels_batch), len(labels_batch))

        # scheduler.step()

        model.eval()
        eval_acc = AverageMeter()
        zipped_eval = zip(dict_eval.get('adjs'),
                          dict_eval.get('features'),
                          dict_eval.get('labels'),
                          dict_eval.get('indexs'),
                          dict_eval.get('indicators'))
        for adjs_batch, features_batch, labels_batch, indexs_batch, indicators_batch in zipped_eval:
            output = model(adjs_batch, features_batch, indexs_batch, indicators_batch)
            eval_acc.update(accuracy(output, labels_batch), len(labels_batch))

        model.eval()
        test_acc = AverageMeter()
        zipped_test = zip(dict_test.get('adjs'),
                          dict_test.get('features'),
                          dict_test.get('labels'),
                          dict_test.get('indexs'),
                          dict_test.get('indicators'))
        for adjs_batch, features_batch, labels_batch, indexs_batch, indicators_batch in zipped_test:
            output = model(adjs_batch, features_batch, indexs_batch, indicators_batch)
            test_acc.update(accuracy(output, labels_batch), len(labels_batch))

        with open(os.path.join(save_folder, 'accuracy.txt'), 'a') as f:
            f.write(f"fold:{args.fold:d} "
                    f"epoch:{epoch:03d} "
                    f"loss:{train_loss.avg:.6f} "
                    f"train_acc:{train_acc.avg:.6f} "
                    f"eval_acc:{eval_acc.avg:.6f} "
                    f"test_acc:{test_acc.avg:.6f}\n")
        print("fold:" + '%d ' % (args.fold) +
              "epoch:" + '%03d ' % (epoch) +
              "loss=" + "{:.6f} ".format(train_loss.avg) +
              "train_acc=" + "{:.6f} ".format(train_acc.avg) +
              "eval_acc=" + "{:.6f} ".format(eval_acc.avg) +
              "test_acc=" + "{:.6f} ".format(test_acc.avg))
        if eval_acc.avg > best_eval_acc:
            best_eval_acc, best_epoch, target_test_acc = eval_acc.avg, epoch, test_acc.avg
            torch.save({'state_dict': model.state_dict()},
                       os.path.join(save_folder, 'model.tar'))

    time_3 = time.time()

    with open(os.path.join(save_folder, 'selected_model_accuracy.txt'), 'a') as f:
        f.write(f"epoch:{best_epoch:03d} test_acc:{target_test_acc:.6f} data_time:{time_2-time_1:.6f}  model_time:{time_3-time_2:.6f}\n")

    os.rename(src=os.path.join(save_folder, 'selected_model_accuracy.txt'),
              dst=os.path.join(save_folder, f'{target_test_acc:.6f}.txt'))

    print("best_epoch={:03d} ".format(best_epoch) + "accuracy={:.6f} ".format(target_test_acc))
    print("done!")


if __name__ == "__main__":
    main()
