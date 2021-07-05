import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim import Adam, AdamW
import os
import logging
import time

from utils.dataloader import QuickDraw414k
from torch_geometric.data import DataLoader
from utils.config import opts
from utils.mGNN import mGNN

device = "cuda:1"


def collect_fn(batch, device):
    batch["x"] = batch["x"].to(device)
    batch["y"] = batch["y"].to(device)
    batch["edge_index"] = batch["edge_index"].to(device)
    return batch


def run_epoch(net, data_loader, optimizer=None, is_train=False):
    num_class = opts["num_class"]
    # acounts = np.zeros((num_class,))
    # counts = np.zeros((num_class,))
    acounts = 0
    counts = 0
    ttime = 0

    freq = int(0.2 * len(data_loader))
    if freq < 1:
        freq = 1
    if is_train:
        net.train()
        criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    else:
        net.eval()
    for i, batch in enumerate(data_loader):

        start_time = time.time()
        batch = collect_fn(batch, device)
        labels = batch["y"]

        if is_train:
            preds, _ = net(batch)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds, _ = net(batch)

        pred_labels = torch.argmax(preds, dim=1)
        corrects = (pred_labels == labels).cpu().numpy()
        acounts += np.sum(corrects)
        counts += len(corrects)
        # for ci in range(num_class):
        #     idx = np.where(labels.cpu().numpy() == ci)[0]
        #     if len(idx) > 0:
        #         counts[ci] += len(idx)
        #         acounts[ci] += np.sum(corrects[idx])

        torch.cuda.synchronize()
        ttime += (time.time() - start_time)



        if (i + 1) % freq == 0:
            tacc = acounts / counts
            # tacc[np.isnan(tacc)] = 0
            print("process {:.2f} data, acc {:.4f}, consume {:.1f} seconds".format(i / len(data_loader), np.mean(tacc), ttime))
    return acounts / counts, ttime


def main():
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    sub_result_dir = os.path.join(result_dir, "raw")
    if not os.path.exists(sub_result_dir):
        os.mkdir(sub_result_dir)

    num_workers = 16
    train_dataloader = DataLoader(QuickDraw414k(opts, "train"), batch_size=opts["batch_size"], shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(QuickDraw414k(opts, "test"), batch_size=opts["batch_size"], shuffle=False, num_workers=num_workers)

    net = mGNN(opts, 4)
    net_paramn = sum(p.numel() for p in net.parameters())
    print("param num: {}".format(net_paramn))

    net = net.to(device)
    optimizer = Adam(net.parameters(), lr=opts["lr"], weight_decay=opts["weight_decay"])

    start_epoch = 0
    epochs = opts["train_epochs"]
    for e in range(epochs):
        temp_path = os.path.join(sub_result_dir, 'model' + str(epochs - e) + '.pth')
        if os.path.exists(temp_path):
            start_epoch = epochs - e
            checkpoint = torch.load(temp_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_acc = checkpoint['train_acc']
            test_acc = checkpoint['test_acc']
            break

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if start_epoch == 0:
        train_acc = []
        test_acc = []
        params = ['{}: {} \n'.format(k, v) for k, v in opts.items()]
        logging.basicConfig(filename=os.path.join(sub_result_dir, 'log.txt'), filemode='w',
                            format='%(asctime)s - %(message)s', level=logging.INFO)
        logging.info(' '.join(params))
        logging.info("param num: {}".format(net_paramn))
    else:
        logging.basicConfig(filename=os.path.join(sub_result_dir, 'log.txt'), filemode='a',
                            format='%(asctime)s - %(message)s', level=logging.INFO)

    for e in range(start_epoch, epochs):
        lr = (0.1 ** (e // 10)) * opts["lr"]
        if lr < 1e-6:
            lr = 1e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        acc, ttime = run_epoch(net, train_dataloader, optimizer, True)
        train_acc.append(acc)
        info_msg = "train epoch {}, acc: {:.4f}, consumed {:.2f} seconds".format(
            e + 1, np.mean(acc), ttime)
        print(info_msg)
        logging.info(info_msg)

        if (e + 1) % opts["test_epoch"] == 0:
            acc, ttime = run_epoch(net, test_dataloader, None, False)
            test_acc.append(acc)
            info_msg = "test epoch {}, acc: {:.4f}, consumed {:.2f} seconds".format(
                e + 1, np.mean(acc), ttime)
            print(info_msg)
            logging.info(info_msg)

        if (e + 1) % opts["save_epoch"] == 0:
            temp_path = os.path.join(sub_result_dir, 'model' + str(e + 1) + '.pth')
            torch.save({
                'net': net.state_dict(),
                "optimizer": optimizer.state_dict(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, temp_path)


if __name__ == "__main__":
    cudnn.enable = True
    # cudnn.deterministic = True
    # cudnn.benchmark = True

    np.random.seed(0)
    torch.manual_seed(0)
    main()