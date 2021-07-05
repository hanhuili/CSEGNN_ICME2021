import torch
import os
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from utils.sketch_util import SketchUtil
import random


class QuickDraw414k(torch.utils.data.Dataset):
    def __init__(self, opts, split="train"):
        super(QuickDraw414k, self).__init__()
        self.is_train = split == "train"
        self.max_stroke_num = opts["max_stroke_num"]
        self.sketches = np.load(os.path.join(opts["data_dir"], "quickdraw", "qd_{}.npy".format(split)), allow_pickle=True)[0]
        self.paint_size = (opts["paint_size"], opts["paint_size"])
        self.digraph = opts["directed_graph"]
        self.split = split
        self.img_files = []
        self.labels = []
        self.translation = int(0.01 * opts["paint_size"])
        self.data_dir = os.path.join(opts["data_dir"], "quickdraw", "coordinate_files")
        split_file = os.path.join(self.data_dir, "tiny_{}_set.txt".format(split))
        with open(split_file, "r") as f:
            for line in f:
                img_file, label = line.split()
                self.img_files.append(img_file)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def get_edge_idx(self, strokes):
        srcs = np.where(strokes[:, 2] == 1)[0]
        srcs = srcs[srcs < self.max_stroke_num - 1]     # so that dsts[-1] = max_stroke_num - 1
        dsts = srcs + 1
        if self.digraph:
            edge_idxs = np.concatenate((np.expand_dims(np.concatenate((srcs, dsts[::-1]), axis=0), axis=0), np.expand_dims(np.concatenate((dsts, srcs[::-1]), axis=0), axis=0)), axis=0)
        else:
            edge_idxs = np.concatenate((np.expand_dims(srcs, axis=0), np.expand_dims(dsts, axis=0)), axis=0)

        return edge_idxs

    def __getitem__(self, item):

        strokes = self.sketches[item]
        pad_strokes = np.zeros((self.max_stroke_num, 4))
        pad_strokes[:strokes.shape[0], :] = strokes
        pad_strokes[pad_strokes == -1] = 0
        strokes = pad_strokes

        edge_index = self.get_edge_idx(strokes)
        ratio = 1.0
        if ratio < 1:
            rnum = int(ratio * edge_index.shape[1])
            if rnum < 1:
                rnum = 1
            edge_index = edge_index[:,np.random.choice(edge_index.shape[1], rnum, replace=False)]

        edge_index = torch.tensor(edge_index, dtype=torch.long)

        stop_idx = np.where(np.sum(strokes, axis=1) == 0)[0]
        if stop_idx.size == 0:
            stop_idx = self.max_stroke_num
        else:
            stop_idx = stop_idx[0]

        if self.is_train and stop_idx > 0:
            strokes[: stop_idx, :2] += np.random.randint(-self.translation, self.translation, size=(stop_idx, 2))

        label = self.labels[item]

        if self.is_train:
            if random.random() >= 0.5:
                strokes[:stop_idx, 0] = self.paint_size[1] - strokes[:stop_idx, 0]

        strokes = strokes.astype("float32")
        strokes = strokes[:, : 3]
        temporal = np.zeros((self.max_stroke_num, 1), dtype=np.float32)
        if stop_idx > 0:
            temporal[: stop_idx, 0] = np.arange(1, stop_idx + 1, dtype=np.float32) / stop_idx
        strokes = np.concatenate((strokes, temporal), axis=1)
        strokes = torch.tensor(strokes[: stop_idx, :])
        strokes[:, :2] /= self.paint_size[0]
        strokes[strokes < 0] = 0
        return Data(x=strokes, edge_index=edge_index, y=label)


class QuickDraw3M(torch.utils.data.Dataset):
    def __init__(self, opts, is_train, strokes, labels, split_len):
        self.is_train = is_train
        self.root_dir = "../dataset/Sketch/quickdraw_full"
        self.split_len = split_len
        self.data = None
        self.paint_size = (opts["paint_size"], opts["paint_size"])
        self.max_stroke_num = opts["max_stroke_num"]
        self.translation = int(0.01 * opts["paint_size"])
        self.digraph = opts["directed_graph"]
        self.strokes = strokes
        self.labels = labels
        self.current_split = 0

    def __len__(self):
        return self.labels.shape[0]

    def get_edge_idx(self, strokes):
        srcs = np.where(strokes[:, 2] == 1)[0]
        srcs = srcs[srcs < self.max_stroke_num - 1]     # so that dsts[-1] = max_stroke_num - 1
        dsts = srcs + 1
        if self.digraph:
            edge_idxs = np.concatenate((np.expand_dims(np.concatenate((srcs, dsts[::-1]), axis=0), axis=0), np.expand_dims(np.concatenate((dsts, srcs[::-1]), axis=0), axis=0)), axis=0)
        else:
            edge_idxs = np.concatenate((np.expand_dims(srcs, axis=0), np.expand_dims(dsts, axis=0)), axis=0)
        return edge_idxs

    def set_current_split(self, split):
        self.current_split = split

    def shuffle_data(self):
        for ci, strokes in enumerate(self.strokes):
            indexs = np.random.permutation(len(strokes))
            self.strokes[ci] = strokes[indexs]

    def __getitem__(self, idx):

        index_tuple = self.labels[idx]
        label = int(index_tuple[0])
        sid = index_tuple[1]

        ptx = self.strokes[label][self.current_split * self.split_len + sid]
        ptx = np.array(ptx, dtype=np.float32)
        ptx[:, 0:2] = np.cumsum(ptx[:, 0:2], axis=0)
        pts3_norm = SketchUtil.normalization(ptx[:, 0:2])
        if pts3_norm is not None:
            ptx[:, 0:2] = pts3_norm

        strokes = np.zeros((self.max_stroke_num, 4))
        strokes[: ptx.shape[0], 0] = (self.paint_size[0]) * (ptx[:, 0] + 1) / 2
        strokes[: ptx.shape[0], 1] = (self.paint_size[1]) * (ptx[:, 1] + 1) / 2
        strokes[: ptx.shape[0], 2] = 1 - ptx[:, 2]
        strokes[: ptx.shape[0], 3] = ptx[:, 2]

        edge_index = self.get_edge_idx(strokes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        stop_idx = np.where(np.sum(strokes, axis=1) == 0)[0]
        if stop_idx.size == 0:
            stop_idx = self.max_stroke_num
        else:
            stop_idx = stop_idx[0]

        if self.is_train and stop_idx > 0:
            strokes[: stop_idx, :2] += np.random.randint(-self.translation, self.translation, size=(stop_idx, 2))

        if self.is_train:
            if random.random() >= 0.5:
                strokes[:stop_idx, 0] = self.paint_size[1] - strokes[:stop_idx, 0]

        strokes = strokes.astype("float32")
        strokes = strokes[:, : 3]
        temporal = np.zeros((self.max_stroke_num, 1), dtype=np.float32)
        if stop_idx > 0:
            temporal[: stop_idx, 0] = np.arange(1, stop_idx + 1, dtype=np.float32) / stop_idx
        strokes = np.concatenate((strokes, temporal), axis=1)
        strokes = torch.tensor(strokes[: stop_idx, :])
        strokes[:, :2] /= self.paint_size[0]
        strokes[strokes < 0] = 0
        return Data(x=strokes, edge_index=edge_index, y=label)


def build_QuickDraw3M(opts, get_valid=False):
    data_dir = "../dataset/Sketch/quickdraw_split"
    categories = []
    with open("categories", "r") as f:
        for line in f:
            categories.append(line.split("\n")[0])
    train_split_num = opts["train_split_num"]
    train_num = 70000
    train_sketch_per_split = train_num // train_split_num
    train_strokes = []
    train_labels = []

    test_num = 2500
    test_strokes = []
    test_labels = []
    if get_valid:
        valid_num = 2500
        valid_strokes = []
        valid_labels = []

    for ci, category in enumerate(categories):
        print(category)
        # if ci > 5:
        #     break
        np_file = os.path.join(data_dir, "{}.npz".format(category))
        nps = np.load(np_file, allow_pickle=True, encoding="latin1")

        "training"
        t_strokes = nps["train"]
        assert len(t_strokes) == train_num
        train_strokes.append(t_strokes)

        tlabels = np.zeros((train_sketch_per_split, 2), dtype=np.int)
        tlabels[:, 0] = ci
        tlabels[:, 1] = np.arange(0, train_sketch_per_split)
        train_labels.append(tlabels)

        "test"
        t_strokes = nps["test"]
        assert len(t_strokes) == test_num
        test_strokes.append(t_strokes)

        tlabels = np.zeros((test_num, 2), dtype=np.int)
        tlabels[:, 0] = ci
        tlabels[:, 1] = np.arange(0, test_num)
        test_labels.append(tlabels)

        "valid"
        if get_valid:
            t_strokes = nps["valid"]
            assert len(t_strokes) == valid_num
            valid_strokes.append(t_strokes)

            tlabels = np.zeros((valid_num, 2), dtype=np.int)
            tlabels[:, 0] = ci
            tlabels[:, 1] = np.arange(0, valid_num)
            valid_labels.append(tlabels)

        # if ci > 1:
        #     break
    print("end of loading data")
    num_workers = 16
    train_labels = np.concatenate(train_labels, axis=0)
    train_dataset = QuickDraw3M(opts, is_train=True, strokes=train_strokes, labels=train_labels, split_len=train_sketch_per_split)
    train_dataloader = DataLoader(train_dataset, batch_size=opts["batch_size"], shuffle=True, num_workers=num_workers)

    test_labels = np.concatenate(test_labels, axis=0)
    test_dataset = QuickDraw3M(opts, is_train=False, strokes=test_strokes, labels=test_labels, split_len=test_num)
    test_dataloader = DataLoader(test_dataset, batch_size=opts["batch_size"], shuffle=False, num_workers=num_workers)
    if not get_valid:
        return train_dataloader, test_dataloader

    else:
        valid_labels = np.concatenate(valid_labels, axis=0)
        valid_dataset = QuickDraw3M(opts, is_train=False, strokes=valid_strokes, labels=valid_labels, plit_len=valid_num)
        valid_dataloader = DataLoader(valid_dataset, batch_size=opts["batch_size"], shuffle=False, num_workers=num_workers)
        return train_dataloader, test_dataloader, valid_dataloader


