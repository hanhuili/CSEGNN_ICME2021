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

