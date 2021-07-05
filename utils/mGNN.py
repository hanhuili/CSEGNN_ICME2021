import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import voxel_grid, avg_pool_x, max_pool_x, global_mean_pool
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter

def make_divisible(channels, groups):
    if channels % groups != 0:
        channels = int(np.ceil(channels // groups)) * groups
    return channels

def shuffle(x, groups):
    sz = x.shape[:]
    if len(sz) == 3:
        x = x.view(sz[0], groups, sz[1] // groups, sz[2])
        x = x.transpose(1, 2).contiguous().view(sz)
    else:
        x = x.view(sz[0], groups, sz[1] // groups, sz[2], sz[3])
        x = x.transpose(1, 2).contiguous().view(sz)
    return x

def sum_pool_x(cluster, x, batch, size):
    batch_size = int(batch.max().item()) + 1
    return scatter(x, cluster, dim=0, dim_size=batch_size * size, reduce="sum"), None


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        return shuffle(x, self.groups)


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), h_swish())
        for i in range(1, len(channels))
    ])


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MLPG(nn.Module):
    def __init__(self, channels, groups, shuffle=True):
        super(MLPG, self).__init__()
        self.groups = groups
        self.shuffle = shuffle
        self.trans = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], 1, bias=False, groups=groups),
                          nn.BatchNorm1d(channels[i]), h_swish())
            for i in range(1, len(channels))])

    def forward(self, x):
        "x : N (num) x D (dim)"
        x = x.unsqueeze(-1)
        x = self.trans(x)
        if self.shuffle:
            x = shuffle(x, self.groups)
        x = x.squeeze(-1)
        return x


class MLPGX(MLPG):
    def __init__(self, channels, groups):
        super(MLPGX, self).__init__(channels, groups)
        self.gate = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], 1, groups=groups),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.trans(x)
        x = shuffle(x, self.groups)
        x = x * self.gate(x)
        x = x.squeeze(-1)
        return x


class GroupResConv(nn.Module):
    def __init__(self, channels, kernel_size, groups):
        super(GroupResConv, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            h_swish(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, groups=groups),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.trans(x)
        x = self.gate(x) * x
        return x


class GridConv(nn.Module):
    def __init__(self, channels, kernel_size, groups, grid_num, pool=False):
        super(GridConv, self).__init__()
        self.grid_num = grid_num
        self.trans = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            h_swish(),
            nn.Conv2d(channels, channels, 1, bias=False, groups=groups),
            nn.BatchNorm2d(channels),
            h_swish(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.pool = pool
        if pool:
            inner_channels = channels // 4
            pool_groups = np.min([groups, 64])
            inner_channels = make_divisible(inner_channels, pool_groups)
            self.spatial_trans = nn.Sequential(
                nn.Conv2d(channels, inner_channels, 1, groups=pool_groups, bias=False),
                nn.BatchNorm2d(inner_channels),
                h_swish(),
            )
            self.spatial_score = nn.Sequential(
                nn.Conv2d(inner_channels * 2, 1, 1),
                nn.Sigmoid(),
            )
            self.channel_score = nn.Sequential(
                nn.Conv2d(channels, channels, 1, groups=groups),
                nn.Sigmoid(),
            )

    def forward(self, x, pos, batch, paint_size):
        cluster = voxel_grid(pos, batch, float(np.ceil(paint_size / self.grid_num)), start=0, end=paint_size)
        x_grid, _ = max_pool_x(cluster, x.clone(), batch, self.grid_num * self.grid_num)
        x_grid = x_grid.view(-1, self.grid_num * self.grid_num, x_grid.shape[-1]).permute(0, 2, 1).contiguous()
        x_grid = x_grid.view(-1, x_grid.shape[1], self.grid_num, self.grid_num)
        x_grid = self.trans(x_grid)
        if self.pool:
            x_gridt = self.spatial_trans(x_grid)
            x_grid_score = self.spatial_score(torch.cat([x_gridt, self.avg(x_gridt).repeat(1, 1, x_gridt.shape[-2], x_gridt.shape[-1])], dim=1))
            x_avg = self.avg(x_grid * (x_grid_score))
            x_avg = self.channel_score(x_avg) * x_avg
        else:
            x_avg = self.avg(x_grid)
        return x_avg


class GroupEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, groups=1):
        super(GroupEdgeConv, self).__init__(aggr="add")
        if groups == 1:
            self.in_conv = MLP([in_channels, out_channels])
        else:
            self.in_conv = MLPG([in_channels, out_channels], groups)

        self.edge_block = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            h_swish(),
            nn.Linear(out_channels // 4, out_channels),
            h_swish(),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid(),
        )
        self.act = h_swish()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        x = self.in_conv(x)
        x = self.act(x + self.propagate(edge_index, x=x))
        return x

    def message(self, x_j, x_i):
        return self.edge_score(x_j - x_i) * self.edge_block(x_j)


class mGNN(nn.Module):
    def __init__(self, opts, input_dim):
        super(mGNN, self).__init__()
        groups = opts["groups"]
        self.max_stroke_num = opts["max_stroke_num"]
        self.paint_size = opts["paint_size"]
        self.grid_nums = opts["grid_nums"]
        self.dense_connect = opts["dense_connect"]
        self.multi_scale = opts["multi_scale"]

        gnn_groups = np.min([input_dim, groups])
        if self.dense_connect:
            block_configs = [
                [input_dim, 64, 1],
                [64 + input_dim, 128, gnn_groups],
                [64 + 128 + input_dim, 192, gnn_groups],
                [64 + 128 + 192 + input_dim, 256, gnn_groups],
                [64 + 128 + 192 + 256 + input_dim, 320, gnn_groups],
            ]
        else:
            block_configs = [
                [input_dim, 64, 1],
                [64, 128, gnn_groups],
                [128, 192, gnn_groups],
                [192, 256, gnn_groups],
                [256, 320, gnn_groups],
            ]
        last_dim = block_configs[-1][1]
        self.blocks = []
        for bi, block in enumerate(block_configs):
            block_name = "layer_{:02d}".format(bi)
            self.add_module(block_name, GroupEdgeConv(*block))
            self.blocks.append(block_name)

        conv_groups = np.min([1 * groups, 32])
        self.grid_convs = []

        if self.multi_scale:
            for grid_num in self.grid_nums:
                grid_block = "grid_conv_{}".format(grid_num)
                self.add_module(grid_block, GridConv(last_dim, 3, conv_groups, grid_num, opts["pool"]))
                self.grid_convs.append(grid_block)
        else:
            grid_block = "direct_conv"
            self.add_module(grid_block, nn.Sequential(
                nn.Conv2d(last_dim, last_dim * len(self.grid_nums), 1, bias=False, groups=conv_groups),
                nn.BatchNorm2d(last_dim * len(self.grid_nums)),
                h_swish(), )
                            )
            self.grid_convs.append(grid_block)

        self.conv2fc = nn.Sequential(
            nn.Conv2d(last_dim * len(self.grid_nums), 1280, 1, bias=False, groups=conv_groups),
            nn.BatchNorm2d(1280),
            h_swish(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(1280, opts["num_class"]),
        )

    def forward(self, data):
        x = data["x"]
        edge_index = data["edge_index"]
        batch = data.batch.to(x.device)
        pos = x[:, :2].detach() * self.paint_size
        x_pre = None

        for block in self.blocks:
            if not self.dense_connect:
                x = getattr(self, block)(x, edge_index)
            else:
                if not (x_pre is None):
                    x_pre = torch.cat([x, x_pre], dim=1)
                else:
                    x_pre = x.clone()
                x = getattr(self, block)(x_pre, edge_index)

        if self.multi_scale:
            x_grid = []
            for grid_block in self.grid_convs:
                x_grid.append(getattr(self, grid_block)(x, pos, batch, self.paint_size))
            x_grid = torch.cat(x_grid, dim=1)
        else:
            x_grid = global_mean_pool(x, batch).unsqueeze(-1).unsqueeze(-1)
            x_grid = getattr(self, self.grid_convs[0])(x_grid)

        x_grid = self.conv2fc(x_grid)

        x_grid = x_grid.squeeze(-1).squeeze(-1)
        return self.fc(x_grid), x_grid


if __name__ == "__main__":
    from utils.config import opts

    net = mGNN(opts, 4)
