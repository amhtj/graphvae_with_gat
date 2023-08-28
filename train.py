import argparse
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch
import numpy as np
import torch_geometric as tg
from torch_geometric import datasets

from graphvae.model import GraphVAE

CUDA = 2

LR_milestones = [500, 1000]


def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == "id":
        input_dim = max_num_nodes
    elif args.feature_type == "deg":
        input_dim = 1
    elif args.feature_type == "struct":
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model


def train(args, dataloader, model, max_num_nodes=10, device='cpu'):
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=1e-6)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=1e-5)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    for epoch in range(100):
        for batch_idx, data in enumerate(dataloader):
            
            optimizer.zero_grad()
            
            adj = tg.utils.to_dense_adj(data.edge_index)
            degs = adj.sum(dim=2, keepdim=True) - 1
            delta = max_num_nodes - adj.shape[-1]
            adj = torch.nn.functional.pad(adj.transpose(2,1), (0,delta,0,delta), "constant", 0).transpose(2,1)
            degs = torch.nn.functional.pad(degs.transpose(2,1), (0,delta), "constant", 0).transpose(2,1)

            adj_input = Variable(adj).to(device=device)
            features = Variable(degs).to(device=device)

            loss = model(features, adj_input)
            print("Epoch: ", epoch, ", Iter: ", batch_idx, ", Loss: ", loss)

            optimizer.step()
            scheduler.step()
            break


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )

    parser.set_defaults(
        dataset="qm7b",
        feature_type="deg",
        lr=0.00001,
        batch_size=16,
        num_workers=1,
        max_num_nodes=-1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA)
    print("CUDA", CUDA)
    device = 'cpu'
    ### running log

    save_dataset_to = '../../../Data/DL/bench/QM7b'
    dataset = datasets.QM7b(save_dataset_to)

    max_num_nodes = 0
    for gi in dataset:
        max_num_nodes = max(max_num_nodes, gi.num_nodes)

    graphs_len = len(dataset)
    
    graphs_train = dataset[0: int(0.85 * graphs_len)]
    graphs_test = dataset[int(0.85 * graphs_len): ]


    print(
        "total graph num: {}, training set: {}, test set: {}".format(len(dataset), len(graphs_train), len(graphs_test))
    )
    print("max number node: {}".format(max_num_nodes))

    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size,
    #        replacement=False)

    # current limitation due to batched adj padding missing
    # batch_size=prog_args.batch_size == 1
    dataset_loader = tg.loader.DataLoader(
        dataset, batch_size=1, num_workers=prog_args.num_workers
    )
    model = build_model(prog_args, max_num_nodes).to(device=device)
    train(prog_args, dataset_loader, model, max_num_nodes=max_num_nodes, device=device)


if __name__ == "__main__":
    main()
