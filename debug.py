from graphrnn.baselines.graphvae.train import GraphAdjSampler, build_model, train
from graphrnn.baselines.graphvae.model import GraphVAE
import networkx as nx 
import torch 



if __name__ == '__main__':
    #freeze_support()
    graphs = []
    for i in range(2, 3):
        for j in range(2, 3):
            graphs.append(nx.grid_2d_graph(i, j))
    num_graphs_raw = len(graphs)

    max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    graphs_len = len(graphs)
    print(
        "Number of graphs removed due to upper-limit of number of nodes: ",
        num_graphs_raw - graphs_len,
    )
    graphs_test = graphs[int(0.8 * graphs_len) :]
    # graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_train = graphs

    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(max_num_nodes))

    dataset = GraphAdjSampler(
        graphs_train, max_num_nodes, features="deg"
    )
    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size,
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1
    )
    model = GraphVAE(1, 64, 256, max_num_nodes)
    train({}, dataset_loader, model)