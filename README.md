# GraphVAE with GAT Reimplementation

This is an implementation of GraphVAE model, initial code taken from 
https://github.com/JiaxuanYou/graph-generation/blob/master/baselines/graphvae/

The code linked above, however, significantly differs from what is described in the paper. 
It also contains several errors preventing it from being runnable on the latest Pytorch version:
-  `adj_recon_loss(self, adj_truth, adj_pred)` have  `target` and `pred` passed to `F.binary_cross_entropy` in reverse order, this prevents gradient from passing
- encoder is commented out completely, and contains inproper dimensions for its GraphConv layers (namely BatchNorm has incorrect dimentions and is also applied to Sequence Length dim rather than channel dim)
- permuted adjacency matrix is not used in the reconstruction, instead original `adj` matrix is used

Deviations from the paper in the `JiaxuanYou` implementation
- no VAE for node and edge features (continuous features are not supported at all) and hence no reconstruction loss for them
- original formula for matching network similarity scores (Formula 4 in paper) is not used
- According to the paper, there should be a conditional VAE in the implementation, but it is a classic VAE

Here we present a somewhat correct (and now working) implementation (now - with the GAT layers - it was fixed) 
However support for node and edge features in reconstruction of VAE is still missing

For launching training on the dataset (cpu device by default) execute `train.py` file in the correct env, and also launch pip install -e in the folder. 
