
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math

import argparse

import torch_geometric
from torch_geometric.utils import to_dense_adj

from layer import GraphConvolution
from model import GCN

def train(model, data, num_epochs, use_edge_index=False):
    if not use_edge_index:

        # Create the adjacency matrix
        adj = to_dense_adj(data.edge_index)[0]

    else:

        # Directly use edge_index, ignore this branch for now
        adj = data.edge_index
        
    

    # Set up the optimizer
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # A utility function to compute the accuracy
    def get_acc(outs, y, mask):
        return (outs[mask].argmax(dim=1) == y[mask]).sum().float() / mask.sum()

    best_acc_val = -1
    for epoch in range(num_epochs):

        # Zero grads -> forward pass -> compute loss -> backprop
        
        # train mode
        model.train()

        optimizer.zero_grad()
        outs = model(data.x, adj)

        # null_loss 

        loss = F.nll_loss(outs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Compute accuracies, print only if this is the best result so far

        # evaluation mode
        model.eval()

        # data.x = the features of the dataset

        outs = model(data.x, adj)

        # validation accuracy 
        acc_val = get_acc(outs, data.y, data.val_mask)

        # test accuracy 
        acc_test = get_acc(outs, data.y, data.test_mask)

        # print the accuracy if it’s incresed
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')

    print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val: {acc_val:.3f} | Test: {acc_test:.3f}')
    
if __name__==__main__:
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--use-edge-index', default=False)
  parser.add_argument('--num-epochs', default=False)
  #parser.add_argument('--data')
  
  parser.add_argument('--num-feat')
  parser.add_argument('--num-hid')
  parser.add_argument('--num-class')
  parser.add_argument('--dropout', default=0.5)
  

  Cora = torch_geometric.datasets.Planetoid(root='/', name='Cora')

  args = parser.parse_args()

  model = GCN(nfeat = args.num_feat, nhid = args.num_hid, nclass = args.num_class, dropout = args.dropout)
  train(model, data = Cora , num_epochs = args.num_epoch)
