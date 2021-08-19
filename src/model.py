improt torch 
from layer import GraphConvolution

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, fts, adj):
        fts = torch.nn.functional.relu(self.gc1(fts, adj))
        fts = torch.nn.functional.dropout(fts, self.dropout, training=self.training)
        #fts = torch.nn.functional.dropout(fts, p=0.5, training=True)
        fts = self.gc2(fts, adj)
        return torch.nn.functional.log_softmax(fts, dim=1)
      
