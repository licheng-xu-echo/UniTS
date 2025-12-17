import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from ..data import NUM_BOND_TYPE,NUM_BOND_DIRECTION,NUM_BOND_STEREO,NUM_BOND_INRING,NUM_BOND_ISCONJ,NUM_ATOM_TYPE,NUM_DEGRESS_TYPE,NUM_FORMCHRG_TYPE,\
                           NUM_HYBRIDTYPE,NUM_CHIRAL_TYPE,NUM_AROMATIC_NUM,NUM_VALENCE_TYPE,NUM_Hs_TYPE,NUM_RS_TPYE,NUM_RADICAL_TYPES
from torch_scatter import scatter_add
import torch.nn.functional as F

class GINConv(MessagePassing):
    """
    Adapted from https://github.com/junxia97/Mole-BERT
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, out_dim, aggr = "add", bond_feat_red="mean"):
        self.aggr = aggr
        super().__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, out_dim))
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]
        self.bond_feat_red = bond_feat_red
    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
            #edge_embeddings = (self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4]))/5
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
            #edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings)
    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GCNConv(MessagePassing):
    # adapted from https://github.com/junxia97/Mole-BERT
    def __init__(self, emb_dim, aggr = "add", bond_feat_red="mean"):
        super().__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        self.edge_embedding3 = torch.nn.Embedding(NUM_BOND_STEREO, emb_dim)
        self.edge_embedding4 = torch.nn.Embedding(NUM_BOND_INRING, emb_dim)
        self.edge_embedding5 = torch.nn.Embedding(NUM_BOND_ISCONJ, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding5.weight.data)
        
        self.edge_embedding_lst = [self.edge_embedding1, self.edge_embedding2, self.edge_embedding3, self.edge_embedding4, self.edge_embedding5]

        self.aggr = aggr
        self.bond_feat_red = bond_feat_red

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        self_loop_attr = torch.zeros(x.size(0), len(self.edge_embedding_lst))
        self_loop_attr[:,0] = NUM_BOND_TYPE - 1  #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        edge_embeddings = []
        for i in range(edge_attr.shape[1]):
            edge_embeddings.append(self.edge_embedding_lst[i](edge_attr[:,i]))
        if self.bond_feat_red == "mean":
            edge_embeddings = torch.stack(edge_embeddings).mean(dim=0)
        elif self.bond_feat_red == "sum":
            edge_embeddings = torch.stack(edge_embeddings).sum(dim=0)
        else:
            raise ValueError("Invalid bond feature reduction method. Please choose from 'mean' or 'sum'")
        
        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)
        return self.propagate(edge_index=edge_index, aggr=self.aggr, x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GraphEncoder(nn.Module):
    
    def __init__(self, gnum_layer=4, emb_dim=256, gnn_aggr="add", bond_feat_red="mean", gnn_type='gcn', JK="last", drop_ratio=0.0, node_readout="sum"):
        super().__init__()
        self.gnum_layer = gnum_layer
        self.emb_dim = emb_dim
        self.gnn_aggr = gnn_aggr
        self.gnn_type = gnn_type
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.node_readout = node_readout
        assert self.gnum_layer >= 2, "Number of GraphEncoder layers must be greater than 1."

        self.x_embedding1 = torch.nn.Embedding(NUM_ATOM_TYPE, self.emb_dim)     ## atom type
        self.x_embedding2 = torch.nn.Embedding(NUM_DEGRESS_TYPE, self.emb_dim)  ## atom degree
        self.x_embedding3 = torch.nn.Embedding(NUM_FORMCHRG_TYPE, self.emb_dim) ## formal charge
        self.x_embedding4 = torch.nn.Embedding(NUM_HYBRIDTYPE, self.emb_dim)    ## hybrid type
        self.x_embedding5 = torch.nn.Embedding(NUM_CHIRAL_TYPE, self.emb_dim)   ## chiral type
        self.x_embedding6 = torch.nn.Embedding(NUM_AROMATIC_NUM, self.emb_dim)  ## aromatic or not
        self.x_embedding7 = torch.nn.Embedding(NUM_VALENCE_TYPE, self.emb_dim)  ## valence
        self.x_embedding8 = torch.nn.Embedding(NUM_Hs_TYPE, self.emb_dim)       ## number of Hs
        self.x_embedding9 = torch.nn.Embedding(NUM_RS_TPYE, self.emb_dim)       ## R or S
        self.x_embedding10 = torch.nn.Embedding(NUM_RADICAL_TYPES, self.emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding9.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding10.weight.data)
        
        self.x_emedding_lst = [self.x_embedding1,self.x_embedding2,self.x_embedding3,
                               self.x_embedding4,self.x_embedding5,self.x_embedding6,
                               self.x_embedding7,self.x_embedding8,self.x_embedding9,
                               self.x_embedding10]

        ## List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            if self.gnn_type.lower() == 'gcn':
                self.gnns.append(GCNConv(self.emb_dim,aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            elif self.gnn_type.lower() == 'gin':
                self.gnns.append(GINConv(self.emb_dim,self.emb_dim, aggr=self.gnn_aggr,bond_feat_red=bond_feat_red))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type.lower()}")
                
        ## List of batchnorms
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(self.gnum_layer):
            self.layer_norms.append(torch.nn.LayerNorm(self.emb_dim))
    
    def forward(self, x, edge_index, edge_attr):
        x_emb_lst = []
        for i in range(x.shape[1]):
            _x_emb = self.x_emedding_lst[i](x[:,i])
            x_emb_lst.append(_x_emb)
        if self.node_readout == 'sum':
            x_emb = torch.stack(x_emb_lst).sum(dim=0)
        elif self.node_readout == 'mean':
            x_emb = torch.stack(x_emb_lst).mean(dim=0)
        h_list = [x_emb]
        for layer in range(self.gnum_layer):
            h = self.gnns[layer](h_list[layer],edge_index=edge_index,edge_attr=edge_attr)
            h = self.layer_norms[layer](h)
            if layer == self.gnum_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=True)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=True)
            h_list.append(h)
        if self.JK == 'last':
            node_representation = h_list[-1]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "max":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == "mean":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.mean(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK == 'last+first':
            node_representation = h_list[-1] + h_list[0]
        else:
            raise NotImplementedError
        
        return node_representation