import torch
import numpy as np
import torch.nn as nn
from .egnn_new import EGNN
from ..utils import remove_mean, remove_mean_with_mask
from torch.nn import functional as F

class EGNN_dynamics_DiffMM(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, rct_cent_node_nf,
                 n_dims=3,
                 hidden_nf=64,
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 scale_range=0.5, add_angle_info=False, add_fpfh=False):
        super().__init__()

        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf + rct_cent_node_nf if not add_fpfh else in_node_nf + context_node_nf + rct_cent_node_nf + 33,
            hidden_nf=hidden_nf, 
            act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            scale_range=scale_range,add_angle_info=add_angle_info)
        self.in_node_nf = in_node_nf


        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.rct_cent_node_nf = rct_cent_node_nf
        #self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.context_dim = context_node_nf
        self.rct_cent_context_dim = rct_cent_node_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def _forward(self, xh, t, atom_mask, edge_mask, context, batch):

        bs, n_nodes, dims = xh.shape # x 
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, xh.device)
        edges = [ed.to(xh.device) for ed in edges]
        atom_mask = atom_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * atom_mask
        x = xh[:, 0:self.n_dims].clone() # xyz
        h = xh[:, self.n_dims:].clone()  # node features, no noises
        # print(h.shape,self.n_dims)
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(x[:, 0:1]).fill_(t.item()) 
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)       
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)
        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf+self.rct_cent_node_nf)
            h = torch.cat([h, context], dim=1) # time and context

        pos_final = self.egnn(h, x, edges, node_mask=atom_mask, edge_mask=edge_mask) 
        vel = (pos_final - x) * atom_mask  # This masking operation is redundant but just in case
        
        vel = vel.view(bs, n_nodes, -1)
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if atom_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, atom_mask.view(bs, n_nodes, 1).float())

        return vel

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
