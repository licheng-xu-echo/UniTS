import torch,math
import numpy as np
import torch.nn as nn
from .hiegnn import HiEGNN
from ..utils import remove_mean, remove_mean_with_mask, split_and_padding

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

class HiEGNN_dynamics_DiffMM(nn.Module):
    def __init__(self, in_node_nf, 
                 context_node_nf, 
                 rct_cent_node_nf,
                 n_dims=3,
                 layers=7,
                 condition_time=True,
                 time_embed_dim=4,
                 add_fpfh=False,
                 otf_graph=True,
                 max_neighbors=100,
                 max_radius=5,
                 max_num_elements=90,
                 sphere_channels=64,
                 attn_hidden_channels=64,
                 num_heads=4,
                 attn_alpha_channels=32,
                 attn_value_channels=16,
                 ffn_hidden_channels=64,
                 lmax_list=[6],
                 mmax_list=[2],
                 add_node_feat=True,
                 ):
        super().__init__()

        self.n_dims = n_dims
        self.context_node_nf = context_node_nf
        self.rct_cent_node_nf = rct_cent_node_nf
        self.condition_time = condition_time
        self.time_embed_dim = time_embed_dim
        self._edges_dict = {}

        input_feat_dim = in_node_nf + context_node_nf + rct_cent_node_nf 
        if condition_time:
            input_feat_dim += time_embed_dim
        if add_fpfh:
            input_feat_dim += 33

        self.time_embed = SinusoidalEmbedding(self.time_embed_dim)
 
        self.equif = HiEGNN(
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            max_radius=max_radius,
            max_num_elements=max_num_elements,
            num_layers=layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            add_node_feat=add_node_feat,
            feat_dim=input_feat_dim,
        )

        #self.atom_embedding = nn.Linear(in_node_nf, in_node_nf)

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def _forward(self, xh, t, atom_mask, edge_mask, context, batch):
        bs, n_nodes, dims = xh.shape
        #h_dims = dims - self.n_dims
        if not self.equif.otf_graph:
            edges = self.get_adj_matrix(n_nodes, bs, xh.device)
            edges = torch.stack(edges).long()
        else:
            edges = None
        #atom_mask = atom_mask.view(bs * n_nodes, 1)
        # edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        # xh = xh.view(bs * n_nodes, -1).clone() * atom_mask
        #print(xh.shape,batch.shape)
        xh = xh[atom_mask.bool().squeeze(2)]
        x = xh[:, :self.n_dims].clone()  # 噪声坐标
        h = xh[:, self.n_dims:].clone()  # 节点特征（原子类型）

        # 1. 准备节点特征
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = self.time_embed(t.item() * torch.ones(xh.shape[0], 1, device=xh.device))
            else:
                #print("This way")
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time[atom_mask.bool().squeeze(2)]
                h_time = self.time_embed(h_time).squeeze(1)
            #print(f"h.shape {h.shape}, h_time.shape {h_time.shape}")
            h = torch.cat([h, h_time], dim=1) if len(h_time.shape) == 2 else torch.cat([h, h_time.squeeze(1)], dim=1)

        if context is not None:
            context = context[atom_mask.bool().squeeze(2)]
            h = torch.cat([h, context], dim=1)
        
        
        atom_mask = atom_mask.view(bs * n_nodes, 1)
        #x_masked = x * atom_mask
        #feat_masked = h * atom_mask

        predicted_noise = self.equif(
            z=h[:,10],
            pos=x,
            feat=h,
            batch=batch,
            edge_index=edges,
        )
        predicted_noise,_ = split_and_padding(predicted_noise,batch,3)
        predicted_noise = predicted_noise.view(bs*n_nodes, 3)
        predicted_noise = predicted_noise * atom_mask
        vel = predicted_noise.view(bs, n_nodes, -1)
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting HiEGNN output to zero.')
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