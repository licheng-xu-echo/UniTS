from torch import nn
import torch
import math
from torch_geometric.typing import SparseTensor
from torch_scatter import scatter

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij

class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0,
                 scale_range=0.5):
        super(EquivariantUpdate, self).__init__()
        #assert self.axis_model in ["as_pos","axis"]
        self.tanh = tanh
        self.coords_range = coords_range
        self.scale_range = scale_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord


    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):

        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord

class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum',
                 scale_range=0.5,add_angle_info=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.scale_range = scale_range
        self.add_angle_info = add_angle_info

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method)) 
        #if 'pos' in self.task_type:
        self.add_module("gcl_equiv_pos", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                    coords_range=self.coords_range_layer,
                                                    normalization_factor=self.normalization_factor,
                                                    aggregation_method=self.aggregation_method)) 

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # edge_attr default only distances
        
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        #print("distances requires grad",distances.requires_grad)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        if self.add_angle_info:
            angles = calc_angles(x, torch.stack(edge_index))
            edge_dist_attr = torch.cat([distances, angles, edge_attr], dim=1)                       # 2 or 2 * sim_emb_dim
            #print("angles requires grad",angles.requires_grad)
        else:
            edge_dist_attr = torch.cat([distances, edge_attr], dim=1)                               # 2 or 2 * sim_emb_dim
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_dist_attr, node_mask=node_mask, edge_mask=edge_mask) # no node_attr?
        x = self._modules["gcl_equiv_pos"](h, x, edge_index, coord_diff, edge_dist_attr, node_mask, edge_mask)
        #print("x requires grad",x.requires_grad)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 scale_range=0.5,add_angle_info=False):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.scale_range = scale_range
        self.add_angle_info = add_angle_info

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2
        if self.add_angle_info:
            edge_feat_nf += 2
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               scale_range=self.scale_range,
                                                               add_angle_info=self.add_angle_info
                                                               ))

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        if not self.add_angle_info:
            for i in range(0, self.n_layers):
                h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
        else:
            angles = calc_angles(x, torch.stack(edge_index))
            #print("angles",angles.requires_grad)
            for i in range(0, self.n_layers):
                h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=torch.cat([distances,angles],dim=-1))

        return x

class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h

class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff

def one_principal_axis2diff(x, edge_index, norm_constant=1e-8):
    row, col = edge_index
    axis_row = x[row]
    axis_col = x[col]
    axis_row_norm = axis_row / (torch.norm(axis_row, dim=-1, keepdim=True) + norm_constant)
    axis_col_norm = axis_col / (torch.norm(axis_col, dim=-1, keepdim=True) + norm_constant)
    axis_diff = axis_row - axis_col
    axis_angle = (axis_row_norm * axis_col_norm).sum(dim=-1,keepdim=True)
    axis_diff_norm = axis_diff / (torch.norm(axis_diff, dim=-1, keepdim=True) + norm_constant)
    return axis_angle, axis_diff_norm

def principal_axis2diff(mat_x, edge_index, norm_constant=1e-8):
    row, col = edge_index
    axis_row = mat_x[row]
    axis_col = mat_x[col]
    axis_row_norm = axis_row / (torch.norm(axis_row, dim=-1, keepdim=True) + norm_constant)
    axis_col_norm = axis_col / (torch.norm(axis_col, dim=-1, keepdim=True) + norm_constant)
    #print(mat_x.shape,axis_row_norm.shape, axis_col_norm.shape)
    rot_row_col = torch.bmm(axis_col_norm.transpose(1, 2), axis_row_norm)
    trace = torch.einsum('bii->b', rot_row_col)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    relative_angle = torch.acos(cos_theta_clamped)
    return relative_angle

def axis2diff(x, edge_index, norm_constant=1e-8):
    row, col = edge_index
    axis_row = x[row]
    axis_col = x[col]
    axis_row_norm = axis_row / (torch.norm(axis_row, dim=-1, keepdim=True) + norm_constant)
    axis_col_norm = axis_col / (torch.norm(axis_col, dim=-1, keepdim=True) + norm_constant)
    axis_simi = (axis_row_norm * axis_col_norm).sum(dim=-1, keepdim=True)
    axis_diff = axis_row - axis_col
    axis_diff_norm = axis_diff / (torch.norm(axis_diff, dim=-1, keepdim=True) + norm_constant)
    return axis_simi, axis_diff_norm

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    elif aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    else:
        raise ValueError(f'Invalid aggregation method: {aggregation_method}')
    return result

def compute_edge_angles(x, full_edges, act_edges):
    """
    计算边之间的夹角平均值，并将结果映射到全边集
    
    Args:
        x: 原子坐标, shape (node_num, 3)
        full_edges: 全边集, shape (2, edge_num)
        act_edges: 实际存在的边, shape (2, act_edge_num)
        
    Returns:
        angles: 每条边的夹角平均值, shape (edge_num, 1)
    """
    device = x.device
    node_num = x.shape[0]
    act_edge_num = act_edges.shape[1]
    
    sum_angles = torch.zeros(act_edge_num, device=device, dtype=torch.float)
    count_angles = torch.zeros(act_edge_num, device=device, dtype=torch.float)
    
    start_nodes = torch.unique(act_edges[0])
    
    for i in start_nodes:
        mask = (act_edges[0] == i)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        k = indices.numel()
        if k < 2: 
            continue
            
        ends = act_edges[1, mask]
        
        vi = x[i]
        v_ends = x[ends]
        vectors = v_ends - vi.unsqueeze(0)  # (k, 3)
        
        norms = torch.norm(vectors, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        vectors_normed = vectors / norms
        
        dot_matrix = torch.mm(vectors_normed, vectors_normed.t())
        cos_matrix = torch.clamp(dot_matrix, -1.0+1e-6, 1.0-1e-6)
        angle_matrix = torch.acos(cos_matrix)
        
        angle_matrix.fill_diagonal_(0.0)
        
        local_sum = angle_matrix.sum(dim=1)
        local_count = torch.ones(k, device=device) * (k - 1)
        
        sum_angles[indices] += local_sum
        count_angles[indices] += local_count
    
    min_vals = torch.min(act_edges, dim=0)[0]
    max_vals = torch.max(act_edges, dim=0)[0]
    undir_keys = min_vals * node_num + max_vals
    
    unique_keys, inverse_indices = torch.unique(undir_keys, return_inverse=True)
    num_undir = unique_keys.size(0)
    
    undir_sum = torch.zeros(num_undir, device=device).scatter_add(
        0, inverse_indices, sum_angles)
    undir_count = torch.zeros(num_undir, device=device).scatter_add(
        0, inverse_indices, count_angles)
    
    undir_avg = undir_sum / undir_count
    undir_avg[undir_count == 0] = 0.0
    
    act_edges_avg = undir_avg[inverse_indices]

    full_keys = full_edges[0] * node_num + full_edges[1]
    act_keys = act_edges[0] * node_num + act_edges[1]

    sorted_full_keys, sorted_indices = torch.sort(full_keys)
    pos = torch.searchsorted(sorted_full_keys, act_keys)
    full_positions = sorted_indices[pos]

    angles = torch.zeros(full_edges.shape[1], 1, device=device, dtype=torch.float)
    angles[full_positions] = act_edges_avg.unsqueeze(1)
    
    return angles

def triplets(
    edge_index,
    num_nodes
):
    # from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/dimenet.py
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

def pairs_to_ids(edges, num_nodes):
    return edges[0] * num_nodes + edges[1]

def calc_angles(x, edges):
    #device = x.device
    eps = 1e-6
    node_num = x.shape[0]
    i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edges, node_num)
    pos_jk, pos_ij = x[idx_j] - x[idx_k], x[idx_i] - x[idx_j]

    norm_ij = torch.norm(pos_ij, dim=1, keepdim=True) + eps
    norm_jk = torch.norm(pos_jk, dim=1, keepdim=True) + eps
    pos_ij_normalized = pos_ij / norm_ij
    pos_jk_normalized = pos_jk / norm_jk

    cos_theta = (pos_ij_normalized * pos_jk_normalized).sum(dim=-1)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_theta) / torch.pi
    
    all_indices = torch.cat([idx_kj, idx_ji])
    all_angles = torch.cat([angle, angle])
    
    return scatter(all_angles,all_indices,reduce='mean').unsqueeze(1)