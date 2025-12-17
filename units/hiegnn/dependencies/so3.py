"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


TODO:
    1. Simplify the case when `num_resolutions` == 1.
    2. Remove indexing when the shape is the same.
    3. Move some functions outside classes and to separate files.
"""

import math
import os
from typing import List

import torch
import torch.nn as nn
from torch.nn import Linear

try:
    from e3nn import o3
    from e3nn.o3 import FromS2Grid, ToS2Grid
except ImportError:
    pass

from .so3_base import CoefficientMappingModule, SO3_Grid
from .wigner import wigner_D


class SO3_Embedding(object):
    """
    Helper functions for performing operations on irreps embedding

    Args:
        length (int):           Batch size
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        num_channels (int):     Number of channels
        device:                 Device of the output
        dtype:                  type of the output tensors
    """

    def __init__(
        self,
        length: int,
        lmax_list: List[int],
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # super().__init__()
        # self.length = torch.jit.annotate(int, length)
        # self.embedding = torch.jit.annotate(torch.Tensor, torch.zeros(0))
        # self.lmax_list = torch.jit.annotate(list, [])
        # self.mmax_list = torch.jit.annotate(list, [])
        self.length = 0
        self.embedding = torch.zeros(0)
        self.lmax_list: List[int] = lmax_list
        self.mmax_list: List[int] = lmax_list

        self.num_channels: int = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int((lmax_list[i] + 1) ** 2)

        embedding = torch.zeros(
            (length, self.num_coefficients, self.num_channels),
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list)

    # Clone an embedding of irreps
    def clone(self):
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    # Initialize an embedding of irreps
    def set_embedding(self, embedding):
        self.length = embedding.shape[0]
        self.embedding = embedding

    # Set the maximum order to be the maximum degree
    def set_lmax_mmax(self, lmax_list: List[int], mmax_list: List[int]):
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    # Expand the node embeddings to the number of edges
    def _expand_edge(self, edge_index):
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    # Initialize an embedding of irreps of a neighborhood
    def expand_edge(self, edge_index):
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    # Compute the sum of the embeddings of the neighborhood
    def _reduce_edge(self, edge_index, num_nodes):
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    # Reshape the embedding l -> m
    def _m_primary(self, mapping):
        self.embedding = torch.einsum("nac, ba -> nbc", self.embedding, mapping.to_m)

    # Reshape the embedding m -> l
    def _l_primary(self, mapping):
        self.embedding = torch.einsum("nac, ab -> nbc", self.embedding, mapping.to_m)

    # Rotate the embedding
    def _rotate(self, SO3_rotation, lmax_list, mmax_list):

        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate(self.embedding, lmax_list[0], mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = int((self.lmax_list[i] + 1) ** 2)
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i]),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    # Rotate the embedding by the inverse of the rotation matrix
    def _rotate_inv(self, SO3_rotation, mappingReduced):

        if self.num_resolutions == 1:
            embedding_rotate = SO3_rotation[0].rotate_inv(self.embedding, self.lmax_list[0], self.mmax_list[0])
        else:
            offset = 0
            embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)
            for i in range(self.num_resolutions):
                num_coefficients = mappingReduced.res_size[i]
                embedding_i = self.embedding[:, offset : offset + num_coefficients]
                embedding_rotate = torch.cat(
                    [
                        embedding_rotate,
                        SO3_rotation[i].rotate_inv(embedding_i, self.lmax_list[i], self.mmax_list[i]),
                    ],
                    dim=1,
                )
                offset = offset + num_coefficients
        self.embedding = embedding_rotate

        # Assume mmax = lmax when rotating back
        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])
        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    # Compute point-wise spherical non-linearity
    def _grid_act(self, SO3_grid: List[List[SO3_Grid]], act, mappingReduced):
        offset = 0
        for i in range(self.num_resolutions):

            num_coefficients = mappingReduced.res_size[i]

            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_to_grid_mat(device=None)
            from_grid_mat = SO3_grid[self.lmax_list[i]][self.mmax_list[i]].get_from_grid_mat(device=None)

            x_grid = torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, x_grid)
            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    # Compute a sample of the grid
    def to_grid(self, SO3_grid, lmax=-1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            if self.num_resolutions == 1:
                x_res = self.embedding
            else:
                x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat = to_grid_mat_lmax[:, :, grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])]
            x_grid = torch.cat([x_grid, torch.einsum("bai, zic -> zbac", to_grid_mat, x_res)], dim=3)
            offset = offset + num_coefficients

        return x_grid

    # Compute irreps from grid representation
    def _from_grid(self, x_grid, SO3_grid: List[List[SO3_Grid]], lmax: int = -1):
        if lmax == -1:
            lmax = max(self.lmax_list)

        t_grid = SO3_grid[lmax][lmax]
        from_grid_mat_lmax = t_grid.get_from_grid_mat(device=None)
        grid_mapping = t_grid.mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            coe_i = grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i])
            from_grid_mat = from_grid_mat_lmax[:, :, coe_i]
            if self.num_resolutions == 1:
                temp = x_grid
            else:
                temp = x_grid[:, :, :, offset_channel : offset_channel + self.num_channels]
            x_res = torch.einsum("bai, zbac -> zic", from_grid_mat, temp)
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)

            if self.num_resolutions == 1:
                self.embedding = x_res
            else:
                self.embedding[:, offset : offset + num_coefficients] = x_res

            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels


class SO3_Rotation(torch.nn.Module):
    """
    Helper functions for Wigner-D rotations

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
    ):
        super().__init__()
        self.lmax = lmax
        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

    def set_wigner(self, rot_mat3x3):
        self.device, self.dtype = rot_mat3x3.device, rot_mat3x3.dtype
        # length = len(rot_mat3x3)
        self.wigner = self.RotationToWignerDMatrix(rot_mat3x3, 0, self.lmax)
        self.wigner_inv = torch.transpose(self.wigner, 1, 2).contiguous()
        self.wigner = self.wigner.detach()
        self.wigner_inv = self.wigner_inv.detach()

    # Rotate the embedding
    def rotate(self, embedding, out_lmax, out_mmax):
        out_mask = self.mapping.coefficient_idx(out_lmax, out_mmax)
        wigner = self.wigner[:, out_mask, :]
        return torch.bmm(wigner, embedding)

    # Rotate the embedding by the inverse of the rotation matrix
    def rotate_inv(self, embedding, in_lmax, in_mmax):
        in_mask = self.mapping.coefficient_idx(in_lmax, in_mmax)
        wigner_inv = self.wigner_inv[:, :, in_mask]
        wigner_inv_rescale = self.mapping.get_rotate_inv_rescale(in_lmax, in_mmax)
        wigner_inv = wigner_inv * wigner_inv_rescale
        return torch.bmm(wigner_inv, embedding)

    # Compute Wigner matrices from rotation matrix
    def RotationToWignerDMatrix(self, edge_rot_mat, start_lmax, end_lmax):
        x = edge_rot_mat @ edge_rot_mat.new_tensor([0.0, 1.0, 0.0])
        alpha, beta = o3.xyz_to_angles(x)
        R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ edge_rot_mat
        gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])

        size = (end_lmax + 1) ** 2 - (start_lmax) ** 2
        wigner = torch.zeros(alpha.shape[0], size, size, device=self.device, dtype=x.dtype)  # qq: fix double dtype bug
        start = 0
        for lmax in range(start_lmax, end_lmax + 1):
            block = wigner_D(lmax, alpha, beta, gamma)
            end = start + block.size()[1]
            wigner[:, start:end, start:end] = block
            start = end

        return wigner.detach()


class SO3_Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.lmax = lmax
        self.linear_list = torch.nn.ModuleList()
        for l in range(lmax + 1):
            if l == 0:
                self.linear_list.append(Linear(in_features, out_features, bias=bias))
            else:
                self.linear_list.append(Linear(in_features, out_features, bias=False))

    def forward(self, input_embedding, output_scale=None):
        out = []
        for l in range(self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            features = input_embedding.embedding.narrow(1, start_idx, length)
            features = self.linear_list[l](features)
            if output_scale is not None:
                scale = output_scale.narrow(1, l, 1)
                features = features * scale
            out.append(features)
        out = torch.cat(out, dim=1)

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=input_embedding.device,
            dtype=input_embedding.dtype,
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy())

        return out_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"


class SO3_LinearV2(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, lmax, bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding):

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum("bmi, moi -> bmo", input_embedding.embedding, weight)  # [N, (L_max + 1) ** 2, C_out]
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=input_embedding.device,
            dtype=input_embedding.dtype,
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy())

        return out_embedding

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"
