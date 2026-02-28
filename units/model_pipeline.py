from __future__ import annotations

from typing import Callable

import torch

from .egnn.model import EGNN_dynamics_DiffMM
from .encoder.graph_encoder import GraphEncoder
from .en_diffusion import EnVariationalDiffusion
from .hiegnn.model import HiEGNN_dynamics_DiffMM
from .registry import Registry


DYNAMICS_BUILDERS: Registry = Registry("dynamics_builder")
GRAPH_ENCODER_BUILDERS: Registry = Registry("graph_encoder_builder")


def register_dynamics_builder(name: str, builder: Callable) -> None:
    DYNAMICS_BUILDERS.register(name, builder)


def register_graph_encoder_builder(name: str, builder: Callable) -> None:
    GRAPH_ENCODER_BUILDERS.register(name, builder)


def _arg(args, key: str, default):
    return getattr(args, key) if hasattr(args, key) else default


@DYNAMICS_BUILDERS.register_fn("egnn")
def _build_egnn_dynamics(args):
    dynamics_in_node_nf = _arg(args, "in_node_nf", 0)
    if _arg(args, "condition_time", True):
        dynamics_in_node_nf = dynamics_in_node_nf + _arg(args, "time_embed_dim", 1)

    return EGNN_dynamics_DiffMM(
        n_dims=_arg(args, "n_dims", 3),
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=_arg(args, "context_node_nf", 0),
        rct_cent_node_nf=_arg(args, "rct_cent_node_nf", 0),
        hidden_nf=_arg(args, "hidden_nf", 64),
        act_fn=torch.nn.SiLU(),
        n_layers=_arg(args, "n_layers", 4),
        attention=_arg(args, "attention", False),
        condition_time=_arg(args, "condition_time", True),
        tanh=_arg(args, "tanh", False),
        norm_constant=_arg(args, "norm_constant", 0),
        inv_sublayers=_arg(args, "inv_sublayers", 2),
        sin_embedding=_arg(args, "sin_embedding", False),
        normalization_factor=_arg(args, "normalization_factor", 100),
        aggregation_method=_arg(args, "aggregation_method", "sum"),
        scale_range=_arg(args, "scale_range", 0.5),
        add_angle_info=_arg(args, "add_angle_info", False),
        add_fpfh=_arg(args, "add_fpfh", False),
    )


@DYNAMICS_BUILDERS.register_fn("hiegnn")
def _build_hiegnn_dynamics(args):
    return HiEGNN_dynamics_DiffMM(
        in_node_nf=_arg(args, "in_node_nf", 0),
        context_node_nf=_arg(args, "context_node_nf", 0),
        rct_cent_node_nf=_arg(args, "rct_cent_node_nf", 0),
        n_dims=_arg(args, "n_dims", 3),
        layers=_arg(args, "n_layers", 7),
        condition_time=True,
        time_embed_dim=4,
        add_fpfh=False,
        otf_graph=True,
        max_neighbors=64,
        max_radius=5,
        max_num_elements=90,
        sphere_channels=_arg(args, "equif_sphere_channels", 64),
        attn_hidden_channels=_arg(args, "equif_attn_hidden_channels", 64),
        num_heads=_arg(args, "equif_num_heads", 4),
        attn_alpha_channels=_arg(args, "equif_attn_alpha_channels", 32),
        attn_value_channels=_arg(args, "equif_attn_value_channels", 16),
        ffn_hidden_channels=_arg(args, "equif_ffn_hidden_channels", 64),
        lmax_list=[_arg(args, "lmax", 6)],
        mmax_list=[_arg(args, "mmax", 2)],
        add_node_feat=_arg(args, "equif_add_node_feat", True),
    )


@GRAPH_ENCODER_BUILDERS.register_fn("default")
def _build_default_graph_encoder(*, role: str, **kwargs):
    # role is reserved for custom builders that need separate behavior
    # for molecular encoder vs reaction-center encoder.
    _ = role
    return GraphEncoder(**kwargs)


def resolve_dynamics_type(args) -> str:
    return str(_arg(args, "dynamic_type", "egnn"))


def build_dynamics_from_args(args):
    key = resolve_dynamics_type(args)
    builder = DYNAMICS_BUILDERS.get(key)
    return builder(args)


def build_graph_encoder_factory_from_args(args):
    mol_builder_name = str(_arg(args, "graph_encoder_builder", "default"))
    rct_builder_name = str(_arg(args, "rct_cent_graph_encoder_builder", mol_builder_name))

    mol_builder = GRAPH_ENCODER_BUILDERS.get(mol_builder_name)
    rct_builder = GRAPH_ENCODER_BUILDERS.get(rct_builder_name)

    def factory(*, role: str, **kwargs):
        if role == "rct_center":
            return rct_builder(role=role, **kwargs)
        return mol_builder(role=role, **kwargs)

    return factory


def build_diffusion_model_from_args(args, dynamics=None):
    dynamics = dynamics or build_dynamics_from_args(args)
    graph_encoder_factory = build_graph_encoder_factory_from_args(args)

    return EnVariationalDiffusion(
        dynamics=dynamics,
        n_dims=_arg(args, "n_dims", 3),
        in_node_nf=_arg(args, "in_node_nf", 0),
        context_node_nf=_arg(args, "context_node_nf", 0),
        rct_cent_node_nf=_arg(args, "rct_cent_node_nf", 0),
        timesteps=_arg(args, "diffusion_steps", 1000),
        parametrization=_arg(args, "diffusion_parametrization", "eps"),
        noise_schedule=_arg(args, "diffusion_noise_schedule", "learned"),
        noise_precision=_arg(args, "diffusion_noise_precision", 1e-4),
        loss_type=_arg(args, "diffusion_loss_type", "vlb"),
        norm_values=_arg(args, "norm_values", (1.0, 1.0)),
        norm_biases=_arg(args, "norm_biases", (None, 0.0)),
        enc_num_layers=_arg(args, "enc_num_layers", 4),
        time_sample_method=_arg(args, "time_sample_method", "power"),
        time_sample_power=_arg(args, "time_sample_power", 4.0),
        min_sample_power=_arg(args, "min_sample_power", 2.0),
        max_sample_power=_arg(args, "max_sample_power", 6.0),
        degree_as_continuous=_arg(args, "degree_as_continuous", True),
        dynamic_context=_arg(args, "dynamic_context", False),
        dynamic_context_temperature=_arg(args, "dynamic_context_temperature", 1.0),
        loss_calc=_arg(args, "loss_calc", "all"),
        tot_x_mae=_arg(args, "tot_x_mae", True),
        mol_encoder_type=_arg(args, "mol_encoder_type", "gcn"),
        use_context=_arg(args, "use_context", False),
        enc_gnn_aggr=_arg(args, "enc_gnn_aggr", "add"),
        enc_bond_feat_red=_arg(args, "enc_bond_feat_red", "mean"),
        enc_JK=_arg(args, "enc_JK", "last"),
        enc_drop_ratio=_arg(args, "enc_drop_ratio", 0.0),
        enc_node_readout=_arg(args, "enc_node_readout", "sum"),
        sample_fix=False,
        seg_sample=False,
        last_step_alldif=400,
        last_refine_ratio=2,
        use_rct_cent=_arg(args, "use_rct_cent", False),
        rct_cent_encoder_type=_arg(args, "rct_cent_encoder_type", "gcn"),
        rct_cent_readout=_arg(args, "rct_cent_readout", "mean"),
        add_fpfh=_arg(args, "add_fpfh", False),
        focus_reaction_center=_arg(args, "focus_reaction_center", False),
        rc_loss_weight=_arg(args, "rc_loss_weight", 2.0),
        rc_distance_weight=_arg(args, "rc_distance_weight", 3.0),
        rc_angle_weight=_arg(args, "rc_angle_weight", 2.0),
        use_init_coords=_arg(args, "use_init_coords", False),
        fix_step_from_init=_arg(args, "fix_step_from_init", 500),
        graph_encoder_factory=graph_encoder_factory,
    )
