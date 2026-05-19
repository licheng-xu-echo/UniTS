import torch
from .utils import Args
import numpy as np
from pathlib import Path
from rdkit import Chem
from qcbot.utils import symbol_pos_to_xyz_file
from .egnn.model import EGNN_dynamics_DiffMM
from .hiegnn.model import HiEGNN_dynamics_DiffMM
from .en_diffusion import EnVariationalDiffusion
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pt = Chem.GetPeriodicTable()

def load_model(model_path,ckpt_file="best_full_model.pth",device=DEVICE):
    
    args = np.load(f"{model_path}/args.npy",allow_pickle=True).item()
    args = Args(**args)
    
    if args.condition_time:
        if hasattr(args,'time_embed_dim'):
            dynamics_in_node_nf = args.in_node_nf + args.time_embed_dim
        else:
            dynamics_in_node_nf = args.in_node_nf + 1
    else:
        dynamics_in_node_nf = args.in_node_nf

    if args.dynamic_type == 'egnn' or not hasattr(args,'dynamic_type'):
        net_dynamics = EGNN_dynamics_DiffMM(
                            n_dims=args.n_dims,
                            in_node_nf=dynamics_in_node_nf, 
                            context_node_nf=args.context_node_nf,
                            rct_cent_node_nf=args.rct_cent_node_nf if hasattr(args,'rct_cent_node_nf') else 0,
                            hidden_nf=args.hidden_nf, 
                            act_fn=torch.nn.SiLU(), 
                            n_layers=args.n_layers, 
                            attention=args.attention,
                            condition_time=args.condition_time, 
                            tanh=args.tanh, 
                            norm_constant=args.norm_constant,
                            inv_sublayers=args.inv_sublayers, 
                            sin_embedding=args.sin_embedding, 
                            normalization_factor=args.normalization_factor, 
                            aggregation_method=args.aggregation_method,
                            scale_range=args.scale_range,
                            add_angle_info=args.add_angle_info,
                            add_fpfh=args.add_fpfh if hasattr(args,'add_fpfh') else False,
                            )

    elif args.dynamic_type == 'hiegnn':
            net_dynamics = HiEGNN_dynamics_DiffMM(
                    in_node_nf=args.in_node_nf,
                    context_node_nf=args.context_node_nf, 
                    rct_cent_node_nf=args.rct_cent_node_nf,
                    n_dims=args.n_dims,
                    layers=args.n_layers,
                    condition_time=True,
                    time_embed_dim=4,
                    add_fpfh=False,
                    otf_graph=True,
                    max_neighbors=64,
                    max_radius=5,
                    max_num_elements=90,
                    sphere_channels=args.equif_sphere_channels,
                    attn_hidden_channels=args.equif_attn_hidden_channels,
                    num_heads=args.equif_num_heads,
                    attn_alpha_channels=args.equif_attn_alpha_channels,
                    attn_value_channels=args.equif_attn_value_channels,
                    ffn_hidden_channels=args.equif_ffn_hidden_channels,
                    lmax_list=[args.lmax if hasattr(args,"lmax") else 6],
                    mmax_list=[args.mmax if hasattr(args,"mmax") else 2],
                    add_node_feat=args.equif_add_node_feat,
            )
    else:
        raise NotImplementedError

    vdm = EnVariationalDiffusion(dynamics=net_dynamics,
                                n_dims=args.n_dims,
                                in_node_nf=args.in_node_nf,
                                context_node_nf=args.context_node_nf,
                                rct_cent_node_nf=args.rct_cent_node_nf if hasattr(args,'rct_cent_node_nf') else 0,
                                timesteps=args.diffusion_steps, 
                                parametrization=args.diffusion_parametrization, 
                                noise_schedule=args.diffusion_noise_schedule,
                                noise_precision=args.diffusion_noise_precision, 
                                loss_type=args.diffusion_loss_type, 
                                norm_values=args.norm_values,
                                norm_biases=args.norm_biases,
                                enc_num_layers=args.enc_num_layers,
                                time_sample_method=args.time_sample_method,
                                time_sample_power=args.time_sample_power,
                                min_sample_power=args.min_sample_power,
                                max_sample_power=args.max_sample_power,
                                degree_as_continuous=args.degree_as_continuous,
                                dynamic_context=args.dynamic_context,
                                dynamic_context_temperature=args.dynamic_context_temperature,
                                loss_calc=args.loss_calc,
                                tot_x_mae=args.tot_x_mae,
                                mol_encoder_type=args.mol_encoder_type,
                                use_context=args.use_context,
                                enc_gnn_aggr=args.enc_gnn_aggr if hasattr(args,'enc_gnn_aggr') else 'add', 
                                enc_bond_feat_red=args.enc_bond_feat_red if hasattr(args,'enc_bond_feat_red') else 'mean', 
                                enc_JK=args.enc_JK if hasattr(args,'enc_JK') else 'last', 
                                enc_drop_ratio=args.enc_drop_ratio if hasattr(args,'enc_drop_ratio') else 0.0, 
                                enc_node_readout=args.enc_node_readout if hasattr(args,'enc_node_readout') else 'sum',
                                use_rct_cent=args.use_rct_cent if hasattr(args,'use_rct_cent') else False,
                                rct_cent_encoder_type=args.rct_cent_encoder_type if hasattr(args,'rct_cent_encoder_type') else 'gcn',
                                rct_cent_readout=args.rct_cent_readout if hasattr(args,'rct_cent_readout') else 'mean',
                                #add_angle_info=args.add_angle_info,
                                add_fpfh=args.add_fpfh if hasattr(args,'add_fpfh') else False,
                                focus_reaction_center= args.focus_reaction_center if hasattr(args,'focus_reaction_center') else False,
                                rc_loss_weight=args.rc_loss_weight if hasattr(args,'rc_loss_weight') else 2,
                                rc_distance_weight=args.rc_distance_weight if hasattr(args,'rc_distance_weight') else 3,
                                rc_angle_weight=args.rc_angle_weight if hasattr(args,'rc_angle_weight') else 2,
                                use_init_coords=args.use_init_coords if hasattr(args,'use_init_coords') else False,
                                fix_step_from_init=args.fix_step_from_init if hasattr(args,'fix_step_from_init') else 500,
                                sample_fix=False,
                                seg_sample=False,
                                last_step_alldif=400,
                                last_refine_ratio=2,
                                )
    model = vdm.to(device)
    model_params = torch.load(f"{model_path}/{ckpt_file}")
    model.load_state_dict(model_params['model'],strict=False)
    model.eval()
    
    return args,model


def load_inference_results(result_path):
    return np.load(result_path, allow_pickle=True).item()


def _to_bool_mask(node_mask_row):
    mask = np.asarray(node_mask_row)
    if mask.ndim == 2 and mask.shape[-1] == 1:
        mask = mask[:, 0]
    return mask.astype(bool)


def _to_symbols(atom_numbers):
    atom_numbers = np.asarray(atom_numbers).reshape(-1)
    return [pt.GetElementSymbol(int(atom_num)) for atom_num in atom_numbers]


def extract_inference_sample(results_dict, batch_idx, sample_idx, repeat_idx=0):
    batch_result = results_dict[batch_idx]
    node_mask = _to_bool_mask(batch_result["node_mask"][sample_idx])

    mol_atoms = np.asarray(batch_result["mol_atoms"][sample_idx]).reshape(-1)
    atoms = _to_symbols(mol_atoms[node_mask])

    truth_xyz = np.asarray(batch_result["truth_xyz"][sample_idx])
    if len(truth_xyz) != len(atoms):
        truth_xyz = truth_xyz[node_mask]

    pred_xyz = np.asarray(batch_result["pred_xyz"][repeat_idx, sample_idx])[node_mask]
    return atoms, truth_xyz, pred_xyz


def export_inference_sample_xyz(results_dict, batch_idx, sample_idx, output_dir,
                                repeat_idx=0, export_all_repeats=False, write_reference=False):
    output_dir = Path(output_dir)
    batch_result = results_dict[batch_idx]
    repeat_time = np.asarray(batch_result["pred_xyz"]).shape[0]

    atoms, truth_xyz, pred_xyz = extract_inference_sample(
        results_dict, batch_idx, sample_idx, repeat_idx=repeat_idx
    )

    sample_dir = output_dir / f"batch_{batch_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = []
    if write_reference:
        ref_path = sample_dir / f"sample_{sample_idx:03d}_reference.xyz"
        symbol_pos_to_xyz_file(atoms, truth_xyz, str(ref_path), title="reference")
        exported_paths.append(ref_path)

    if export_all_repeats:
        for ridx in range(repeat_time):
            _, _, pred_xyz = extract_inference_sample(
                results_dict, batch_idx, sample_idx, repeat_idx=ridx
            )
            gen_path = sample_dir / f"sample_{sample_idx:03d}_generated_repeat_{ridx:03d}.xyz"
            symbol_pos_to_xyz_file(atoms, pred_xyz, str(gen_path), title=f"generated_repeat_{ridx}")
            exported_paths.append(gen_path)
    else:
        gen_path = sample_dir / f"sample_{sample_idx:03d}_generated_repeat_{repeat_idx:03d}.xyz"
        symbol_pos_to_xyz_file(atoms, pred_xyz, str(gen_path), title=f"generated_repeat_{repeat_idx}")
        exported_paths.append(gen_path)

    return exported_paths


def export_inference_batch_xyz(results_dict, batch_idx, output_dir,
                               repeat_idx=0, export_all_repeats=False,
                               write_reference=False):
    batch_result = results_dict[batch_idx]
    batch_size = len(batch_result["truth_xyz"])
    exported_paths = []

    for sample_idx in range(batch_size):
        exported_paths.extend(
            export_inference_sample_xyz(
                results_dict,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                output_dir=output_dir,
                repeat_idx=repeat_idx,
                export_all_repeats=export_all_repeats,
                write_reference=write_reference
            )
        )

    return exported_paths
