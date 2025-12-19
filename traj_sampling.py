import torch,os,logging,argparse,time
import numpy as np
from datetime import datetime
from units.data import MultiDataset,get_idx_split,get_fnidx_split,MultiDatasetV2,MultiDataset1x
from torch_geometric.data import DataLoader
from units.en_diffusion import EnVariationalDiffusion
from units.egnn.model import EGNN_dynamics_DiffMM
from units.hiegnn.model import HiEGNN_dynamics_DiffMM
from units.utils import split_and_padding,setup_logger,str_to_bool,Args
from rdkit import Chem

pt = Chem.GetPeriodicTable()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_root', type=str, default="./model_path")
    parser.add_argument('--model_tag', type=str, default="units_hiegnn")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ckpt_file', type=str, default="best_full_model.pth")
    parser.add_argument('--fix_noise', type=str_to_bool, default=False)
    parser.add_argument('--repeat_time', type=int, default=10)
    parser.add_argument('--save_traj', type=str_to_bool, default=True)
    parser.add_argument('--eval_root', type=str, default="./sample_traj")

    start_time = time.time()
    eval_args = parser.parse_args()
    
    eval_root = eval_args.eval_root
    model_tag = eval_args.model_tag
    eval_save_path = f"{eval_root}/traj-sample-repeat-{eval_args.repeat_time}-{model_tag}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(eval_save_path, exist_ok=True)
    setup_logger(eval_save_path)
    args = np.load(f"{eval_args.model_root}/{model_tag}/args.npy",allow_pickle=True).item()
    args = Args(**args)
    if not hasattr(args,'dataset_type') or args.dataset_type == 1:
        dataset = MultiDataset(root=args.dataset_path,name_regrex=args.name_regrex)
    elif args.dataset_type == 2:
        dataset = MultiDatasetV2(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=args.link_rc,
                                 data_enhance=args.data_enhance if hasattr(args,'data_enhance') else False)
    elif args.dataset_type == 3:
        dataset = MultiDataset1x(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=args.link_rc,
                                 iptgraph_type=args.iptgraph_type)

    sf_index = list(range(len(dataset)))
    np.random.seed(args.seed)
    np.random.shuffle(sf_index)
    if args.data_truncated > 0:
        dataset = dataset[sf_index[:args.data_truncated]]
    else:
        dataset = dataset[sf_index]

    if not args.specific_test:
        train_ratio = args.train_ratio if hasattr(args,'train_ratio') else 0.8
        valid_ratio = args.valid_ratio if hasattr(args,'valid_ratio') else 0.1
        if not hasattr(args,"data_enhance") or not args.data_enhance:
            split_ids_map = get_idx_split(len(dataset), 
                                            int(train_ratio*len(dataset)), 
                                            int(valid_ratio*len(dataset)), 
                                            args.seed)
        else:
            dataset[0]
            split_ids_map = get_fnidx_split(dataset.data.fn_id,
                                            train_ratio,
                                            valid_ratio,
                                            args.seed)
        test_dataset = dataset[split_ids_map['test']]
    else:
        assert args.test_name_regrex != ''
        print("[INFO] Splitting dataset into train, valid. Test is specified")
        assert abs(args.train_ratio + args.valid_ratio - 1.0) < 1e-6
        if not hasattr(args,'dataset_type') or args.dataset_type == 1:
            test_dataset = MultiDataset(root=args.dataset_path, name_regrex=args.test_name_regrex)
        elif args.dataset_type == 2:
            test_dataset = MultiDatasetV2(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc)
        elif args.dataset_type == 3:
            test_dataset = MultiDataset1x(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc, 
                                          iptgraph_type=args.iptgraph_type)

    logging.info(f"test set size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=eval_args.batch_size, shuffle=False, num_workers=args.num_workers)
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
    model_params = torch.load(f"{eval_args.model_root}/{model_tag}/{eval_args.ckpt_file}")
    model.load_state_dict(model_params['model'],strict=False)

    logging.info(str(eval_args))
    logging.info(f"[INFO] Model parameters from '{model_tag}' will be sued")

    model.eval()
    tot_batch = len(test_dataloader)
    batch_idx = 0
    torch.manual_seed(args.seed)
    logging.info(f"[INFO] random seed = {args.seed}")
    with torch.no_grad():
        results = {}
        for data in test_dataloader:
            batch_idx += 1
            logging.info(f"[INFO] batch_idx = {batch_idx}/{tot_batch} ...")
            data = data.to(device)
            x_truth,node_mask = split_and_padding(data.mol_coords.cpu(),data.batch.cpu(),3)
            mol_xyz_truth_lst = []
            for data_idx in range(len(x_truth)):
                mol_xyz_truth = x_truth[data_idx]
                mol_node_mask = node_mask[data_idx]
                mol_xyz_truth = mol_xyz_truth[mol_node_mask.bool().squeeze()].numpy()
                mol_xyz_truth_lst.append(mol_xyz_truth)
                
            pred_final_lst = []
            pred_traj_lst = []
            for _ in range(eval_args.repeat_time):
                logging.info(f"[INFO] Sample traj {_+1} / {eval_args.repeat_time}...")
                x_traj,mol_atoms,node_mask = model.sample_traj(data,fix_noise=eval_args.fix_noise)
                pred_final = x_traj[-1]
                pred_final_lst.append(pred_final)
                pred_traj_lst.append(x_traj)
            pred_final_lst = np.array(pred_final_lst)
            pred_traj_lst = np.array(pred_traj_lst)
            if not eval_args.save_traj:
                batch_info = {"batch_idx": batch_idx, "truth_xyz": mol_xyz_truth_lst, "pred_xyz": pred_final_lst, "node_mask": node_mask, "mol_atoms": mol_atoms}
            else:
                batch_info = {"batch_idx": batch_idx, "truth_xyz": mol_xyz_truth_lst, "pred_xyz": pred_final_lst, "pred_traj": pred_traj_lst, "node_mask": node_mask, "mol_atoms": mol_atoms}
            results[batch_idx] = batch_info
            logging.info(f"[INFO] Saving results...")
            if not eval_args.save_traj:
                np.save(f"{eval_save_path}/truth_pred.npy", 
                        results)
            else:
                np.save(f"{eval_save_path}/truth_pred_traj.npy", 
                        results)

    
    end_time = time.time()
    logging.info(f"[INFO] all sampling trajs are saved, repeat {eval_args.repeat_time} time, time eclapsed: {end_time - start_time:.4f} seconds")
    
if __name__ == "__main__":
    main()