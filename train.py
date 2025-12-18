import torch
from units.data import MultiDataset,get_idx_split,get_fnidx_split,MultiDatasetV2,MultiDataset1x
from copy import deepcopy
from torch_geometric.data import DataLoader
from units.utils import get_optim,Queue,EMA,setup_logger,str_to_bool,str_to_bool,Args
from units.en_diffusion import EnVariationalDiffusion
from units.egnn.model import EGNN_dynamics_DiffMM
from units.hiegnn.model import HiEGNN_dynamics_DiffMM
from units.train import train_epoch,test
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
from units.scheduler import NoamLR
import numpy as np
import argparse
import os,logging
from datetime import datetime
from json import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config.json')
    
    args_ = parser.parse_args()
    config_file = args_.config_file
    if config_file.endswith('.json'):
        with open(config_file,'r') as f:
            args = load(f)
    elif config_file.endswith('.npy'):
        args = np.load(config_file,allow_pickle=True).item()
    else:
        raise ValueError("config_file must be .json or .npy")
    
    args = Args(**args)
    
    if args.dynamic_type == 'egnn' and args.time_embed_dim != 1:
        raise ValueError("When use 'egnn', time_embed_dim should be 1")
    if not args.use_context and args.context_node_nf > 0:
        raise ValueError("Context node feature size must be 0 if not using context")
    if not args.use_rct_cent and args.rct_cent_node_nf > 0:
        raise ValueError("reactive center node feature size must be 0 if not using reactive center")
    if args.link_rc and args.dataset_type == 1:
        raise ValueError("When use link_rc, dataset should be type 2")
    
    if args.dataset_type == 1:
        dataset = MultiDataset(root=args.dataset_path,name_regrex=args.name_regrex)
    elif args.dataset_type == 2:
        dataset = MultiDatasetV2(root=args.dataset_path,name_regrex=args.name_regrex,link_rc=args.link_rc,
                                 data_enhance=args.data_enhance)
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
        print("[INFO] Splitting dataset into train, valid and test")
        print(len(dataset))
        if not args.data_enhance:
            split_ids_map = get_idx_split(len(dataset), 
                                        int(args.train_ratio*len(dataset)), 
                                        int(args.valid_ratio*len(dataset)), 
                                        args.seed)
        else:
            dataset[0]
            split_ids_map = get_fnidx_split(dataset.data.fn_id,
                                            args.train_ratio,
                                            args.valid_ratio,
                                            args.seed)
        train_dataset = dataset[split_ids_map['train']]
        valid_dataset = dataset[split_ids_map['valid']]
        test_dataset = dataset[split_ids_map['test']]
    else:
        assert args.test_name_regrex != ''
        print("[INFO] Splitting dataset into train, valid. Test is specified")
        assert abs(args.train_ratio + args.valid_ratio - 1.0) < 1e-6
        if not args.data_enhance:
            split_ids_map = get_idx_split(len(dataset), 
                                        int(args.train_ratio*len(dataset)), 
                                        int(args.valid_ratio*len(dataset)), 
                                        args.seed)
        else:
            dataset[0]
            split_ids_map = get_fnidx_split(dataset.data.fn_id,
                                            args.train_ratio,
                                            args.valid_ratio,
                                            args.seed)
            
        train_dataset = dataset[split_ids_map['train']]
        valid_dataset = dataset[split_ids_map['valid']]
        if args.dataset_type == 1:
            test_dataset = MultiDataset(root=args.dataset_path, name_regrex=args.test_name_regrex)
        elif args.dataset_type == 2:
            test_dataset = MultiDatasetV2(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc,
                                          data_enhance=args.data_enhance)
        elif args.dataset_type == 3:
            test_dataset = MultiDataset1x(root=args.dataset_path, name_regrex=args.test_name_regrex, link_rc=args.link_rc,
                                          iptgraph_type=args.iptgraph_type)

        
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size//args.test_reduce_ratio, shuffle=False, num_workers=args.num_workers)

    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tot_train_steps = len(train_dataloader) * args.n_epochs
    args.tot_steps = tot_train_steps


    if args.condition_time:
        dynamics_in_node_nf = args.in_node_nf + args.time_embed_dim
    else:
        dynamics_in_node_nf = args.in_node_nf

    if args.dynamic_type == 'egnn':
        net_dynamics = EGNN_dynamics_DiffMM(
                            n_dims=args.n_dims,
                            in_node_nf=dynamics_in_node_nf,
                            context_node_nf=args.context_node_nf,
                            rct_cent_node_nf=args.rct_cent_node_nf,
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
                            add_fpfh=args.add_fpfh
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
                 lmax_list=[args.lmax],
                 mmax_list=[args.mmax],
                 add_node_feat=args.equif_add_node_feat,
        )
    else:
        raise NotImplementedError


    unitsgen = EnVariationalDiffusion(dynamics=net_dynamics,
                                n_dims=args.n_dims,
                                in_node_nf=args.in_node_nf,
                                context_node_nf=args.context_node_nf,
                                rct_cent_node_nf=args.rct_cent_node_nf,
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
                                enc_gnn_aggr=args.enc_gnn_aggr, 
                                enc_bond_feat_red=args.enc_bond_feat_red, 
                                enc_JK=args.enc_JK, 
                                enc_drop_ratio=args.enc_drop_ratio, 
                                enc_node_readout=args.enc_node_readout,
                                sample_fix=False,
                                seg_sample=False,
                                last_step_alldif=400,
                                last_refine_ratio=2,
                                use_rct_cent=args.use_rct_cent,
                                rct_cent_encoder_type=args.rct_cent_encoder_type,
                                rct_cent_readout=args.rct_cent_readout,
                                add_fpfh=args.add_fpfh,
                                focus_reaction_center=args.focus_reaction_center,
                                rc_loss_weight=args.rc_loss_weight,                 # loss weight for reaction center
                                rc_distance_weight=args.rc_distance_weight,         # loss weight for bond length
                                rc_angle_weight=args.rc_angle_weight,               # loss weight for bond angle
                                use_init_coords=args.use_init_coords,               # use init coords in training process
                                fix_step_from_init=args.fix_step_from_init,
                                )


    model = unitsgen.to(device)

    optim = get_optim(args.learning_rate, model, args.optimizer_type)

    if args.scheduler_type.lower() == "steplr":
        scheduler = StepLR(optim, step_size=args.steplr_step_size, gamma=args.steplr_gamma)

    elif args.scheduler_type.lower() == "onecycle":
        scheduler = OneCycleLR(optim,
                    max_lr=args.learning_rate,
                    total_steps=args.tot_steps,
                    div_factor=args.ocyclr_div_factor,
                    pct_start=args.ocyclr_pct_start,
                    anneal_strategy=args.ocyclr_anneal_strategy,
                    final_div_factor=args.ocyclr_final_div_factor)
        
    elif args.scheduler_type.lower() == "reduceonplateau":
        scheduler = ReduceLROnPlateau(optim, 
                                        mode='min',     
                                        factor=args.rpllr_factor,          # 0.95
                                        patience=args.rpllr_patience,      # 50
                                        verbose=False,
                                        min_lr=args.rpllr_min_lr)
    elif args.scheduler_type.lower() == "noamlr":
        scheduler = NoamLR(optim,model_size=args.hidden_nf,
                                    warmup_steps=args.warmup_step)
    else:
        raise NotImplementedError

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = deepcopy(model)
        ema = EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_loss_val = 1e8
    best_nll_test = 1e8
    best_loss_test = 1e8
    best_nll = 1e8
    best_loss = 1e8
    best_rcloss_val = 1e8
    best_rcloss_test = 1e8


    save_path = f"{args.save_path}/{args.tag}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(save_path, exist_ok=True)
    setup_logger(save_path)
    
    if args.dynamic_type == 'hiegnn':
        logging.info(f"[WARNING] dynamic_type is HiEGNN. lmax_attr, lmax_h, add_angle_info are ignored")
    logging.info(f"[INFO] Saving to {save_path}")
    if not args.specific_test:
        logging.info(f"[INFO] No specific test, which is random splitted from full dataset")
    else:
        logging.info(f"[INFO] Specific test, which is loaded from {args.test_name_regrex}")
    logging.info(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    np.save(f'{save_path}/args.npy', args.__dict__)
    logging.info(str(args.__dict__))


    for epoch in range(0, args.n_epochs):
        best_nll,best_loss = train_epoch(args=args, loader=train_dataloader, epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, optim=optim, scheduler=scheduler, gradnorm_queue=gradnorm_queue,device=device,best_nll=best_nll,
                    best_loss=best_loss,save_path=save_path,tot_epoch=args.n_epochs)
        best_nll_val, best_loss_val, best_rcloss_val = test(args, loader=valid_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                                               optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_val, best_loss=best_loss_val, 
                                               best_rcloss=best_rcloss_val,save_path=save_path, tot_epoch=args.n_epochs, mode='validation')
        best_nll_test, best_loss_test, best_rcloss_test = test(args, loader=test_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                                             optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_test, best_loss=best_loss_test, 
                                             best_rcloss=best_rcloss_test,save_path=save_path, tot_epoch=args.n_epochs, mode='test')
if __name__ == '__main__':
    main()