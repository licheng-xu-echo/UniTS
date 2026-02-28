import torch
import torch.distributed as dist
from units.data_pipeline import build_dataloaders, build_train_valid_test_datasets
from units.model_pipeline import build_diffusion_model_from_args, build_dynamics_from_args
from units.runtime import build_scheduler_from_args, prepare_parallel_models
from units.utils import get_optim,Queue,setup_logger,Args
from units.train import train_epoch,test
import numpy as np
import argparse
import os,logging
from datetime import datetime
from json import load

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return True, rank, world_size, gpu
    return False, 0, 1, 0

# Initialize distributed training
distributed, rank, world_size, gpu = setup_distributed()
if distributed:
    device = torch.device(f'cuda:{gpu}')
else:
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

    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets(args, rank=rank)
    dataloaders = build_dataloaders(
        args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    train_dataloader = dataloaders.train
    valid_dataloader = dataloaders.valid
    test_dataloader = dataloaders.test

    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tot_train_steps = len(train_dataloader) * args.n_epochs
    args.tot_steps = tot_train_steps

    net_dynamics = build_dynamics_from_args(args)
    model = build_diffusion_model_from_args(args, dynamics=net_dynamics).to(device)

    optim = get_optim(args.learning_rate, model, args.optimizer_type)
    scheduler = build_scheduler_from_args(args, optim)

    gradnorm_queue = Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.

    if distributed and rank == 0:
        print(f'Training using {world_size} GPUs with DistributedDataParallel')
    elif args.dp and torch.cuda.device_count() > 1 and rank == 0:
        print(f'Training using {torch.cuda.device_count()} GPUs with DataParallel')

    parallel_models = prepare_parallel_models(model, args, distributed=distributed, gpu=gpu)
    model_dp = parallel_models.model_dp
    model_ema = parallel_models.model_ema
    ema = parallel_models.ema

    best_nll_val = 1e8
    best_loss_val = 1e8
    best_nll_test = 1e8
    best_loss_test = 1e8
    best_nll = 1e8
    best_loss = 1e8
    best_rcloss_val = 1e8
    best_rcloss_test = 1e8


    save_path = f"{args.save_path}/{args.tag}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if rank == 0:
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
    else:
        # Setup minimal logging for non-rank 0 processes
        logging.basicConfig(level=logging.WARNING)

    # Synchronize all processes before starting training
    if distributed:
        dist.barrier()

    for epoch in range(0, args.n_epochs):
        # Set epoch for distributed sampler
        if distributed and dataloaders.train_sampler is not None:
            dataloaders.train_sampler.set_epoch(epoch)
        
        best_nll,best_loss = train_epoch(args=args, loader=train_dataloader, epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, optim=optim, scheduler=scheduler, gradnorm_queue=gradnorm_queue,device=device,best_nll=best_nll,
                    best_loss=best_loss,save_path=save_path,tot_epoch=args.n_epochs)
        
        # Only rank 0 performs validation and testing
        if rank == 0:
            best_nll_val, best_loss_val, best_rcloss_val = test(args, loader=valid_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                                                   optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_val, best_loss=best_loss_val, 
                                                   best_rcloss=best_rcloss_val,save_path=save_path, tot_epoch=args.n_epochs, mode='validation')
            best_nll_test, best_loss_test, best_rcloss_test = test(args, loader=test_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                                                 optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_test, best_loss=best_loss_test, 
                                                 best_rcloss=best_rcloss_test,save_path=save_path, tot_epoch=args.n_epochs, mode='test')
        else:
            # Non-rank 0 processes still need to run test to maintain synchronization
            # but they don't save or log
            _ = test(args, loader=valid_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                     optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_val, best_loss=best_loss_val, 
                     best_rcloss=best_rcloss_val,save_path=save_path, tot_epoch=args.n_epochs, mode='validation')
            _ = test(args, loader=test_dataloader, epoch=epoch, model_dp=model_dp, model_ema=model_ema, 
                     optim=optim, scheduler=scheduler, device=device, best_nll=best_nll_test, best_loss=best_loss_test, 
                     best_rcloss=best_rcloss_test,save_path=save_path, tot_epoch=args.n_epochs, mode='test')
    
    # Cleanup distributed training
    if distributed:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
