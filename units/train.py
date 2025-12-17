import torch,time,logging,glob,os
import numpy as np
from .utils import gradient_clipping
from .losses import compute_loss_and_nll



def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]

def test(args, loader, epoch, model_dp, model_ema, optim, scheduler, device, best_nll, best_loss, best_rcloss, save_path, tot_epoch, mode='validation'):
    start_time = time.time()
    model_dp.eval()
    nll_epoch = []
    loss_epoch = []
    rc_loss_epoch = []
    dist_loss_epoch = []
    angle_loss_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        data = data.to(device)
        nll, reg_term, mean_abs_z, loss_, rc_loss, dist_loss, angle_loss = compute_loss_and_nll(args, model_dp, data)
        loss = nll + args.ode_regularization * reg_term
        nll_epoch.append(nll.item())
        loss_epoch.append(loss_.item())
        rc_loss_epoch.append(rc_loss.item())
        dist_loss_epoch.append(dist_loss.item())
        angle_loss_epoch.append(angle_loss.item())

    current_lr = optim.param_groups[0]["lr"]
    end_time = time.time()
    logging.info(f"[INFO] {mode} Epoch {epoch} LOSS: {np.mean(loss_epoch):.4f}, RC LOSS: {np.mean(rc_loss_epoch):.4f}, DIST LOSS: {np.mean(dist_loss_epoch):.4f}, ANGLE LOSS: {np.mean(angle_loss_epoch):.4f}, current lr: {current_lr:.8f}, time duration: {end_time - start_time:.2f} s")
    if mode == 'validation' and np.mean(loss_epoch) < best_loss:
        logging.info(f"!!!!!!!!!!!!!![INFO] New best validation loss: {np.mean(loss_epoch):.4f}, saving model...!!!!!!!!!!!!!!")
        best_loss = np.mean(loss_epoch)
        state_dict = {'model':model_ema.state_dict(), 'optim':optim.state_dict(), 'scheduler':scheduler.state_dict(), 'epoch':epoch}

        torch.save(state_dict, f'{save_path}/best_full_model.pth')
        torch.save(state_dict, f'{save_path}/model_full_{epoch}.pth')
        model_files = sorted(glob.glob(f'{save_path}/model_full_*.pth'), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(model_files) > args.n_keep_ckpt:
            for model_file in model_files[:-args.n_keep_ckpt]:
                os.remove(model_file)
    if mode == 'validation' and np.mean(rc_loss_epoch) < best_rcloss:
        logging.info(f"~~~~~~~~~~~[INFO] New best validation reactive center loss: {np.mean(rc_loss_epoch):.4f}, saving model...~~~~~~~~~~~")
        best_rcloss = np.mean(rc_loss_epoch)
        state_dict = {'model':model_ema.state_dict(), 'optim':optim.state_dict(), 'scheduler':scheduler.state_dict(), 'epoch':epoch}
        torch.save(state_dict, f'{save_path}/best_rcfull_model.pth')
        torch.save(state_dict, f'{save_path}/model_rcfull_{epoch}.pth')
        model_files = sorted(glob.glob(f'{save_path}/model_rcfull_*.pth'), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(model_files) > args.n_keep_ckpt:
            for model_file in model_files[:-args.n_keep_ckpt]:
                os.remove(model_file)
        
        
    return best_nll, best_loss, best_rcloss

def train_epoch(args, loader, epoch, model, 
                model_dp, model_ema, ema, 
                optim, scheduler, gradnorm_queue, 
                device, best_nll, best_loss, save_path, tot_epoch):
    start_time = time.time()
    model_dp.train()
    model.train()
    nll_epoch = []
    loss_epoch = []
    rc_loss_epoch = []
    dist_loss_epoch = []
    angle_loss_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        data = data.to(device)
        optim.zero_grad()
        # transform batch through flow
        # nll, reg_term, mean_abs_z, loss
        nll, reg_term, mean_abs_z, loss_, rc_loss, dist_loss, angle_loss = compute_loss_and_nll(args, model_dp, data)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term
        loss.backward()

        if args.clip_grad:
            grad_norm = gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            logging.info(f"[INFO] Epoch: {epoch}, iter: {i}/{n_iterations}, Loss {loss.item():.4f}, rc loss {rc_loss.item():.4f}, dist loss {dist_loss.item():.4f}, angle loss {angle_loss.item():.4f}, GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        loss_epoch.append(loss_.item())
        rc_loss_epoch.append(rc_loss.item())
        dist_loss_epoch.append(dist_loss.item())
        angle_loss_epoch.append(angle_loss.item())

        # test part TODO
        if epoch % args.test_epochs == 0:
            # TODO
            pass

        if args.break_train_epoch:
            break
        if args.scheduler_type.lower() == 'onecycle' or args.scheduler_type.lower() == 'noamlr':
            scheduler.step()
    if args.scheduler_type.lower() == 'steplr':
        scheduler.step()
    elif args.scheduler_type.lower() == 'reduceonplateau':
        avg_nll = torch.mean(torch.tensor(nll_epoch)).item()
        scheduler.step(avg_nll)

    current_lr = optim.param_groups[0]["lr"]
    end_time = time.time()
    logging.info(f"[INFO] Train Epoch {epoch} LOSS: {np.mean(loss_epoch):.4f}, RC LOSS: {np.mean(rc_loss_epoch):.4f}, DIST LOSS: {np.mean(dist_loss_epoch):.4f}, ANGLE LOSS: {np.mean(angle_loss_epoch):.4f} current lr: {current_lr:.8f}, time duration: {end_time - start_time:.2f} s")
    #wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    return best_nll, best_loss
