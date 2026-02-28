import torch,os,logging,argparse,time
import numpy as np
from datetime import datetime
from torch_geometric.data import DataLoader
from units.data_pipeline import build_test_dataset_for_sampling
from units.model_pipeline import build_diffusion_model_from_args
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
    test_dataset = build_test_dataset_for_sampling(args)

    logging.info(f"test set size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset, batch_size=eval_args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_diffusion_model_from_args(args).to(device)
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
