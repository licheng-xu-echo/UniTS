import logging,argparse,os
from datetime import datetime
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from units.eval import calc_rmsd
from units.utils import setup_logger
pt = Chem.GetPeriodicTable()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default="./sample_traj")
    parser.add_argument("--result_tag", type=str, default="traj-sample-repeat-10-transition1x_egnn-2025-12-26-01-24-57")
    parser.add_argument("--rmsd_result_path", type=str, default="./rmsd_calc")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    result_path = args.result_path
    rmsd_result_path = args.rmsd_result_path
    result_tag = args.result_tag
    timeout = args.timeout

    save_path = f"{rmsd_result_path}/rmsd-calc-{result_tag}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(save_path, exist_ok=True)
    setup_logger(save_path)
    logging.info(str(args))
    truth_pred_results_dict = np.load(f"{result_path}/{result_tag}/truth_pred_traj.npy",allow_pickle=True).item()


    logging.info(result_tag)
    top1_rmsd = []
    top5_rmsd = []
    top10_rmsd = []
    for batch_idx in truth_pred_results_dict.keys():
        logging.info(f"batch {batch_idx} / {len(truth_pred_results_dict.keys())} ...")
        truth_pred_results = truth_pred_results_dict[batch_idx]
        truth_xyz = truth_pred_results['truth_xyz']
        pred_xyz_ensemble = truth_pred_results['pred_xyz']
        node_mask = truth_pred_results['node_mask']
        mol_atoms = truth_pred_results['mol_atoms']
        
        rmsd_batch_repeat = []
        for repeat_id in range(len(pred_xyz_ensemble)):
            pred_xyz = pred_xyz_ensemble[repeat_id]
            rmsd_batch = []
            for batch_id in range(len(pred_xyz)):
                _node_mask = node_mask[batch_id]
                _pred_xyz = pred_xyz[batch_id][_node_mask.squeeze().bool()]
                _truth_xyz = truth_xyz[batch_id]
                
                _mol_atoms = mol_atoms[batch_id][_node_mask.squeeze().bool()].reshape(-1)
                _mol_atoms = [pt.GetElementSymbol(int(_mol_atom)) for _mol_atom in _mol_atoms]
                rmsd = calc_rmsd(_mol_atoms, _truth_xyz, _pred_xyz, timeout=timeout)
                logging.info(f"batch {batch_idx} / {len(truth_pred_results_dict.keys())} repeat {repeat_id+1} / {len(pred_xyz_ensemble)}, step {batch_id+1} / {len(pred_xyz)} atom num: {len(_mol_atoms)} RMSD: {rmsd:.4f}")
                rmsd_batch.append(rmsd)
            rmsd_batch_repeat.append(rmsd_batch)
        rmsd_batch_repeat = np.array(rmsd_batch_repeat)

        top1_rmsd.append(rmsd_batch_repeat[0])
        top5_rmsd.append(rmsd_batch_repeat[:5].min(axis=0))
        top10_rmsd.append(rmsd_batch_repeat[:10].min(axis=0))
        logging.info(f"batch {batch_idx} top1 rmsd: {np.mean(np.concatenate(top1_rmsd)):.4f} , top5 rmsd: {np.mean(np.concatenate(top5_rmsd)):.4f} , top10 rmsd: {np.mean(np.concatenate(top10_rmsd)):.4f} ({np.concatenate(top10_rmsd).shape})")
    logging.info(f"top1 rmsd: {np.mean(np.concatenate(top1_rmsd)):.4f} , top5 rmsd: {np.mean(np.concatenate(top5_rmsd)):.4f} , top10 rmsd: {np.mean(np.concatenate(top10_rmsd)):.4f} ({np.concatenate(top10_rmsd).shape})")
if __name__ == '__main__':
    main()