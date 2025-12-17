import torch
def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def compute_loss_and_nll(args, generative_model, data):
    #bs, n_nodes, n_dims = x.size()


    if args.probabilistic_model == 'diffusion':

        #assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll,loss_dict = generative_model(data)

        #N = node_mask.squeeze(2).sum(1).long()

        #log_pN = nodes_dist.log_prob(N)

        #assert nll.size() == log_pN.size()
        #nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)
        loss = loss_dict["error"].mean()
        if args.focus_reaction_center:
            rc_loss = loss_dict["rc_loss"].mean()
            dist_loss = loss_dict["dist_loss"].mean()
            angle_loss = loss_dict["angle_loss"].mean()
        else:
            rc_loss = torch.tensor([0.]).to(nll.device)
            dist_loss = torch.tensor([0.]).to(nll.device)
            angle_loss = torch.tensor([0.]).to(nll.device)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z, loss, \
           rc_loss, dist_loss, angle_loss