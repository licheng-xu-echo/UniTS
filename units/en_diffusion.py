from .utils import assert_mean_zero_with_mask,sample_center_gravity_zero_gaussian_with_mask,\
split_and_padding,remove_mean_with_mask,random_transform,gather_selected_nodes_and_compute_mean,compute_angles
import logging
from torch.nn import functional as F
import torch,math
import numpy as np
from .encoder.graph_encoder import GraphEncoder
from tqdm import tqdm


def clamp_to_pi(tensor):
    return torch.clamp(tensor, min=0.0, max=torch.pi)  # clamp to [0, pi]

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(-1)

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )

def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )

def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d

class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)

        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)
        alphas2 = np.clip(alphas2, a_min=0.0, a_max=0.99999)
        
        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([0.0]))   ## 原始版本是-5.0，影响初始噪声大小，-5.0太小了
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics,
            n_dims=3,
            in_node_nf=0,
            context_node_nf=128,
            rct_cent_node_nf=32,
            timesteps: int = 1000, 
            init_timesteps=100,
            parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1.),
            norm_biases=(None, 0.0),enc_num_layers=4,
            time_sample_method="power",time_sample_power=4.0,min_sample_power=2.0,
            max_sample_power=6.0,degree_as_continuous=True,dynamic_context=False,dynamic_context_temperature=1.0,
            loss_calc='all',tot_x_mae=True,sample_fix=False,seg_sample=False,last_step_alldif=400,last_refine_ratio=2,
            mol_encoder_type="gcn",use_context=False,enc_gnn_aggr='add', enc_bond_feat_red='mean', enc_JK='last', enc_drop_ratio=0, enc_node_readout='sum',use_rct_cent=False,rct_cent_encoder_type='gcn',rct_cent_readout='mean',
            add_fpfh=False,
            focus_reaction_center=False,
            rc_loss_weight=2.0,           # loss weight for reaction center
            rc_distance_weight=3.0,       # loss weight for bond length
            rc_angle_weight=2.0,          # loss weight for bond angle
            use_init_coords=False,
            fix_step_from_init=500,
            ):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        #assert time_sample_method in ["uniform", "power","dynamic_power","loss_sensing"]

        self.loss_type = loss_type
        self.time_sample_method = time_sample_method
        self.time_sample_power = time_sample_power
        self.min_sample_power = min_sample_power
        self.max_sample_power = max_sample_power
        self.degree_as_continuous = degree_as_continuous
        self.dynamic_context = dynamic_context
        self.dynamic_context_temperature = dynamic_context_temperature
        self.loss_calc = loss_calc.lower()
        self.tot_x_mae = tot_x_mae
        self.sample_fix = sample_fix
        self.seg_sample = seg_sample
        self.last_step_alldif = last_step_alldif
        self.last_refine_ratio = last_refine_ratio
        self.use_context = use_context
        self.use_rct_cent = use_rct_cent
        self.rct_cent_encoder_type = rct_cent_encoder_type
        self.rct_cent_readout = rct_cent_readout
        self.add_fpfh = add_fpfh
        self.focus_reaction_center = focus_reaction_center
        self.rc_loss_weight = rc_loss_weight
        self.rc_distance_weight = rc_distance_weight
        self.rc_angle_weight = rc_angle_weight
        self.use_init_coords = use_init_coords
        self.fix_step_from_init = fix_step_from_init
        assert self.rct_cent_readout == 'mean', 'Only `mean` readout is supported for rct_cent_encoder_type.'

        assert self.loss_calc in ['all','xyz','rot_mat']
        logging.info(f"[INFO] Loss calculation type: {self.loss_calc}")
        if self.focus_reaction_center:
            logging.info(f"[INFO] Reaction center focus enabled with weights: "
                        f"loss={self.rc_loss_weight}, distance={self.rc_distance_weight}, angle={self.rc_angle_weight}")
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims

        # self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.init_timesteps = init_timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()

        if self.use_context:
            self.mol_encoder = GraphEncoder(gnum_layer=enc_num_layers, emb_dim=context_node_nf,
                                            gnn_aggr=enc_gnn_aggr, bond_feat_red=enc_bond_feat_red,
                                            gnn_type=mol_encoder_type, JK=enc_JK, drop_ratio=enc_drop_ratio, node_readout=enc_node_readout)
        if self.use_rct_cent:
            self.rct_cent_encoder = GraphEncoder(gnum_layer=enc_num_layers, emb_dim=rct_cent_node_nf,
                                            gnn_aggr=enc_gnn_aggr, bond_feat_red=enc_bond_feat_red,
                                            gnn_type=rct_cent_encoder_type, JK=enc_JK, drop_ratio=enc_drop_ratio, node_readout=rct_cent_readout)

        self.context_dim = context_node_nf
        self.context_node_nf = context_node_nf
        self.rct_cent_context_dim = rct_cent_node_nf
        self.rct_cent_node_nf = rct_cent_node_nf
    
    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = self.norm_values[1]

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context, batch):
        net_out = self.dynamics._forward(x, t, node_mask, edge_mask, context, batch)  # x no hidden information
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, node_mask):
        #print("node_mask",node_mask)
        x = x / self.norm_values[0]      ## coords + axis
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        return x, delta_log_px

    def unnormalize(self, x, node_mask):
        x = x * self.norm_values[0]

        return x

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x = z
        #h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        #assert h_int.size(2) == self.include_charges

        # Unnormalize
        x, = self.unnormalize(x, node_mask)
        output = x
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, x, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((x.size(0), 1), device=x.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, x)

        # Compute means.
        mu_T = alpha_T * x
        mu_T_x = mu_T[:, :, :self.n_dims]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze(-1).squeeze(-1)  # Remove inflate, only keep batch dimension for x-part.


        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred
    
    def compute_error(self, net_out, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        eps_t_xyz = eps_t[:, :, :3]
        eps_xyz = eps[:, :, :3]
        xyz_error = sum_except_batch((eps_xyz - eps_t_xyz) ** 2) 
        return xyz_error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False, batch=None):
        """Samples x ~ p(x|z0)."""
        # z0 xyz + h
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context, batch=batch)    # input xyz + h, output xyz

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0[:, :, :3], gamma_0)
        x = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        #h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x = self.unnormalize(x, node_mask)
        return x

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
            self, eps, net_out):

        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.

        log_p_x_given_z_without_constants = self.compute_error(net_x, eps_x)   # tot_error, xyz_error, pa1_error, pa2_error, pa3_error
        
        log_p_x_given_z_without_constants = -0.5 * log_p_x_given_z_without_constants

        log_p_xh_given_z = log_p_x_given_z_without_constants

        return log_p_xh_given_z

    def time_step_sample(self, x, lowest_t, power=4.0, reverse=False):
        # control the probability of sampling small t_int, larger the value, the higher the probability of sampling small t_int
        #power = 4.0  
        t_candidates = torch.arange(lowest_t, self.T + 1, device=x.device).float()  
        if not reverse:
            t_weights = (self.T + 1 - t_candidates) ** power 
        else:
            t_weights = (t_candidates - lowest_t + 1) ** 4
        t_weights /= t_weights.sum()  

        t_indices = torch.multinomial(t_weights, x.size(0), replacement=True)  
        t_int = t_candidates[t_indices].unsqueeze(1)
        return t_int


    def compute_rc_distance_loss(self, x_real, x_pred, rc_bond_indices):
        """
        x_real, x_pred: [batch_size, max_node_num, 3]
        rc_bond_indices: [batch_size, 2]
        """
        if not self.focus_reaction_center:
            return torch.tensor(0.0, device=x_real.device)
        #batch_size, max_index_num, _ = rc_bond_indices.shape

        valid_mask = (rc_bond_indices[..., 0] >= 0) & (rc_bond_indices[..., 1] >= 0)  # [batch_size, max_index_num]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=x_real.device)
        
        valid_batch_indices, valid_index_indices = torch.where(valid_mask)
        valid_bond_indices = rc_bond_indices[valid_batch_indices, valid_index_indices]  # [n_valid, 2]
        

        batch_indices = valid_batch_indices.unsqueeze(1).expand(-1, 2)
        atom_indices = valid_bond_indices
        
        real_positions = x_real[batch_indices, atom_indices]
        pred_positions = x_pred[batch_indices, atom_indices]
        
        real_distances = torch.norm(real_positions[:, 0] - real_positions[:, 1], dim=1)
        pred_distances = torch.norm(pred_positions[:, 0] - pred_positions[:, 1], dim=1)
        
        dist_loss = sum_except_batch((real_distances - pred_distances) ** 2)
        return dist_loss * self.rc_distance_weight
    
    def compute_rc_angle_loss(self, x_real, x_pred, rc_angle_indices):
        """
        x_real, x_pred: [batch_size, max_node_num, 3]
        rc_angle_indices: [batch_size, 3]
        """
        if not self.focus_reaction_center:
            return torch.tensor(0.0, device=x_real.device)

        # batch_size, max_index_num, _ = rc_angle_indices.shape

        valid_mask = (rc_angle_indices[..., 0] >= 0) & (rc_angle_indices[..., 1] >= 0) & (rc_angle_indices[..., 2] >= 0)  # [batch_size, max_index_num]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=x_real.device)
        
        valid_batch_indices, valid_index_indices = torch.where(valid_mask)
        valid_angle_indices = rc_angle_indices[valid_batch_indices, valid_index_indices]  # [n_valid, 3]
        
        batch_indices = valid_batch_indices.unsqueeze(1).expand(-1, 3)
        atom_indices = valid_angle_indices
        
        real_positions = x_real[batch_indices, atom_indices]
        pred_positions = x_pred[batch_indices, atom_indices]
        

        real_ba = real_positions[:, 0] - real_positions[:, 1]  # A -> B
        real_bc = real_positions[:, 2] - real_positions[:, 1]  # C -> B
        
        pred_ba = pred_positions[:, 0] - pred_positions[:, 1]
        pred_bc = pred_positions[:, 2] - pred_positions[:, 1]
        
        real_angles = compute_angles(real_ba, real_bc)
        pred_angles = compute_angles(pred_ba, pred_bc)
        angle_loss = sum_except_batch((real_angles - pred_angles) ** 2)
        
        return angle_loss * self.rc_angle_weight 
    
    def compute_rc_geometry_loss(self, x, x_pred, rc_bond_indices, rc_angle_indices):
        """
        x, x_pred: [batch_size, n_nodes, 3]

        """
        if not self.focus_reaction_center:
            return torch.tensor(0.0, device=x.device)
        
        batch_size, _, _ = rc_angle_indices.shape
        dist_loss_ = self.compute_rc_distance_loss(x, x_pred, rc_bond_indices).sum()
        dist_loss = torch.zeros((batch_size,), device=x.device) + dist_loss_/batch_size
        
        
        angle_loss_ = self.compute_rc_angle_loss(x, x_pred, rc_angle_indices).sum()
        angle_loss = torch.zeros((batch_size,), device=x.device) + angle_loss_/batch_size

        return dist_loss + angle_loss, dist_loss, angle_loss
    
    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always, init_x, batch):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        # here, x and h are truth values and are normalized.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0
        
        # Sample a timestep t.
        if self.time_sample_method == "uniform":
            t_int = torch.randint(
                lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
            if self.use_init_coords:
                is_init = torch.randint(0, 10, size=(x.size(0), 1), device=x.device) == 9  # 10%
                if self.fix_step_from_init != 0:
                    t_int = torch.where(is_init,self.fix_step_from_init,t_int)  # from 500 step
        elif self.time_sample_method == "power":
            t_int = self.time_step_sample(x, lowest_t, self.time_sample_power)
        elif self.time_sample_method == "power_reverse":
            t_int = self.time_step_sample(x, lowest_t, self.time_sample_power, reverse=True)
        elif self.time_sample_method == "dynamic_power":
            t_int = self.time_step_sample(x, lowest_t, self.min_sample_power + (self.max_sample_power - self.min_sample_power) * self.epoch_ratio)
        elif self.time_sample_method == "loss_sensing":
            raise NotImplementedError("loss_sensing not implemented.")
        else:
            raise ValueError("Unknown time_sample_method: {}".format(self.time_sample_method))
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)        ## gamma from -11.xx to 11.xxx
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)                    # retain level
        sigma_t = self.sigma(gamma_t, x)                    # noise level

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)  ## xyz, pa1, pa2, pa3
        # Concatenate x, h.
        #xh = torch.cat([x, h], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        #z_t = alpha_t * xh + sigma_t * eps  
        z_t_x = alpha_t * x + sigma_t * eps
        if self.use_init_coords:
            z_t_x[torch.where(is_init)[0]] = init_x[torch.where(is_init)[0]]
            eps = (z_t_x - alpha_t * x) / sigma_t
            
            
            
        z_t = torch.cat([z_t_x, h], dim=2)

        assert_mean_zero_with_mask(z_t[:, :, :3], node_mask)

        # Neural net prediction.
        # here context is fixed and is not related to node distance
        if self.dynamic_context:
            raise NotImplementedError("dynamic_context not implemented.")
            #context = aggr_context(z_t, context, node_mask, temperature=self.dynamic_context_temperature)

        net_out = self.phi(z_t, t, node_mask, edge_mask, context, batch) 

        
        # Compute the error.

        error = self.compute_error(net_out, eps)

        if self.focus_reaction_center:
            pos_pred = (z_t[:, :, :3] - sigma_t * net_out[:, :, :3]) / alpha_t
            pos_truth = x[:, :, :3].clone()
            rc_loss, dist_loss, angle_loss = self.compute_rc_geometry_loss(pos_truth, pos_pred, self.rc_bond_indices, self.rc_angle_indices)
            rc_loss = rc_loss  * self.rc_loss_weight

        # tot_error, degree_error, x_error
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(x, node_mask)      # h is not the target
        #print(f"rc_loss.shape: {rc_loss.shape}, kl_prior.shape: {kl_prior.shape}")
        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            #z_0 = alpha_0 * xh + sigma_0 * eps_0
            z_0_x = alpha_0 * x + sigma_0 * eps_0
            z_0 = torch.cat([z_0_x, h], dim=2)


            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context, batch)
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                 eps_0, net_out)            
            if self.focus_reaction_center:
                pos_pred_0 = (z_0[:, :, :3] - sigma_0 * net_out[:, :, :3]) / alpha_0
                pos_truth_0 = x[:, :, :3].clone()
                rc_loss_0, dist_loss_0, angle_loss_0 = self.compute_rc_geometry_loss(pos_truth_0, pos_pred_0, 
                                                                                     self.rc_bond_indices, self.rc_angle_indices)
                rc_loss_0 = rc_loss_0 * self.rc_loss_weight * 0.5 
                loss_term_0 += rc_loss_0

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
            if self.focus_reaction_center:

                loss = loss + rc_loss
        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                eps, net_out)
            if self.focus_reaction_center:
                pos_pred_0 = (z_t[:, :, :3] - sigma_t * net_out[:, :, :3]) / alpha_t
                pos_truth_0 = x[:, :, :3].clone()
                rc_loss_0, dist_loss_0, angle_loss_0 = self.compute_rc_geometry_loss(pos_truth_0, pos_pred_0, 
                                                                                     self.rc_bond_indices, self.rc_angle_indices)
                rc_loss_0 = rc_loss_0 * self.rc_loss_weight * 0.5
                #print(f"loss_term_0.shape: {loss_term_0.shape}, rc_loss_0.shape: {rc_loss_0.shape}")
                loss_term_0 += rc_loss_0


            t_is_not_zero = 1 - t_is_zero
            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
            if self.focus_reaction_center:
                loss = loss + rc_loss
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'
        loss_dict = {'t': t_int.squeeze(),'loss_t': loss.squeeze(),'error': error.squeeze()}
        if self.focus_reaction_center:
            loss_dict['rc_loss'] = rc_loss.squeeze()
            loss_dict['dist_loss'] = dist_loss.squeeze()
            loss_dict['angle_loss'] = angle_loss.squeeze()
        return loss, loss_dict
    def forward(self, data):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        # prepare x, atom mask, edge_mask and h

        xyz,atom_mask = split_and_padding(data.mol_coords,data.batch)
        node_feat,_ = split_and_padding(data.x,data.batch,data.x.shape[1])
        atom_mask = atom_mask.float()
        if self.use_init_coords:
            init_xyz,_ = split_and_padding(data.init_mol_coords,data.batch)
        else:
            init_xyz = None
        x = xyz.clone()
        h = node_feat.clone()
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask.to(edge_mask.device)
        atom_mask = atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        x[:,:,:3] = remove_mean_with_mask(x[:,:,:3], atom_mask)
        x, delta_log_px = self.normalize(x, atom_mask)    ## x is coord
        if self.use_context:
            context = self.mol_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)
            context,_ = split_and_padding(context,data.batch,self.context_dim)
        else:
            context = None
        if self.use_rct_cent:
            assert context is not None
            rct_cent_context = self.rct_cent_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)
            assert rct_cent_context.shape[-1] == self.rct_cent_context_dim
            rct_cent_context,_ = split_and_padding(rct_cent_context,data.batch,self.rct_cent_context_dim)
            rct_cent_context = gather_selected_nodes_and_compute_mean(rct_cent_context, data.reactive_atoms)
            rct_cent_context = rct_cent_context.unsqueeze(1).expand(-1, context.shape[1], -1)
            context = torch.cat([context, rct_cent_context], dim=2)
        if self.add_fpfh:
            fpfh,_ = split_and_padding(data.init_mol_fpfh,data.batch,33)
            h = torch.cat([h, fpfh], dim=2)

        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        self.rc_bond_indices = data.rc_bond_indices
        self.rc_angle_indices = data.rc_angle_indices
        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss, loss_dict = self.compute_loss(x, h, atom_mask, edge_mask, context, t0_always=False, init_x=init_xyz,batch=data.batch)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, atom_mask, edge_mask, context, t0_always=True, init_x=init_xyz,batch=data.batch)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh, loss_dict

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False, batch=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        # zt: xyz + h
        # output xyz + h
        # h is not denoised
        zt_x = zt[:, :, :3]
        zt_h = zt[:, :, 3:]
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt_x)

        sigma_s = self.sigma(gamma_s, target_tensor=zt_x)
        sigma_t = self.sigma(gamma_t, target_tensor=zt_x)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context, batch)  # input xyz+h return noise for xyz

        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, :3], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, :3], node_mask)
        mu = zt_x / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t


        # Sample zs given the paramters derived from zt.
        zs_x = self.sample_normal(mu, sigma, node_mask, fix_noise)
       
        zs_x = remove_mean_with_mask(zs_x, node_mask)
        zs = torch.cat([zs_x, zt_h], dim=-1)
        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, 3), device=node_mask.device,
            node_mask=node_mask)
        return z

    @torch.no_grad()
    def prepare_sample_data(self,data):
        device = data.x.device
        
        mol_atoms,_ = split_and_padding(data.mol_atoms.unsqueeze(1),data.batch,size=1)
        node_feat,_ = split_and_padding(data.x,data.batch,data.x.shape[1])
        h = node_feat.clone()
        if self.add_fpfh:
            fpfh,_ = split_and_padding(data.init_mol_fpfh,data.batch,33)
            h = torch.cat([h, fpfh], dim=2)
        node_bincount = torch.bincount(data.batch)
        n_nodes = torch.max(node_bincount)
        max_n_nodes = torch.max(node_bincount)
        batch_size = torch.max(data.batch) + 1
        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0:node_bincount[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
        node_mask = node_mask.unsqueeze(2).to(device)
        return batch_size,n_nodes,node_mask,edge_mask,mol_atoms,h

    def prepare_init_data(self,data):
        mol_atoms,_ = split_and_padding(data.mol_atoms.unsqueeze(1),data.batch,size=1)
        mol_coords_pad,atom_mask = split_and_padding(data.mol_coords.cpu(),data.batch.cpu(),3)
        mol_atoms_pad,_ = split_and_padding(data.mol_atoms.cpu().unsqueeze(1),data.batch.cpu(),1)
        frame_batch_pad,_ = split_and_padding(data.frame_batch.cpu().unsqueeze(1)+1,data.batch.cpu(),1)
        frame_batch_pad = frame_batch_pad.squeeze(2).long()
        frame_batch_pad = frame_batch_pad - frame_batch_pad[:,0].unsqueeze(1)
        new_coords_batch = []
        for idx in range(len(mol_coords_pad)):
            coords, atoms, sub_frame_idx = mol_coords_pad[idx][atom_mask[idx]].numpy(), mol_atoms_pad[idx][atom_mask[idx]].reshape(-1).long().numpy(), frame_batch_pad[idx][atom_mask[idx]].numpy()
            new_coords = np.zeros_like(coords)
            for i in range(max(sub_frame_idx)+1):
                sub_coords = coords[sub_frame_idx==i]
                rand_sub_coords = random_transform(sub_coords,max_translation=5,max_rotation_deg=360)
                rand_sub_coords_pert = rand_sub_coords + (np.random.rand(*rand_sub_coords.shape) - 0.5) * 0.2
                new_coords[sub_frame_idx==i] = rand_sub_coords_pert
            new_coords_batch.append(new_coords)
        new_coords_batch_ = torch.from_numpy(np.concatenate(new_coords_batch))
        new_coords_batch,_ = split_and_padding(new_coords_batch_,data.batch.cpu(),3)
        new_coords_batch = remove_mean_with_mask(new_coords_batch,atom_mask.float().unsqueeze(-1))
        batch_size = torch.max(data.batch) + 1
        node_bincount = torch.bincount(data.batch)
        max_n_nodes = torch.max(node_bincount)
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(data.x.device)

        return mol_atoms,new_coords_batch.to(data.x.device),new_coords_batch_.to(data.x.device),atom_mask.float().unsqueeze(-1).to(data.x.device),edge_mask.to(data.x.device),batch_size

    
    @torch.no_grad()
    def sample_traj_from_init(self, data):
        x_traj = []
        batch_size,n_nodes,node_mask,edge_mask,mol_atoms,h = self.prepare_sample_data(data)
        n_samples = batch_size
        z_x,_ = split_and_padding(data.init_mol_coords,data.batch,size=3)

        assert_mean_zero_with_mask(z_x[:, :, :3], node_mask)
        z = torch.cat([z_x,h],dim=-1)

        if self.use_context:
            # context 在随时间降噪过程中不能变
            context = self.mol_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)        
            context,_ = split_and_padding(context,data.batch,self.context_dim)
        else:
            context = None
        
        if self.use_rct_cent:
            assert context is not None
            rct_cent_context = self.rct_cent_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)
            rct_cent_context,_ = split_and_padding(rct_cent_context,data.batch,self.rct_cent_context_dim)
            rct_cent_context = gather_selected_nodes_and_compute_mean(rct_cent_context, data.reactive_atoms)
            rct_cent_context = rct_cent_context.unsqueeze(1).expand(-1, context.shape[1], -1)
            context = torch.cat([context, rct_cent_context], dim=2)

        for s in tqdm(reversed(range(0, self.init_timesteps))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=True, batch=data.batch)
            x_traj.append(z[:,:,:3].cpu().detach().numpy())
        x = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=True,batch=data.batch) ## [pos,pa1_norm,pa2_norm,pa3_norm]
            
        assert_mean_zero_with_mask(x[:, :, :3], node_mask)
        max_cog = torch.sum(x[:, :, :3], dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                    f'the positions down.')
            x_pos = x[:, :, :3]
            x_pos = remove_mean_with_mask(x_pos, node_mask)
            x[:, :, :3] = x_pos
        x_traj.append(x[:,:,:3].cpu().detach().numpy())
        return x_traj,mol_atoms.cpu(),node_mask.cpu()
    

    def noise_to_step_t(self, x_s, time_t, time_s, node_mask):
        """
        from time_s to time_t
        """
        assert torch.all(time_t >= time_s), "Target timestep t must be >= current timestep s"
        
        # Convert timesteps to continuous values in [0, 1]
        t = time_t.float() / self.T
        s = time_s.float() / self.T
        
        # Compute gamma values
        gamma_t = self.inflate_batch_array(self.gamma(t),x_s)
        gamma_s = self.inflate_batch_array(self.gamma(s),x_s)
        
        # Compute alpha and sigma values
        alpha_t = self.alpha(gamma_t, x_s)
        #alpha_s = self.alpha(gamma_s, x_s)
        
        sigma_t = self.sigma(gamma_t, x_s)
        #sigma_s = self.sigma(gamma_s, x_s)
        
        # Generate noise
        eps = self.sample_combined_position_feature_noise(
            n_samples=x_s.size(0), n_nodes=x_s.size(1), node_mask=node_mask
        )
        
        # Compute x_t using the transition formula
        x_t = alpha_t * x_s + sigma_t * eps
        
        return x_t

    @torch.no_grad()
    def sample_traj(self, data, fix_noise=False, resample=False, resample_steps=10, start_step=40, jump_len=2):
        x_traj = []
        batch_size, n_nodes, node_mask, edge_mask, mol_atoms, h = self.prepare_sample_data(data)
        n_samples = batch_size
        
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z_x = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)          # xyz
        else:
            z_x = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)  # xyz

        assert_mean_zero_with_mask(z_x[:, :, :3], node_mask)
        z = torch.cat([z_x,h],dim=-1)
        if self.use_context:
            context = self.mol_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)        
            context,_ = split_and_padding(context,data.batch,self.context_dim)
        else:
            context = None
        
        if self.use_rct_cent:
            assert context is not None
            rct_cent_context = self.rct_cent_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)
            rct_cent_context,_ = split_and_padding(rct_cent_context,data.batch,self.rct_cent_context_dim)
            rct_cent_context = gather_selected_nodes_and_compute_mean(rct_cent_context, data.reactive_atoms)
            rct_cent_context = rct_cent_context.unsqueeze(1).expand(-1, context.shape[1], -1)
            context = torch.cat([context, rct_cent_context], dim=2)
            
        for step in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=step, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise, batch=data.batch)
            x_traj.append(z[:,:,:3].cpu().detach().numpy())
            
            # resample
            if resample and step < start_step and step % jump_len == 0:
                logging.info(f"Resampling at step {step}...")
                for _ in range(resample_steps):
                
                    # diffuse current state to step + jump_len
                    time_s = torch.full((n_samples,), fill_value=step, device=z.device)
                    time_t = torch.full((n_samples,), fill_value=step + jump_len, device=z.device)
                    
                    # add noise to position and feature
                    z_x = self.noise_to_step_t(z[:,:,:3], time_t, time_s, node_mask)       # from time_s to time_t
                    z_h = z[:,:,3:]
                    # z_x = remove_mean_with_mask(z_x, node_mask)
                    z = torch.cat([z_x, z_h], dim=-1)
                    
                    for j in range(step, step + jump_len)[::-1]: # 40, 42 //  41 -> 40
                        s_array_ = torch.full((n_samples,), fill_value=j, device=z.device)
                        t_array_ = s_array_ + 1
                        s_array_ = s_array_ / self.T
                        t_array_ = t_array_ / self.T
                        z = self.sample_p_zs_given_zt(s_array_, t_array_, z, node_mask, edge_mask, context, fix_noise=fix_noise, batch=data.batch)
                        x_traj.append(z[:,:,:3].cpu().detach().numpy())
                        

        x = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise, batch=data.batch)
        assert_mean_zero_with_mask(x[:, :, :3], node_mask)
        max_cog = torch.sum(x[:, :, :3], dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting the positions down.')
            x_pos = x[:, :, :3]
            x_pos = remove_mean_with_mask(x_pos, node_mask)
            x[:, :, :3] = x_pos
        
        x_traj.append(x[:,:,:3].cpu().detach().numpy())
        return x_traj, mol_atoms.cpu(), node_mask.cpu()
    @torch.no_grad()
    def sample_traj_bk(self, data, fix_noise=False):
        x_traj = []
        batch_size, n_nodes, node_mask, edge_mask, mol_atoms, h = self.prepare_sample_data(data)
        n_samples = batch_size
        # if is not used
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z_x = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)          # xyz
        else:
            
            z_x = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)  # xyz

        assert_mean_zero_with_mask(z_x[:, :, :3], node_mask)
        z = torch.cat([z_x,h],dim=-1)


        if self.use_context:
            context = self.mol_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)        
            context,_ = split_and_padding(context,data.batch,self.context_dim)
        else:
            context = None
        
        if self.use_rct_cent:
            assert context is not None
            rct_cent_context = self.rct_cent_encoder(data.x[:,:-1],data.edge_index,data.edge_attr)
            rct_cent_context,_ = split_and_padding(rct_cent_context,data.batch,self.rct_cent_context_dim)
            rct_cent_context = gather_selected_nodes_and_compute_mean(rct_cent_context, data.reactive_atoms)
            rct_cent_context = rct_cent_context.unsqueeze(1).expand(-1, context.shape[1], -1)
            context = torch.cat([context, rct_cent_context], dim=2)

        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise, batch=data.batch)
            x_traj.append(z[:,:,:3].cpu().detach().numpy())
        x = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise, batch=data.batch) ## [pos,pa1_norm,pa2_norm,pa3_norm]
            
        assert_mean_zero_with_mask(x[:, :, :3], node_mask)
        max_cog = torch.sum(x[:, :, :3], dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                    f'the positions down.')
            x_pos = x[:, :, :3]
            x_pos = remove_mean_with_mask(x_pos, node_mask)
            x[:, :, :3] = x_pos
        x_traj.append(x[:,:,:3].cpu().detach().numpy())
        return x_traj,mol_atoms.cpu(),node_mask.cpu()

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info