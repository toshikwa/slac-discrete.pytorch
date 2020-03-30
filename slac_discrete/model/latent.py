import numpy as np
import torch
import torch.nn as nn

from slac_discrete.network import Gaussian, ConstantGaussian, Encoder, Decoder


class LatentNetwork(nn.Module):

    def __init__(self, observation_shape, action_shape, feature_dim=256,
                 latent1_dim=32, latent2_dim=256, hidden_units=[256, 256],
                 leaky_slope=0.2):
        super(LatentNetwork, self).__init__()
        # NOTE: We encode x as the feature vector to share convolutional
        # part of the network with the policy.

        # p(z1(0)) = N(0, I)
        self.latent1_init_prior = ConstantGaussian(latent1_dim)
        # p(z2(0) | z1(0))
        self.latent2_init_prior = Gaussian(
            latent1_dim, latent2_dim, hidden_units, leaky_slope=leaky_slope)
        # p(z1(t+1) | z2(t), a(t))
        self.latent1_prior = Gaussian(
            latent2_dim + action_shape[0], latent1_dim, hidden_units,
            leaky_slope=leaky_slope)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_prior = Gaussian(
            latent1_dim + latent2_dim + action_shape[0], latent2_dim,
            hidden_units, leaky_slope=leaky_slope)

        # q(z1(0) | feat(0))
        self.latent1_init_posterior = Gaussian(
            feature_dim, latent1_dim, hidden_units, leaky_slope=leaky_slope)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.latent2_init_posterior = self.latent2_init_prior
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.latent1_posterior = Gaussian(
            feature_dim + latent2_dim + action_shape[0], latent1_dim,
            hidden_units, leaky_slope=leaky_slope)
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        self.latent2_posterior = self.latent2_prior

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward_predictor = Gaussian(
            2 * latent1_dim + 2 * latent2_dim + action_shape[0],
            1, hidden_units, leaky_slope=leaky_slope)

        # feat(t) = x(t) : This encoding is performed deterministically.
        self.encoder = Encoder(
            observation_shape[0], feature_dim, leaky_slope=leaky_slope)
        # p(x(t) | z1(t), z2(t))
        self.decoder = Decoder(
            latent1_dim + latent2_dim, observation_shape[0],
            std=np.sqrt(0.1), leaky_slope=leaky_slope)

    def sample_prior(self, actions_seq, init_features=None):
        ''' Sample from prior dynamics (with conditionning on the initial frames).
        Args:
            actions_seq   : (N, S, *action_shape) tensor of action sequences.
            init_features : (N, *) tensor of initial frames or None.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # Condition on initial frames.
                if init_features is not None:
                    # q(z1(0) | feat(0))
                    latent1_dist = self.latent1_init_posterior(init_features)
                    latent1_sample = latent1_dist.rsample()
                    # q(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

                # Not conditionning.
                else:
                    # p(z1(0)) = N(0, I)
                    latent1_dist = self.latent1_init_prior(actions_seq[t])
                    latent1_sample = latent1_dist.rsample()
                    # p(z2(0) | z1(0))
                    latent2_dist = self.latent2_init_prior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()

            else:
                # p(z1(t) | z2(t-1), a(t-1))
                latent1_dist = self.latent1_prior(
                    [latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # p(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_prior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def sample_posterior(self, features_seq, actions_seq):
        ''' Sample from posterior dynamics.
        Args:
            features_seq : (N, S+1, 256) tensor of feature sequenses.
            actions_seq  : (N, S, *action_space) tensor of action sequenses.
        Returns:
            latent1_samples : (N, S+1, L1) tensor of sampled latent vectors.
            latent2_samples : (N, S+1, L2) tensor of sampled latent vectors.
            latent1_dists   : (S+1) length list of (N, L1) distributions.
            latent2_dists   : (S+1) length list of (N, L2) distributions.
        '''
        num_sequences = actions_seq.size(1)
        features_seq = torch.transpose(features_seq, 0, 1)
        actions_seq = torch.transpose(actions_seq, 0, 1)

        latent1_samples = []
        latent2_samples = []
        latent1_dists = []
        latent2_dists = []

        for t in range(num_sequences + 1):
            if t == 0:
                # q(z1(0) | feat(0))
                latent1_dist = self.latent1_init_posterior(features_seq[t])
                latent1_sample = latent1_dist.rsample()
                # q(z2(0) | z1(0))
                latent2_dist = self.latent2_init_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()
            else:
                # q(z1(t) | feat(t), z2(t-1), a(t-1))
                latent1_dist = self.latent1_posterior(
                    [features_seq[t], latent2_samples[t-1], actions_seq[t-1]])
                latent1_sample = latent1_dist.rsample()
                # q(z2(t) | z1(t), z2(t-1), a(t-1))
                latent2_dist = self.latent2_posterior(
                    [latent1_sample, latent2_samples[t-1], actions_seq[t-1]])
                latent2_sample = latent2_dist.rsample()

            latent1_samples.append(latent1_sample)
            latent2_samples.append(latent2_sample)
            latent1_dists.append(latent1_dist)
            latent2_dists.append(latent2_dist)

        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples),\
            (latent1_dists, latent2_dists)

    def visualize(self, states_seq, actions_seq):
        features_seq = self.encoder(states_seq)

        post_samples, _ = self.sample_posterior(features_seq, actions_seq)
        post_images = self.decoder(post_samples).loc

        cond_samples, _ = self.sample_prior(actions_seq, features_seq[:, 0])
        cond_images = self.decoder(cond_samples).loc

        pri_samples, _ = self.sample_prior(actions_seq)
        pri_images = self.decoder(pri_samples).loc

        return states_seq.cpu(), post_images.cpu(), cond_images.cpu(),\
            pri_images.cpu()
