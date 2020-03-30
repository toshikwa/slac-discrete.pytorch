from collections import deque
import numpy as np
import torch
from torch.distributions.kl import kl_divergence


def create_feature_actions(features_seq, actions_seq, num_actions):
    N = features_seq.size(0)

    if actions_seq.shape[-1] == 1:
        actions_seq = to_onehot(actions_seq, num_actions)

    # sequence of features
    f = features_seq[:, :-1].view(N, -1)
    n_f = features_seq[:, 1:].view(N, -1)
    # sequence of actions
    a = actions_seq[:, :-1].view(N, -1)
    n_a = actions_seq[:, 1:].view(N, -1)

    # feature_actions
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)

    return fa, n_fa


def calc_kl_divergence(p_list, q_list):
    assert len(p_list) == len(q_list)

    kld = 0.0
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        _kld = kl_divergence(p_list[i], q_list[i])
        # Average along batches, sum along sequences and elements.
        kld += _kld.mean(dim=0).sum()

    return kld


def update_params(optim, loss, networks, retain_graph=False,
                  grad_cliping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        for net in networks:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def to_onehot(y, num_categories):
    assert y.shape[-1] == 1
    scatter_dim = len(y.shape[:-1])
    y_tensor = y.type(torch.LongTensor).to(y.device)
    zeros = torch.zeros(
        *y.shape[:-1], num_categories, dtype=torch.float32).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)
