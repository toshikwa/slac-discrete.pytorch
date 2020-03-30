import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from slac_discrete.network import create_linear_network,\
    initialize_weights_xavier


class CateoricalPolicy(nn.Module):

    def __init__(self, input_dim, num_actions, hidden_units=(256, 256)):
        super(CateoricalPolicy, self).__init__()

        # NOTE: Convolutional layers are shared with the latent model.
        self.net = create_linear_network(
            input_dim, num_actions, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initialize_weights_xavier)

        self.input_dim = input_dim
        self.num_actions = num_actions

    def forward(self, feature_actions, deterministic=False):
        assert feature_actions.shape[1:] == (self.embedding_dim, )

        if deterministic:
            return torch.argmax(self.net(feature_actions), dim=1, keepdim=True)

        else:
            log_action_probs = F.log_softmax(self(feature_actions), dim=1)
            action_probs = log_action_probs.exp()
            actions = Categorical(action_probs).sample().view(-1, 1)

            return actions, action_probs, log_action_probs
