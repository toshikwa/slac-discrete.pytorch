from torch import nn

from slac_discrete.network import create_linear_network,\
    initialize_weights_xavier


class QNetwork(nn.Module):

    def __init__(self, num_actions, latent_dim=288, hidden_units=(256, 256)):
        super(QNetwork, self).__init__()

        self.net = create_linear_network(
            latent_dim, num_actions, hidden_units=hidden_units,
            hidden_activation=nn.ReLU(), initializer=initialize_weights_xavier)

    def forward(self, embeddings):
        return self.net(embeddings)


class TwinnedQNetwork(nn.Module):

    def __init__(self, num_actions, latent_dim=288, hidden_units=(256, 256)):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(
            num_actions, latent_dim, hidden_units)
        self.Q2 = QNetwork(
            num_actions, latent_dim, hidden_units)

        self.num_actions = num_actions
        self.latent_dim = latent_dim

    def forward(self, latents, actions=None):
        assert latents.shape[1:] == (self.latent_dim, )

        q1 = self.Q1(latents)
        q2 = self.Q2(latents)

        if actions is not None:
            q1 = q1.gather(1, actions)
            q2 = q2.gather(1, actions)

        return q1, q2
