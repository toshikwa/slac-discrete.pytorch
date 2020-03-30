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
            latent_dim, num_actions, hidden_units)
        self.Q2 = QNetwork(
            latent_dim, num_actions, hidden_units)

        self.num_actions = num_actions
        self.latent_dim = latent_dim

    def forward(self, latents):
        assert latents.shape[1:] == (self.latent_dim, )
        return self.Q1(latents), self.Q2(latents)
