import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def create_linear_network(input_dim, output_dim, hidden_units=(256, 256),
                          hidden_activation=nn.ReLU(), output_activation=None,
                          initializer=initialize_weights_xavier):
    model = []
    units = input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(hidden_activation)
        units = next_units

    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(output_activation)

    return nn.Sequential(*model).apply(initializer)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Gaussian(nn.Module):
    """ Diagonal gaussian distribution parametrized by DNN. """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256),
                 std=None, leaky_slope=0.2):
        super(Gaussian, self).__init__()

        self.net = create_linear_network(
            input_dim, 2*output_dim if std is None else output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(leaky_slope),
            initializer=initialize_weights_xavier)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        assert x.shape[1:] == (self.input_dim, )

        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return Normal(loc=mean, scale=std)


class ConstantGaussian(nn.Module):
    """ Diagonal gaussian distribution with zero means. """

    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()

        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return Normal(loc=mean, scale=std)


class Encoder(nn.Module):

    def __init__(self, input_dim=3, output_dim=256, leaky_slope=0.2):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (output_dim, 1, 1)
            nn.Conv2d(256, output_dim, kernel_size=4),
            nn.LeakyReLU(leaky_slope)
        ).apply(initialize_weights_he)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        assert x.ndim == 5 and x.shape[2:] == (self.input_dim, 84, 84)
        batch_size, num_sequences = x.shape[:2]

        x = self.net(
            x.view(batch_size * num_sequences, self.input_dim, 84, 84)
            ).view(batch_size, num_sequences, self.output_dim)

        return x


class Decoder(nn.Module):

    def __init__(self, input_dim=288, output_dim=3, std=1.0, leaky_slope=0.2):
        super(Decoder, self).__init__()

        self.net = nn.Sequential(
            # (input_dim, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (output_dim, 64, 64)
            nn.ConvTranspose2d(32, output_dim, 5, 2, 2, output_padding=1),
            nn.LeakyReLU(leaky_slope)
        ).apply(initialize_weights_he)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x,  dim=-1)

        assert x.ndim == 3 and x.shape[2:] == (self.input_dim, )
        batch_size, num_sequences = x.shape[:2]

        x = self.net(
            x.view(batch_size * num_sequences, self.input_dim, 1, 1)
            ).view(batch_size, num_sequences, self.output_dim, 84, 84)

        return Normal(loc=x, scale=torch.ones_like(x) * self.std)
