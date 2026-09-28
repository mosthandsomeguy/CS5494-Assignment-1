"""Small, explicit models for MNIST generation. No pretrained generative weights."""
import torch
from torch import nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, inputs, outputs, mask):
        super().__init__(inputs, outputs)
        self.register_buffer('mask', mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class Autoregressive(nn.Module):
    """FVSBN when hidden is empty; fixed-order MADE otherwise."""
    def __init__(self, dim=784, hidden=()):
        super().__init__()
        self.dim = dim
        degrees = [torch.arange(1, dim + 1)]
        # Hidden degrees cover 1..D-1 deterministically and approximately evenly.
        for width in hidden:
            degrees.append(torch.floor(torch.arange(width) * (dim - 1) / width).long() + 1)
        degrees.append(torch.arange(1, dim + 1))
        layers = []
        for i in range(len(degrees) - 1):
            prev, nxt = degrees[i], degrees[i + 1]
            output = i == len(degrees) - 2
            mask = (nxt[:, None] > prev[None, :]) if output else (nxt[:, None] >= prev[None, :])
            layers.append(MaskedLinear(len(prev), len(nxt), mask))
            if not output:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def sample(self, uniforms):
        """Ancestral Bernoulli sampling; uniforms make progress grids comparable."""
        x = torch.zeros_like(uniforms)
        if len(self.net) == 1:
            layer = self.net[0]
            for i in range(self.dim):
                logits = F.linear(x[:, :i], layer.weight[i:i+1, :i], layer.bias[i:i+1])[:, 0]
                x[:, i] = (uniforms[:, i] < logits.sigmoid()).float()
        else:
            for i in range(self.dim):
                x[:, i] = (uniforms[:, i] < self(x)[:, i].sigmoid()).float()
        return x


class Generator(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 128*7*7), nn.BatchNorm1d(128*7*7), nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(.2), nn.Flatten(), nn.Linear(64*7*7, 1))

    def forward(self, x):
        return self.net(x).flatten()


class DigitClassifier(nn.Module):
    """Auxiliary evaluator only; labels never train the generative models."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def parameter_counts(model):
    stored = sum(p.numel() for p in model.parameters())
    active = stored
    for module in model.modules():
        if isinstance(module, MaskedLinear):
            active -= module.weight.numel() - int(module.mask.sum())
    return {'stored': stored, 'active': active}
