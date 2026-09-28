"""Mathematical correctness checks; execute `python test_models.py`."""
import itertools
import torch
from torch.nn import functional as F
from models import Autoregressive


def main():
    torch.manual_seed(7)
    for hidden in [(), (8,), (8, 8)]:
        model = Autoregressive(dim=4, hidden=hidden)
        states = torch.tensor(list(itertools.product([0., 1.], repeat=4)))
        logp = -F.binary_cross_entropy_with_logits(model(states), states, reduction='none').sum(1)
        assert torch.allclose(logp.exp().sum(), torch.tensor(1.), atol=1e-6), 'Joint distribution is not normalized'
        for i in range(4):
            x = torch.rand(3, 4, requires_grad=True)
            grad = torch.autograd.grad(model(x)[:, i].sum(), x)[0]
            assert torch.equal(grad[:, i:], torch.zeros_like(grad[:, i:]))
        # Empirical sampling should agree with the enumerated joint distribution.
        samples = model.sample(torch.rand(20000, 4))
        indices = (samples * torch.tensor([8,4,2,1])).sum(1).long()
        frequency = torch.bincount(indices, minlength=16).float()/len(samples)
        assert (frequency-logp.exp()).abs().max() < .015
        print('PASS: normalization, causal gradient and ancestral sampler', hidden)


if __name__ == '__main__':
    main()
