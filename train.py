"""Reproducible experiments. Run `python train.py --help` for stage selection."""
import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from models import Autoregressive, Generator, Discriminator, DigitClassifier, parameter_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def data():
    """One fixed split of official train; official test is never used for selection."""
    tr = MNIST(ROOT / 'data', train=True, download=True)
    te = MNIST(ROOT / 'data', train=False, download=True)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(2026))
    np.savez(ROOT / 'data' / 'split_indices.npz', train=order[:55000].numpy(), validation=order[55000:].numpy())
    x = tr.data.float().div(255)
    t = te.data.float().div(255)
    return (x[order[:55000]].to(DEVICE), x[order[55000:]].to(DEVICE), t.to(DEVICE),
            tr.targets[order[:55000]].to(DEVICE), tr.targets[order[55000:]].to(DEVICE), te.targets.to(DEVICE))


@torch.no_grad()
def nll(model, x):
    model.eval()
    values = []
    for batch in x.split(512):
        values.append(F.binary_cross_entropy_with_logits(model(batch), batch, reduction='none').sum(1).cpu())
    return torch.cat(values)


def causal_check(model):
    """Changing present/future pixels must not change the current conditional."""
    model.eval()
    x = torch.rand(3, 784, device=DEVICE, requires_grad=True)
    for i in [0, 1, 27, 28, 392, 783]:
        g = torch.autograd.grad(model(x)[:, i].sum(), x)[0]
        assert g[:, i:].abs().max().item() == 0, f'Future leakage at pixel {i}'
    return True


def benchmark(fn):
    fn()
    times = []
    for _ in range(3):
        sync(); start = time.perf_counter(); fn(); sync()
        times.append(time.perf_counter() - start)
    return {'batch_size': 100, 'seconds_median': float(np.median(times)), 'seconds_runs': times}


def train_ar(name, hidden, lr, count, epochs, arrays, seed=42):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train, val, test = [(a >= .5).float().flatten(1) for a in arrays[:3]]
    train = train[:count]
    model = Autoregressive(hidden=hidden).to(DEVICE)
    # Marginal-logit initialization gives sensible initial background probabilities.
    with torch.no_grad():
        model.net[-1].bias.copy_(torch.logit(train.mean(0).clamp(.001, .999)))
    causal_check(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    uniforms = torch.rand(100, 784, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(1234))
    history, stages = [], []
    best, best_epoch, best_state = float('inf'), 0, None
    stage_epochs = sorted(set([0, 1, epochs//3, 2*epochs//3, epochs]))
    stages.append(model.sample(uniforms[:4]).cpu())
    sync(); start = time.perf_counter()
    update_seconds = 0
    for epoch in range(1, epochs + 1):
        model.train()
        sync(); ts = time.perf_counter()
        total = 0.
        for ids in torch.randperm(len(train), device=DEVICE).split(256):
            x = train[ids]
            loss = F.binary_cross_entropy_with_logits(model(x), x, reduction='none').sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += loss.item() * len(x)
        sync(); update_seconds += time.perf_counter() - ts
        vl = nll(model, val).mean().item()
        history.append({'epoch': epoch, 'train_nll': total/len(train), 'val_nll': vl})
        if vl < best:
            best, best_epoch = vl, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch in stage_epochs:
            stages.append(model.sample(uniforms[:4]).cpu())
        write_json(folder / 'history.json', history)
        print(name, epoch, 'train', round(total/len(train), 3), 'val', round(vl, 3), flush=True)
    sync(); training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    losses = nll(model, test)
    np.save(folder / 'test_nll.npy', losses.numpy())
    sample = model.sample(uniforms).cpu().reshape(-1, 1, 28, 28)
    save_image(sample, folder / 'samples.png', nrow=10, padding=2)
    save_image(torch.cat(stages).reshape(-1, 1, 28, 28), folder / 'progress.png', nrow=4, padding=3)
    torch.save({'model': best_state, 'hidden': hidden, 'seed': seed}, folder / 'checkpoint.pt')
    result = {'name': name, 'hidden': hidden, 'learning_rate': lr, 'train_count': count, 'epochs': epochs,
        'seed': seed, 'best_epoch': best_epoch, 'validation_nll': best, 'test_nll': losses.mean().item(),
        'test_nll_se': losses.std().item()/len(losses)**.5, 'bits_per_pixel': losses.mean().item()/(784*np.log(2)),
        'training_seconds': training_seconds, 'update_seconds': update_seconds,
        'inference': benchmark(lambda: model.sample(uniforms)), 'parameters': parameter_counts(model),
        'progress_epochs': stage_epochs, 'causal_check_passed': causal_check(model)}
    write_json(folder / 'metrics.json', result)
    return result


def train_gan(arrays, epochs, name='gan', smooth=.9, seed=42):
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train = arrays[0].unsqueeze(1)*2-1
    g, d = Generator().to(DEVICE), Discriminator().to(DEVICE)
    go = torch.optim.Adam(g.parameters(), lr=.0002, betas=(.5, .999))
    do = torch.optim.Adam(d.parameters(), lr=.0002, betas=(.5, .999))
    fixed = torch.randn(100, 64, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(1234))
    history, stages = [], []
    stage_epochs = sorted(set([0, 1, epochs//3, 2*epochs//3, epochs]))
    g.eval()
    with torch.no_grad(): stages.append((g(fixed[:4]).cpu()+1)/2)
    sync(); start = time.perf_counter(); update_seconds = 0
    for epoch in range(1, epochs+1):
        g.train(); d.train(); dl, gl = 0., 0.
        sync(); ts = time.perf_counter()
        for ids in torch.randperm(len(train), device=DEVICE).split(256):
            real = train[ids]; b = len(real)
            for p in d.parameters(): p.requires_grad_(True)
            fake = g(torch.randn(b, 64, device=DEVICE))
            loss_d = (F.binary_cross_entropy_with_logits(d(real), torch.full((b,), smooth, device=DEVICE)) +
                      F.binary_cross_entropy_with_logits(d(fake.detach()), torch.zeros(b, device=DEVICE)))
            do.zero_grad(set_to_none=True); loss_d.backward(); do.step()
            for p in d.parameters(): p.requires_grad_(False)
            loss_g = F.binary_cross_entropy_with_logits(d(fake), torch.ones(b, device=DEVICE))
            go.zero_grad(set_to_none=True); loss_g.backward(); go.step()
            dl += loss_d.item()*b; gl += loss_g.item()*b
        sync(); update_seconds += time.perf_counter()-ts
        history.append({'epoch': epoch, 'discriminator_loss': dl/len(train), 'generator_loss': gl/len(train)})
        g.eval()
        with torch.no_grad():
            if epoch in stage_epochs: stages.append((g(fixed[:4]).cpu()+1)/2)
            save_image((g(fixed).cpu()+1)/2, folder / 'samples.png', nrow=10)
        write_json(folder / 'history.json', history)
        print(name, epoch, 'D', round(dl/len(train),4), 'G', round(gl/len(train),4), flush=True)
    sync(); elapsed = time.perf_counter()-start
    save_image(torch.cat(stages), folder / 'progress.png', nrow=4, padding=3)
    torch.save({'generator': g.state_dict(), 'discriminator': d.state_dict()}, folder / 'checkpoint.pt')
    with torch.no_grad(): timing = benchmark(lambda: g(fixed))
    result = {'name': name, 'epochs': epochs, 'seed': seed, 'train_count': len(train), 'real_label': smooth,
        'training_seconds': elapsed, 'update_seconds': update_seconds, 'inference': timing,
        'generator_parameters': parameter_counts(g), 'discriminator_parameters': parameter_counts(d),
        'progress_epochs': stage_epochs, 'checkpoint_selection': 'final epoch (predefined)'}
    write_json(folder / 'metrics.json', result)
    return result


def evaluate_gan(arrays):
    """Classifier diagnostics are proxies, not proof that samples are realistic."""
    seed_all(314)
    tr, va, te, ty, vy, ey = arrays
    c = DigitClassifier().to(DEVICE)
    opt = torch.optim.Adam(c.parameters(), lr=.001)
    best, state = 0., None
    for epoch in range(5):
        c.train()
        for ids in torch.randperm(len(tr), device=DEVICE).split(256):
            loss = F.cross_entropy(c(tr[ids,None]), ty[ids])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        c.eval()
        with torch.no_grad():
            pred = torch.cat([c(x[:,None]).argmax(1) for x in va.split(512)])
            acc = (pred==vy).float().mean().item()
        if acc > best: best, state = acc, copy.deepcopy(c.state_dict())
        print('classifier',epoch+1,acc,flush=True)
    c.load_state_dict(state); c.eval()
    with torch.no_grad():
        pred = torch.cat([c(x[:,None]).argmax(1) for x in te.split(512)])
        test_acc = (pred==ey).float().mean().item()
    torch.save(state, OUT / 'classifier.pt')
    for name in ['gan','gan_no_smoothing','gan_long']:
        folder = OUT/name
        if not (folder/'checkpoint.pt').exists(): continue
        g = Generator().to(DEVICE)
        g.load_state_dict(torch.load(folder/'checkpoint.pt', weights_only=True, map_location=DEVICE)['generator']); g.eval()
        seed_all(999)
        probabilities, fake_batches = [], []
        with torch.no_grad():
            for _ in range(20):
                fake=(g(torch.randn(100,64,device=DEVICE))+1)/2
                probabilities.append(c(fake).softmax(1).cpu()); fake_batches.append(fake.cpu())
        probs=torch.cat(probabilities); fake=torch.cat(fake_batches)
        counts=torch.bincount(probs.argmax(1),minlength=10)
        p=counts.float()/counts.sum()
        entropy=-(p[p>0]*p[p>0].log()).sum().item()
        # Exact nearest-neighbour distances for the first 100 generated samples.
        # Distances are pixel RMSE on [0,1], against the entire training subset.
        q=fake[:100].flatten(1).to(DEVICE); min_d=torch.full((100,),float('inf'),device=DEVICE)
        nearest=torch.zeros(100,dtype=torch.long,device=DEVICE)
        with torch.no_grad():
            for start in range(0,len(tr),1000):
                dist=torch.cdist(q,tr[start:start+1000].flatten(1))/28
                v,ix=dist.min(1); better=v<min_d
                nearest[better]=ix[better]+start; min_d=torch.minimum(min_d,v)
        pairs=torch.stack([fake[:10],tr[nearest[:10]].cpu()[:,None]],dim=1).flatten(0,1)
        save_image(pairs,folder/'nearest_neighbors.png',nrow=2,padding=3)
        result={'classifier_validation_accuracy':best,'classifier_test_accuracy':test_acc,'sample_count':2000,
            'class_counts':counts.tolist(),'class_entropy_nats':entropy,'effective_classes':float(np.exp(entropy)),
            'mean_max_class_probability':probs.max(1).values.mean().item(),
            'fraction_confidence_ge_0_9':(probs.max(1).values>=.9).float().mean().item(),
            'nearest_train_rmse_mean':min_d.mean().item(),'nearest_train_rmse_min':min_d.min().item()}
        write_json(folder/'diagnostics.json',result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','ar','gan','extended','evaluate','smoke'],default='all')
    parser.add_argument('--epochs-ar',type=int,default=30)
    parser.add_argument('--epochs-gan',type=int,default=30)
    args=parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    write_json(OUT/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'device':str(DEVICE),'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
        'split_seed':2026,'binarization':'fixed threshold >= 0.5','image_shape':[28,28]})
    arrays=data()
    if args.stage=='smoke':
        for hidden in [(),(256,),(512,512)]: causal_check(Autoregressive(hidden=hidden).to(DEVICE))
        g,d=Generator().to(DEVICE),Discriminator().to(DEVICE)
        loss=d(g(torch.randn(4,64,device=DEVICE))).mean(); loss.backward()
        print('All smoke checks passed.'); return
    if args.stage in ['all','ar']:
        configs=[('fvsbn',(),.001,55000),('made_1layer',(512,),.001,55000),
            ('made_2layer',(512,512),.001,55000),('made_lr_low',(512,512),.0003,55000),
            ('made_small_data',(512,512),.001,10000)]
        for name,hidden,lr,count in configs:
            if not (OUT/name/'metrics.json').exists(): train_ar(name,hidden,lr,count,args.epochs_ar,arrays)
    if args.stage in ['all','gan']:
        for name,smooth in [('gan',.9),('gan_no_smoothing',1.)]:
            if not (OUT/name/'metrics.json').exists(): train_gan(arrays,args.epochs_gan,name,smooth)
    if args.stage in ['all','extended']:
        if not (OUT/'made_long'/'metrics.json').exists():
            train_ar('made_long',(512,512),.001,55000,120,arrays)
        if not (OUT/'gan_long'/'metrics.json').exists():
            train_gan(arrays,100,'gan_long',.9)
    if args.stage in ['all','extended','evaluate']: evaluate_gan(arrays)


if __name__=='__main__': main()
