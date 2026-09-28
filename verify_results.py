"""Verify the saved evidence, independently of the report builder."""
import json
from pathlib import Path
import numpy as np
import torch
from models import Autoregressive, Generator, parameter_counts
from train import causal_check, DEVICE

ROOT=Path(__file__).resolve().parent


def main():
    split=np.load(ROOT/'data'/'split_indices.npz')
    assert len(split['train'])==55000 and len(split['validation'])==5000
    assert len(np.intersect1d(split['train'],split['validation']))==0
    assert len(np.unique(np.concatenate([split['train'],split['validation']])))==60000
    checks=[]
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir():continue
        metric=json.loads((folder/'metrics.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        assert len(history)==metric['epochs']
        assert metric['training_seconds']>0 and metric['inference']['seconds_median']>0
        checkpoint=torch.load(folder/'checkpoint.pt',map_location=DEVICE,weights_only=True)
        if 'hidden' in metric:
            losses=np.load(folder/'test_nll.npy')
            assert losses.shape==(10000,) and np.isfinite(losses).all()
            assert abs(float(losses.mean())-metric['test_nll'])<1e-4
            assert abs(metric['test_nll']/(784*np.log(2))-metric['bits_per_pixel'])<1e-6
            best=min(history,key=lambda row:row['val_nll'])
            assert best['epoch']==metric['best_epoch']
            model=Autoregressive(hidden=metric['hidden']).to(DEVICE)
            model.load_state_dict(checkpoint['model']);causal_check(model)
            assert parameter_counts(model)==metric['parameters']
        else:
            model=Generator().to(DEVICE);model.load_state_dict(checkpoint['generator']);model.eval()
            with torch.no_grad():sample=model(torch.randn(10,64,device=DEVICE))
            assert sample.shape==(10,1,28,28) and sample.min()>=-1 and sample.max()<=1
            diag=json.loads((folder/'diagnostics.json').read_text())
            assert sum(diag['class_counts'])==diag['sample_count']==2000
        for f in ['samples.png','progress.png','checkpoint.pt','history.json','metrics.json']:
            assert (folder/f).is_file()
        checks.append(folder.name)
    assert len(checks)==9
    result={'passed':True,'experiments_checked':checks,'checks':['disjoint split','complete epoch logs',
        'finite metrics','validation checkpoint selection','saved NLL consistency','checkpoint loading',
        'causal gradients','parameter counts','generator shape/range','diagnostic counts','artifact presence']}
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
