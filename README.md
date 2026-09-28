# CS5494 Assignment 1

**Huang ChuanBin · 60434764**

From-scratch FVSBN, MADE and convolutional GAN experiments on MNIST.

- [Report](60434764.pdf)
- [Detailed execution instructions](README.txt)
- [Open notebook in Colab](https://colab.research.google.com/github/mosthandsomeguy/CS5494-Assignment-1/blob/main/run_colab.ipynb)
- [Saved experiment results](results/)
- [MNIST data and split indices](data/)

## Run

Use Python 3.12 and a GPU runtime for training:

```bash
git clone https://github.com/mosthandsomeguy/CS5494-Assignment-1.git
cd CS5494-Assignment-1
pip install -r requirements.txt
python test_models.py
python train.py --stage smoke
python verify_results.py
python build_report.py
```

The saved checkpoints and results are included. To train all nine runs from scratch,
use a fresh working copy with its `results` folder moved to a backup location, then:

```bash
python train.py --stage all
python verify_results.py
python build_report.py
```

Completed configurations are skipped. Changing epoch arguments does not invalidate
existing results automatically. See `README.txt` for stage selection and CUDA installation.

## Protocol

The official 60,000-image MNIST training set is divided into 55,000 training and
5,000 validation examples (split seed 2026); the official 10,000-image test set is held out.
Autoregressive models use fixed threshold binarization at 0.5. GANs use the original
grayscale images scaled to [-1, 1]. All images remain 28 by 28.

Five 30-epoch AR runs compare model depth, learning rate and training-set size.
Two 30-epoch GAN runs compare real-label smoothing. Additional runs train MADE
for 120 epochs and GAN for 100 epochs. Checkpoint selection and limitations are
documented in the report. All displayed grids retain the first 100 samples without curation.

The repository contains the original MNIST files obtained through torchvision,
checkpoints, per-epoch histories, measured timings, diagnostic results, and the scripts
that generate the report. No pretrained generative weights are used.

Do not modify the shared code/data after submitting the assignment.
