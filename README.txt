CS5494 Assignment 1 - Huang ChuanBin (60434764)
================================================

1. Contents
models.py: explicitly masked FVSBN/MADE, convolutional GAN, auxiliary classifier.
train.py: downloads MNIST, fixes data split, trains, times and evaluates experiments.
results/: real experiment logs, checkpoints, uncurated samples and diagnostics.
data/: original MNIST files and reproducible training/validation split indices.
build_report.py: builds figures and the English PDF report from saved measurements.
60434764.pdf: report (generated after training).

2. Run on Google Colab (recommended: GPU runtime)
Select Runtime > Change runtime type > GPU. Upload/extract this entire folder,
including results/ and data/ if reproducing the submitted outputs offline.
Change working directory to the extracted folder, then run:
    !pip install -r requirements.txt
    !python train.py --stage smoke
    !python train.py --stage all
    !python build_report.py
Colab's preinstalled compatible torch/torchvision can alternatively be retained;
the exact local versions used in this submission are recorded in environment.json.

3. Run locally
Use Python 3.12 and a virtual environment. Install requirements.txt.
For CUDA 12.1 wheels on Windows/Linux:
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    python train.py --stage smoke
    python train.py --stage all
    python build_report.py
CPU fallback is automatic, but full training will take substantially longer.
No API keys, paid service or pretrained generative model is required.

4. Reproduction and stage selection
Completed experiments (with metrics.json) are skipped to avoid accidental overwrites.
To rerun an experiment, first move its results subfolder to a backup location.
For an entirely fresh run, use a fresh copy without results/.
    python train.py --stage ar
    python train.py --stage gan
    python train.py --stage evaluate
Defaults: 30 epochs for each core AR configuration and each core GAN configuration.
The all stage additionally runs a 120-epoch MADE and a 100-epoch GAN budget experiment.
Run only these follow-ups with: python train.py --stage extended
--epochs-ar and --epochs-gan change the budget; changing them does not invalidate
an existing cached experiment automatically. Do not mix caches from different runs.
The smoke stage tests masks and a complete GAN backward pass; it is not training.

5. Experimental protocol
Original MNIST: 60,000 official training / 10,000 official test images.
Fixed split seed 2026: 55,000 training / 5,000 validation from official train.
AR: 28x28, raster scan, fixed threshold pixel/255 >= 0.5.
GAN: original 28x28 grayscale, pixel/127.5 - 1, no thresholding.
The 10,000-image data ablation uses a nested subset of the same 55,000 training set.
AR checkpoints minimize validation NLL. GAN checkpoints use predefined final epoch.
Test data do not determine checkpoint selection or hyperparameters.
Generative training is unconditional: labels are used only for auxiliary diagnostics.
Seeds are fixed, but exact floating-point reproducibility across hardware is not guaranteed.
Classifier diagnostics are imperfect proxies; neither confidence nor class coverage
proves realism or excludes within-class mode collapse.

6. Timing and plots
Training wall time includes epoch validation/progress output; update-only time is also saved.
Inference: 100 images, warm-up then median of three synchronized runs, excludes disk I/O.
All AR models use identical saved sampling uniforms, and GAN grids use fixed latent vectors.
The first 100 generated samples are displayed without manual selection.
The exact installed environment is listed in environment-lock.txt. It records local
CUDA wheel suffixes; requirements.txt is the portable installation entry point.

7. Sharing and submission
Upload this folder (code, README, data, results, notebook and report) to a stable shared
location accessible to the marker, and insert its URL into submission_metadata.json.
Rebuild the report, test the link while signed out, and upload 60434764.pdf to Canvas.
The code/data sharing URL is mandatory and must be provided by the student.
Freeze the shared code/data after submission, as required by the assignment.
