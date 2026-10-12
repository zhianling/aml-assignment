# Bird Species Classification on CUB-200-2010

COS30082 Applied Machine Learning, Assignment 1: fine-grained classification of 200 North American bird
species from about 24 training photographs per species.

Fourteen pretrained networks were screened as frozen feature extractors, four were fine-tuned, and the best
one was taken apart in an ablation study. The final model, a fine-tuned **ConvNeXt-L** pretrained with CLIP
and then on iNaturalist 2021, reaches **96.6% test top-1 accuracy and 96.6% average per-class accuracy**,
against 77.9% for the best fine-tuned ImageNet ResNet-50.

## Contents

- [Method](#method)
- [Results](#results)
- [Findings and discussion](#findings-and-discussion)
- [Code structure](#code-structure)
- [Reproducing the results](#reproducing-the-results)

## Method

**Data and evaluation protocol.** CUB-200-2010 has 4,829 training and 1,204 test photographs. `train.txt`
is split once, stratified by class, into a training split (3,870 images, 13-25 per class) and a validation
(val) split (959 images, 3-6 per class). Every choice (checkpoint, probe weight decay, feature, resolution,
temperature, ensemble members) is made on val; the test set is scored once, after training. An audit with
perceptual hashes and embedding similarity found 63 test images that duplicate training images, four of them
with conflicting labels. Accuracy on the remaining 1,141 images (the *clean test* subset) is reported beside
the official test set; the two differ negligibly for every model.

**Metrics.** Top-1 accuracy and average per-class accuracy (the mean of the 200 per-class accuracies) are
the target metrics. One test image is worth 0.08 percentage points (pp), and the 95% interval of a test
top-1 near 96% is about ±0.9 pp, so models are compared with stratified bootstrap 95% intervals and exact
McNemar tests on paired predictions. Expected calibration error (ECE, 15 bins) after temperature scaling
shows whether the predicted confidence can be trusted.

**Models.** Two CNNs are studied in depth, with two vision transformers (ViTs) as references:

- **ResNet-50**, the classic CNN baseline (25.6 M parameters, ImageNet-1k supervised weights);
- **ConvNeXt-L** (214 M parameters), CLIP on LAION-2B, then iNaturalist 2021 (iNat21, 10,000 species);
- **EVA-02-L** (CLIP, then iNat21) and **DINOv3 ViT-L/16** (self-supervised), both at 336 px.

They were shortlisted from 14 backbones, each screened by a linear probe on its frozen features, by role
rather than by probe score: the classic CNN baseline, the best CNN, a ViT with similar pretraining, and the
best self-supervised ViT.

**Training.**

- *Linear probing*: frozen features, averaged over the image and its mirror image, fitted by multinomial
  logistic regression (L-BFGS; L2 penalty chosen on val from seven values, 1e-5 to 1e-2).
- *Fine-tuning with LP-FT*: the whole network is trained with its classifier initialized from the linear
  probe, so the first gradients come from a fitted head and do not distort the pretrained features.
- *Loss and optimizer*: cross-entropy with label smoothing 0.1; AdamW (weight decay 0.05), effective batch
  64, two warm-up epochs then cosine decay, gradient clipping at norm 1, fp16 mixed precision. Layer-wise
  learning-rate (LR) decay gives the earliest layers the smallest LR.
- *Regularization*: random resized crops and horizontal flips in every run, plus TrivialAugment, random
  erasing, drop-path, weight decay, an exponential moving average (EMA) of the weights and label smoothing.
- *Selection and inference*: each epoch scores the EMA weights on val and keeps the best. Predictions
  average the softmax outputs of the image and its mirror image, and a temperature fitted on val calibrates
  them without changing any prediction.

| Model | Input (px) | Epochs (kept) | Peak LR | Layer decay | Drop-path |
|---|--:|--:|--:|--:|--:|
| ResNet-50 (cosine LR) | 448 | 30 (28) | 3e-4 | – | 0.05 |
| ResNet-50 (plateau LR) | 448 | 51 (35) | 3e-4 | – | 0.05 |
| DINOv3 ViT-L/16 | 336 | 30 (17) | 1e-4 | 0.8 | 0.2 |
| EVA-02-L | 336 | 20 (17) | 5e-5 | 0.8 | 0.2 |
| ConvNeXt-L | 448 | 20 (10) | 5e-5 | 0.9 | 0.3 |

*Kept* is the epoch chosen on val, counted from 0. The plateau schedule keeps the peak LR until val accuracy
stops improving for 5 epochs, then cuts it tenfold, and ends the run after 15 epochs without improvement.

To measure overfitting directly, every epoch also scores a fixed 959-image subset of the training images
exactly like val (no augmentation), so training and val curves of accuracy and negative log-likelihood (NLL)
compare like for like.

## Results

All numbers are percentages except ECE. Every model is trained on the training split only.

| Model | Pretraining | Val top-1 | Test top-1 [95% CI] | Test per-class | Clean test | ECE |
|---|---|--:|--:|--:|--:|--:|
| ResNet-50, fine-tuned (cosine LR) | ImageNet-1k | 77.3 | 74.7 [72.7, 76.7] | 74.1 | 73.7 | 0.029 |
| ResNet-50, fine-tuned (plateau LR) | ImageNet-1k | 79.1 | 77.9 [76.0, 79.9] | 77.5 | 77.0 | 0.037 |
| DINOv3 ViT-L/16, fine-tuned | self-supervised | 96.4 | 95.3 [94.3, 96.4] | 95.5 | 95.4 | 0.015 |
| EVA-02-L, fine-tuned | CLIP → iNat21 | 97.2 | 96.4 [95.5, 97.3] | 96.4 | 96.6 | 0.016 |
| **ConvNeXt-L, fine-tuned (final)** | **CLIP → iNat21** | **97.5** | **96.6 [95.7, 97.4]** | **96.6** | **96.7** | **0.015** |
| ConvNeXt-L, linear probe | CLIP → iNat21 | 97.8 | 96.1 [95.2, 97.0] | 96.1 | 96.2 | 0.019 |
| EVA-02-L, linear probe | CLIP → iNat21 | 97.9 | 96.8 [95.9, 97.7] | 96.9 | 97.0 | 0.020 |
| BioCLIP 2.5 ViT-H/14, linear probe | CLIP on TreeOfLife | 98.2 | 96.4 [95.5, 97.3] | 96.4 | 96.6 | 0.016 |
| Ensemble of four (chosen on val) | mixed | 98.2 | 97.1 [96.2, 97.8] | 97.1 | 97.3 | 0.014 |

The full tables are in [`results/report/comparison.csv`](results/report/comparison.csv) (every model and
probe) and [`results/phase1_probes.csv`](results/phase1_probes.csv) (every probe of the 14 backbones).

<details>
<summary>Ablation of the ConvNeXt-L recipe (one change per run)</summary>

Changes are against the mean of three control seeds (val 97.74 ± 0.22%, test 96.23 ± 0.33%); Δval is in
control standard deviations (σ), and McNemar *p* compares each run with control seed 0 on test.

| Run | Val top-1 | Δval (σ) | Test top-1 | Δtest (pp) | McNemar *p* |
|---|--:|--:|--:|--:|--:|
| Control, seed 0 | 97.5 | -1.1 | 96.6 | +0.36 | – |
| Control, seed 1 | 97.9 | +0.8 | 96.2 | -0.06 | 0.33 |
| Control, seed 2 | 97.8 | +0.3 | 95.9 | -0.30 | 0.08 |
| No label smoothing | 97.5 | -1.1 | 96.6 | +0.36 | 1.00 |
| No TrivialAugment | 97.6 | -0.6 | 96.2 | -0.06 | 0.38 |
| No random erasing | 97.9 | +0.8 | 96.4 | +0.19 | 0.80 |
| No drop-path | 97.7 | -0.2 | 95.8 | -0.39 | 0.06 |
| No weight decay | 97.5 | -1.1 | 96.6 | +0.36 | 1.00 |
| No EMA | 97.6 | -0.6 | 96.2 | -0.06 | 0.18 |
| All six regularizers off | 97.4 | -1.6 | 95.3 | -0.89 | 0.004 |
| Random head (no LP-FT) | 96.7 | -5.0 | 96.1 | -0.14 | 0.36 |
| One LR for all layers | 97.1 | -3.0 | 95.6 | -0.64 | 0.03 |
| + Mixup/CutMix | 97.6 | -0.6 | 96.3 | +0.03 | 0.45 |
| + Head dropout 0.3 | 97.5 | -1.1 | 96.3 | +0.03 | 0.42 |
| Drop-path 0.3 → 0.6 | 97.4 | -1.6 | 96.2 | -0.06 | 0.33 |
| TrivialAugment → RandAugment | 97.7 | -0.2 | 95.9 | -0.30 | 0.08 |
| Peak LR × 0.5 | 97.9 | +0.8 | 96.4 | +0.19 | 0.73 |
| Peak LR × 2 | 97.5 | -1.1 | 96.4 | +0.19 | 0.77 |

Tables with every logged column: [`results/ablation.csv`](results/ablation.csv) and, for the ResNet-50
study, [`results/ablation_resnet.csv`](results/ablation_resnet.csv).

</details>

## Findings and discussion

**Pretraining data matters more than the architecture.** The same ConvNeXt-L reaches 86.6% val as a probe
with distilled DINOv3 weights, 93.2% with CLIP then ImageNet-12k, 93.7% with ImageNet-22k, and 97.8% with
CLIP then iNat21. The second and fourth both start from CLIP on LAION-2B, and the species-labeled variant is
4.6 pp higher. The three best probes all saw species-labeled data, and BioCLIP 2.5 reaches 92.4% test even
zero-shot, matching each photograph to the 200 species names with no CUB training. Fine-tuning helps less
the closer the pretraining is to birds: it adds 0.5 pp on test to the ConvNeXt-L probe but 10.6-13.9 pp to
the ResNet-50 probe.

**Size and resolution help up to about ViT-L and 336 px.** Species differ by small cues (bill shape, wing
bars, eye rings), and at the standard 224 px input the median photograph loses about two thirds of its
pixels. But DINOv3 ViT-H+ (840 M parameters) probes below ViT-L (303 M), DINOv3's probe gains only 0.3 pp
on val from 256 to 448 px, and EVA-02-L is as accurate at 336 as at 448 px. Only 8.8% of the photographs
have a short side of 448 px or more, so larger inputs only enlarge the photograph. The three large
fine-tuned models are past that limit, so their differences come from their pretraining data.

**The final model.** ConvNeXt-L is the best fine-tuned model on val. On test it ties EVA-02-L (10 against 8
images that only one of the two gets right; McNemar *p* = 0.82), is ahead of DINOv3 (25 against 10,
*p* = 0.017) and beats the best ResNet-50 by 18.7 pp. The fine-tuned models are under-confident (fitted
temperatures 0.59-0.77, an effect of label smoothing), and temperature scaling lowers ConvNeXt-L's ECE from
0.063 to 0.015. The ensemble of four reaches 97.1% test, but its gain over the single model is not
significant (*p* = 0.077) and it needs four forward passes per image, so the single ConvNeXt-L is the final
model.

**Overfitting: better features fixed it, regularization did not.** ResNet-50 classifies 100% of its
training images by epoch 7 while val stays between 72% and 77%; its val NLL is lowest at epoch 8 (1.02) and
then rises to 1.21 as the network grows over-confident on the images it gets wrong. The gap is already in
its frozen ImageNet-1k features: their probe classifies 99.95% of the training images but only 65.3% of
val. None of five one-change variants (all regularizers off, Mixup/CutMix, layer-wise LR decay, more
drop-path, a third of the LR) raised its val accuracy; restraining the weights hurt most, because ImageNet
features have to move far to become bird features. Only the LR schedule helped: the plateau schedule raises
val from 77.3% to 79.1% and test from 74.7% to 77.9% over cosine. ConvNeXt-L shows no harmful overfitting:
its val accuracy stays at 96.9-97.5% from the first epoch, and its gap between val and training NLL stays at
0.13, against 1.0-1.15 for ResNet-50.

**In the ablation, the fine-tuning scheme matters most and the regularizers only together.** A random head
instead of LP-FT is the largest effect (-5.0σ on val; it needs four epochs to reach 94%), and one LR for all
layers costs 3.0σ. Removing any one of the six regularizers keeps val within 1.1σ, but removing all six makes
the run peak at the first epoch and overfit at once, and test accuracy drops by 0.9 pp (*p* = 0.004).
Individually, weight decay is inert at this LR, label smoothing moves only the calibration, drop-path keeps
the val NLL down (0.17 instead of 0.25), and EMA mainly stabilizes training. Halving or doubling the LR, or
RandAugment instead of TrivialAugment, moves val by at most 1.1σ: the recipe sits on a plateau. Control seeds
1 and 2 disagree with seed 0 on 17 and 16 test images, so test changes below about 0.7 pp are noise.

**Errors are mostly look-alike species.** 169 of the 200 species are classified without error, and 8 fall
below 80%. The weakest are look-alikes that birders separate mainly by voice or range: American Crow (40%,
mostly called Fish Crow), Common Tern (43%) and Whip-poor-will (50%, called Chuck-will's-widow). Grad-CAM
shows that in 33 of the 41 test errors the model found the bird and matched it to a look-alike; in 5 the
photograph holds several birds, and in 3 the bird is hidden or outside the center crop. Two errors look like
label errors.

**Could test photographs have been seen in pretraining?** CUB's photographs come from Flickr, and none of
the pretraining sets was deduplicated against them, so memorized test images cannot be ruled out. But
exposure does not explain the ranking: ResNet-50, trained only on ImageNet-1k (the overlap the dataset's
authors warn about), is last, while the final ConvNeXt-L has no ImageNet stage. What the two leaders share
is iNat21, whose photographs come from iNaturalist, not Flickr, and whose classes include 180 of the 200 CUB
species: a class overlap rather than a photo overlap.

## Code structure

```
src/birdcls/        the Python package behind every script
  config.py           run settings: one TOML file per run, loaded into a typed TrainConfig
  data.py             dataset parsing, the stratified val split, transforms and augmentation
  audit.py            exact and near-duplicate detection; the flagged test images
  backbones.py        registry of the 14 pretrained backbones
  features.py         frozen-feature extraction, cached once per backbone
  probe.py            linear probes, k-NN and zero-shot scoring on cached features
  models.py           fine-tuning model: 200-way head, LP-FT head and layer-wise LR decay
  train.py            resumable fine-tuning loop with per-epoch logging
  checkpoint.py       crash-safe checkpoints written in the background
  infer.py            saved logits of a finished run (image and mirror image)
  analysis.py         calibration, ensembles and selective prediction from saved logits
  metrics.py          accuracy, bootstrap intervals, McNemar tests and ECE
  probs.py            softmax, test-time averaging and log-loss
  fileio.py           JSON Lines, CSV and NumPy file helpers
  runtime.py          device and mixed-precision selection
scripts/            command-line entry points (below)
configs/            the five fine-tuning runs; configs/ablation/ holds the 24 ablation runs
heads/              linear-probe heads that initialize fine-tuning (LP-FT), fitted on the training split
metadata/           class names, the val split and the duplicate audit
results/            probe table, model comparison, ablation tables and curves
  run_logs/           config, environment, per-epoch metrics, per-update steps and summary of every run
notebooks/          data exploration and model analysis, saved with their outputs
tests/              pytest suite
```

| Script | Purpose |
|---|---|
| `prepare_data.py` | Unpack and check the dataset, draw the val split and audit the images for duplicates |
| `extract_features.py` | Cache the frozen embeddings of every image for each backbone |
| `audit_embeddings.py` | Match each test image to its most similar training image (same-session shots) |
| `run_probes.py` | Fit and score linear probes and k-NN on every feature cache |
| `make_probe_run.py` | Package a linear probe as a run folder, so it is scored like a fine-tuned model |
| `benchmark_step.py` | Measure a config's training speed and memory on random images |
| `compare_models.py` | Score every model and probe, choose an ensemble on val and run the McNemar tests |
| `make_ablation_configs.py` | Write the configs of both ablation studies |
| `ablation_report.py` | Score an ablation study against its control seeds |
| `kaggle_push.py`, `kaggle_kernel.py`, `remote_job.py` | Train configs on a hosted Kaggle GPU, resuming across sessions |

## Reproducing the results

The dataset, cached features and trained weights are not in the repository. The CUB-200-2010 files
(`Train.zip`, `Test.zip`, `train.txt`, `test.txt`) come with the assignment. With
[uv](https://docs.astral.sh/uv/) and Python 3.11 or newer:

```bash
uv sync --extra local --group dev
uv run pytest

# Data, frozen features and probes
uv run python scripts/prepare_data.py --src <folder with the four dataset files>
uv run python scripts/extract_features.py
uv run python scripts/extract_features.py --img-size 448
uv run python scripts/audit_embeddings.py
uv run python scripts/run_probes.py

# Fine-tuning: writes runs/<run>/ with the weights, logs, predictions.npz and summary.json
uv run python -m birdcls.train --config configs/convnext_l_inat21_448.toml

# Comparison and ablation tables
uv run python scripts/compare_models.py --runs-dir runs --include '^(LP:|FT:(?!abl_))' --out-dir results/report
uv run python scripts/make_ablation_configs.py
uv run python scripts/ablation_report.py --study convnext
```

Rerunning a training command after an interruption resumes from the last checkpoint, and a finished run is
skipped. `python -m birdcls.infer --run runs/<run>` recreates a run's logits from its weights.
