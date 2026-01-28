# Biased Complementary Label Learning (bias_cll)

[![Documentation Status](https://readthedocs.org/projects/libcll/badge/?version=latest)](https://libcll.readthedocs.io/en/latest/?badge=latest) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository extends `libcll` with support for **biased complementary label distributions** through custom transition matrices. It provides a comprehensive collection of pre-computed transition matrices for various datasets and bias patterns, enabling researchers to study complementary-label learning under realistic, non-uniform labeling conditions.

## Key Features

- **Custom Transition Matrices**: Load and apply transition matrices from a repository of pre-computed bias patterns
- **Extended Dataset Support**: Includes support for CIFAR-10, CIFAR-20, CIFAR-100, and Tiny-ImageNet-200 with biased complementary labels
- **Flexible Bias Patterns**: Support for various bias types including VLM-annotated patterns, clustering-based biases, and noise-injected distributions

## Installation

- Python version >= 3.8, <= 3.12
- Pytorch version >= 1.11, <= 2.0
- Pytorch Lightning version >= 2.0
- To install and develop locally:

```bash
git clone <this-repository-url>
cd bias_cll
pip install -e .
```

## Transition Matrix Repository

This repository includes a comprehensive collection of transition matrices located in `libcll/transition_matrix/`:

### Supported Datasets

- **CIFAR-10** (`libcll/transition_matrix/cifar10/`)
- **CIFAR-20** (`libcll/transition_matrix/cifar20/`)
- **CIFAR-100** (`libcll/transition_matrix/cifar100/`)
- **Tiny-ImageNet-200** (`libcll/transition_matrix/tiny200/`)

### Transition Matrix Types

Each dataset directory contains multiple transition matrix files representing different bias patterns:

- **VLM-Annotated**: Transition matrices derived from Vision-Language Model annotations (e.g., `llava_*.txt`)
- **Clustering-Based**: Matrices based on visual similarity clustering (e.g., `llava_kmean=*_nrand=*.txt`)
- **Most and Least**: Least/most relevant classes to 4 random labels (e.g., `least.txt`, `most.txt`)
- **Noise-Injected**: Controlled noise condition, whether the options has true label (e.g., `llava_noise=True-nrand=*.txt`)
- **Random**: Random uniform patterns (e.g., `random.txt`)
- **Custom**: Custom-defined patterns (e.g., `weak_10.txt`, `strong_20.txt`)

## Quick Start: Training with Transition Matrices

### Default: Uniform Distribution

When no `--transition_matrix` argument is provided, the system defaults to uniform complementary label distribution:

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model ResNet18 \
  --dataset cifar10 \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy
```

### Using a Custom Transition Matrix

To use a specific transition matrix from the repository, use the `--transition_matrix` argument with the filename (without `.txt` extension):

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model ResNet18 \
  --dataset cifar10 \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --transition_matrix llava_kmean=10_nrand=4
```

### Example: CIFAR-10 with VLM-Based Bias

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy CPE \
  --type T \
  --model ResNet18 \
  --dataset cifar10 \
  --lr 1e-4 \
  --batch_size 128 \
  --valid_type Accuracy \
  --transition_matrix llava_noise=False-nrand=4

### Example: CIFAR-100 with Clustering-Based Bias

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model ResNet34 \
  --dataset cifar100 \
  --lr 1e-4 \
  --batch_size 128 \
  --valid_type Accuracy \
  --transition_matrix llava_kmean=100_nrand=4
```

### Example: Tiny-ImageNet-200 with Random Bias

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy URE \
  --type TNN \
  --model ResNet50 \
  --dataset tiny200 \
  --lr 1e-4 \
  --batch_size 64 \
  --valid_type Accuracy \
  --transition_matrix random
```

## Supported Strategies

All strategies from the original `libcll` are supported:

| Strategy | Type | Description |
|----------|------|-------------|
| [PC](https://arxiv.org/pdf/1705.07541) | None | Pairwise-Comparison Loss |
| [SCL](https://arxiv.org/pdf/2007.02235.pdf) | NL, EXP | Surrogate Complementary Loss |
| [URE](https://arxiv.org/pdf/1810.04327.pdf) | NN, GA, TNN, TGA | Unbiased Risk Estimator |
| [FWD](https://arxiv.org/pdf/1711.09535.pdf) | None | Forward Correction |
| [DM](http://proceedings.mlr.press/v139/gao21d/gao21d.pdf) | None | Discriminative Models with Weighted Loss |
| [CPE](https://arxiv.org/pdf/2209.09500.pdf) | I, F, T | Complementary Probability Estimates |
| [MCL](https://arxiv.org/pdf/1912.12927.pdf) | MAE, EXP, LOG | Multiple Complementary Label Learning |
| [OP](https://proceedings.mlr.press/v206/liu23g/liu23g.pdf) | None | Order-Preserving Loss |
| [SCARCE](https://arxiv.org/pdf/2311.15502) | None | Selected-Completely-At-Random CLL |

## Supported Datasets

### Extended Datasets (with transition matrix support)
- **CIFAR-10**: 10 classes, 3×32×32 colored images
- **CIFAR-20**: 20 classes, 3×32×32 colored images  
- **CIFAR-100**: 100 classes, 3×32×32 colored images
- **Tiny-ImageNet-200**: 200 classes, 3×64×64 colored images

### Standard Datasets (from libcll)
- MNIST, FMNIST, KMNIST
- Yeast, Texture, Dermatology, Control
- CIFAR-10, CIFAR-20, CIFAR-100
- Micro ImageNet-10, Micro ImageNet-20
- CLCIFAR-10, CLCIFAR-20 (with human-annotated complementary labels)
- ACLCIFAR-10, ACLCIFAR-20 (with VLM-annotated complementary labels)


## Batch Training Scripts

Run comprehensive experiments across different bias patterns:

```bash
# Uniform distribution baseline
./scripts/uniform.sh <strategy> <type>

# Biased distributions (weak/strong)
./scripts/biased.sh <strategy> <type>

# Noisy distributions
./scripts/noisy.sh <strategy> <type>

# Multiple complementary labels
./scripts/multi.sh <strategy> <type>
./scripts/multi_hard.sh <strategy> <type>
```

Example:
```bash
./scripts/uniform.sh SCL NL
./scripts/biased.sh CPE T
```

## Understanding Transition Matrices

A transition matrix `Q` defines the probability distribution for complementary label generation. Each element `Q[i][j]` represents the probability of assigning complementary label `j` given true label `i`.

- **Uniform**: All complementary labels (excluding true label) are equally likely
- **Biased**: Some complementary labels are more likely than others
- **VLM-Based**: Bias patterns derived from Vision-Language Model confusion patterns
- **Clustering-Based**: Bias based on visual similarity clustering before VLM

The transition matrix is automatically loaded when specified via `--transition_matrix`. The system searches for the corresponding `.txt` file in the appropriate dataset subdirectory under `libcll/transition_matrix/`. If the flag isn't specified, it would use random uniform distribution.

## Advanced Usage


### Custom Seed for Reproducibility

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy SCL \
  --type NL \
  --model ResNet18 \
  --dataset cifar20 \
  --lr 1e-4 \
  --batch_size 256 \
  --valid_type Accuracy \
  --transition_matrix least \
  --seed 42
```

### Data Augmentation Options

```bash
python scripts/train.py \
  --do_train \
  --do_predict \
  --strategy CPE \
  --type T \
  --model ResNet18 \
  --dataset cifar100 \
  --lr 1e-4 \
  --batch_size 128 \
  --valid_type Accuracy \
  --transition_matrix llava_kmean=100_nrand=4 \
  --augment autoaugment
```

## Output and Logging

Training results are logged to `lightning_logs/` with TensorBoard support. Each run creates:

- **Metrics CSV**: Detailed epoch-by-epoch metrics (`metrics.csv`)
- **Summary JSON**: Final results and hyperparameters (`summary.json`)
- **Transition Matrix**: The used transition matrix saved as `<dataset>.txt`
- **Checkpoints**: Best model checkpoints (if enabled)

View training progress with TensorBoard:
```bash
tensorboard --logdir lightning_logs/
```

## Citation

If you use this repository, please cite the original `libcll` work and mention this extension:

```bibtex
@techreport{libcll2024,
  author = {Nai-Xuan Ye and Tan-Ha Mai and Hsiu-Hsuan Wang and Wei-I Lin and Hsuan-Tien Lin},
  title = {libcll: an Extendable Python Toolkit for Complementary-Label Learning},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libcll},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/2411.12276}},
  month = nov,
  year = 2024
}
```

## Documentation

For more details on the base library, visit the [libcll documentation](https://libcll.readthedocs.io/en/latest/).

## Acknowledgments

This work extends [libcll](https://github.com/ntucllab/libcll) with transition matrix support for biased complementary label learning research. We thank the original authors and all contributors to the repositories that made this work possible.
