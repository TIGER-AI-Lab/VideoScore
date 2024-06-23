# MantisScore
official repo for "MantisScore: A Reliable Fine-grained Metric for Video Generation"

## Introduction


## Installation

- for inference
```bash
pip install -e . 
```
- for evaluation
```bash
pip install -e .[eval] 
```
- for training
```bash
git clone https://github.com/TIGER-AI-Lab/Mantis
cd Mantis
pip install -e .[train,eval]
pip install flash-attn --no-build-isolation
# then training scripts are in Mantis/train/scripts
```

## Inference examples
```bash
cd examples
python run_mantiscore.py
```


## Citation
```bibtex
```
