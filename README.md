# Distributed TensorFlow Comparison
Comparing single GPU vs distributed TensorFlow training speed on CIFAR-10 dataset.

## Project Structure
- source_code/: Python scripts for training
- jupyter_notebooks/: Colab notebooks with experiments
- training_results/: Logs, plots, and performance metrics
- dataset/: CIFAR-10 data (auto-downloaded)

## Goal
Measure how much faster distributed training is compared to single GPU training.

## Progress So Far
✅ **Single GPU Training Completed**
- Dataset: CIFAR-10 (60,000 images, 10 classes)
- Model: Simple CNN with 2 Conv layers
- Training time: **356.01 seconds** (5 epochs)
- Final accuracy: 67%
- Script: `source_code/train_single_gpu.py`

## Next Steps
- [ ] Create distributed training script
- [ ] Compare training speeds
- [ ] Generate performance charts
- [ ] Document findings

## Results Summary
| Method | Training Time | Accuracy |
|--------|---------------|----------|
| Single GPU | 356.01 seconds | 67% |
| Distributed | TBD | TBD |
