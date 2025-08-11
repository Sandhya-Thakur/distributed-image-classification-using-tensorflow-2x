# Distributed TensorFlow Comparison
Comparing single GPU vs distributed TensorFlow training speed on CIFAR-10 dataset.

## Project Structure
- source_code/: Python scripts for training
- jupyter_notebooks/: Colab notebooks with experiments
- training_results/: Logs, plots, and performance metrics
- dataset/: CIFAR-10 data (auto-downloaded)

## Goal
Measure how much faster distributed training is compared to single GPU training.

## ✅ EXPERIMENT COMPLETED!

### Results Summary
| Method | Training Time | Accuracy | Speed Improvement |
|--------|---------------|----------|------------------|
| Single GPU | 356.01 seconds | 67% | Baseline |
| Distributed | 303.74 seconds | 66% | **14.7% faster** |

### Key Findings
🚀 **Distributed training saved 52.27 seconds (14.7% improvement)**
- Both methods achieved similar accuracy (~66-67%)
- Distributed approach optimized training even with 1 GPU
- Demonstrates the power of TensorFlow's distributed strategies

## Scripts Created
- ✅ `source_code/train_single_gpu.py` - Standard training approach
- ✅ `source_code/train_distributed.py` - Optimized distributed training
- ✅ `training_results/comparison_results.txt` - Detailed results

## Conclusion
Even with limited hardware, TensorFlow's distributed strategies can provide meaningful performance improvements for machine learning training tasks.
