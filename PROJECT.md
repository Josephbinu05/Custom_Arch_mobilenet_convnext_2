# Project methodology and provenance

## Problem and motivation

This completed research project investigates efficient plant-disease image classification for deployment settings where memory and computational resources are constrained. It frames the task as three-class classification of `Healthy`, `Powdery`, and `Rust` leaf images.

The central question is whether teacher-guided training can preserve useful predictive performance in a deployable MobileNetV2 student while the heavier **ConvNeXt V2-Inspired Teacher** is used only during training.

## Experimental pipeline

The recorded work contains two related experiments:

1. A six-model transfer-learning comparison: ResNet50, MobileNetV2, EfficientNet-B0, DenseNet121, ConvNeXt-Tiny, and VGG16.
2. A baseline-versus-teacher-versus-student comparison consisting of MobileNetV2 baseline, a ConvNeXt V2-Inspired Teacher, and a distilled MobileNetV2 student.

The primary final comparison is [`results/metrics/final_3_model_comparison.csv`](results/metrics/final_3_model_comparison.csv). The separate six-model comparison is retained in [`results/metrics/model_comparison.csv`](results/metrics/model_comparison.csv).

## Dataset and preprocessing

The retained directory structure provides 1,322 training, 60 validation, and 150 test images, evenly represented in validation and test splits across the three classes. The main workflow resizes inputs to 224 x 224 and converts them to tensors. The six-model comparison additionally records normalization and training augmentation. The original data source and license cannot be verified from the retained files and are not claimed here.

## Models

### MobileNetV2 baseline

The baseline is torchvision MobileNetV2 with its final classifier adapted to three classes. It is trained using cross-entropy loss and AdamW in the main workflow.

### ConvNeXt V2-Inspired Teacher

The teacher is torchvision ConvNeXt-Tiny with a modified classifier containing flattening, LayerNorm, a GRN-inspired component, and a final linear classifier. It is accurately described as a **ConvNeXt V2-Inspired Teacher** or as **ConvNeXt-Tiny with a GRN-inspired classification head**. It is not an official/full ConvNeXt V2 implementation.

The recorded teacher training uses label smoothing, AdamW, weight decay, and cosine annealing. Its role is to generate logits for training the student; it is frozen before distillation.

### Distilled MobileNetV2 student

The student is MobileNetV2 with a three-class output layer. Only the student is retained for inference. The teacher and student are never merged into a single inference architecture.

## Knowledge-distillation objective

For student logits `z_s`, teacher logits `z_t`, labels `y`, temperature `T`, and hard-label weight `alpha`, the retained implementation computes:

`L_CE = CrossEntropy(z_s, y)`

`L_KL = T^2 * KLDiv(log_softmax(z_s / T), softmax(z_t / T))`

`L_KD = alpha * L_CE + (1 - alpha) * L_KL`

The recorded configuration uses `alpha = 0.7`, `T = 4.0`, and five KD epochs. The student optimizer is AdamW with learning rate and weight decay both set to `1e-4`.

## Evaluation methodology

Predictive metrics are accuracy, macro precision, macro recall, and macro F1-score. The final comparison also records parameter count, FLOPs, inference time, checkpoint size, and accuracy per GFLOP.

The main experiment uses `fvcore` for FLOPs; the separate six-model notebook uses THOP and computes FLOPs as twice the reported MACs. These results have different conventions and should not be mixed. `fvcore` reported unsupported operators during profiling, so recorded FLOPs should be interpreted as the original profiling outputs rather than a universal compute measurement. Model-size values labeled MB in the original CSVs were calculated using 1024 squared bytes and thus correspond to MiB.

## Final recorded findings

| Model | Accuracy | F1 | Parameters (M) | GFLOPs |
|---|---:|---:|---:|---:|
| MobileNetV2 baseline | 98.00% | 0.979854 | 2.227715 | 0.312917 |
| ConvNeXt V2-Inspired Teacher | 97.33% | 0.973319 | 27.823971 | 4.469672 |
| Distilled MobileNetV2 | 98.00% | 0.979854 | 2.227715 | 0.312917 |

These values are copied from the final saved comparison. They support performance preservation for the distilled MobileNetV2 relative to the recorded baseline while retaining the same MobileNetV2 architecture-level parameter count and GFLOPs. They do not support an assertion that KD improved accuracy or reduced baseline MobileNetV2 parameters or GFLOPs.

Relative to the recorded teacher, the student has approximately 92% fewer parameters and approximately 93% lower recorded GFLOPs. This is a deployment comparison between two architectures, not an efficiency change introduced to the baseline by distillation.

## Experimental provenance limitation

An earlier distilled-student output and `hybrid_model_efficiency_metrics.csv` report 97.33% accuracy and 0.973503 macro F1, while the final three-model CSV and the paper report 98% accuracy and 0.979854 macro F1. The available saved execution history does not establish the exact cause of the difference, and the final comparison cell reused prior metric variables. Both records are preserved; the final reported comparison is used for the repository results table because it is the designated final CSV and is consistent with the paper.

## TPE / Optuna status

Saved notebook history names `optuna`, `study`, and `tpe_objective`, but no retained source cell, study database, trial results, search ranges, or best parameter record was found. There are no recoverable TPE findings in this package, and none are claimed.

## Optional advisory component

`ollama_helper.py` calls a local Ollama endpoint using the Mistral model name. It accepts a classifier output and produces contextual agricultural guidance. It is an optional post-classification component, separate from the evaluation pipeline, and requires a locally running Ollama service.

## Limitations and future directions

The recorded dataset is small and limited to three classes; latency depends on the original hardware and runtime; checkpoint provenance is incomplete; and the notebooks retain historic outputs and execution-order dependencies. Useful next research steps include evaluation on larger and more diverse data, repeatable tracked experiment runs, target-device benchmarking, systematic KD search with retained Optuna artifacts, and region-specific multilingual advisory output.

## References

See the completed paper for the full bibliography: [`paper/edge_efficient_knowledge_distilled_plant_disease_detection.pdf`](paper/edge_efficient_knowledge_distilled_plant_disease_detection.pdf). It includes references for ResNet, DenseNet, EfficientNet, VGG, ConvNeXt, ConvNeXt V2, MobileNet, MobileNetV2, knowledge distillation, FitNets, and DeiT.
