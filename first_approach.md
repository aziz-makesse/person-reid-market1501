# Baseline Approach: ImageNet‑Pretrained ResNet50 + Cosine Similarity

## Overview

In the baseline, we use a ResNet50 model pretrained on ImageNet without any further training on Market‑1501. The goal is to assess how generic ImageNet features perform on the Re‑ID task.

## Pipeline

1. **Data Loading**  
   - A custom `Dataset` reads all `.jpg` files from `query/` and `bounding_box_test/` directories.

2. **Feature Extraction**  
   ```python
   resnet = models.resnet50(pretrained=True)
   backbone = nn.Sequential(*list(resnet.children())[:-1])
   backbone.eval()
  - Remove the final classification layer to obtain a 2048‑dim vector per image.
  
3. **Distance Computation**
  ```python
  dist = 1 - F.cosine_similarity(
    q_feats.unsqueeze(1),
    g_feats.unsqueeze(0),
    dim=2
  )
  ```

4. **Metric Calculation**
  - Rank‑1 Accuracy: ratio of queries whose nearest gallery vector matches the same person ID.

  - mAP: mean Average Precision, taking into account multiple correct matches and excluding same‑camera cases.

## Baseline Results

  - Rank‑1 Accuracy: 91.06%

  - mAP: 11.98%

## Analysis

  - High Rank‑1 is misleading: many correct matches come from the same camera or even identical frames.

  - Very low mAP indicates the model fails to consistently rank all true matches highly.

  - This baseline motivated the fine‑tuning strategy to improve overall ranking quality.
