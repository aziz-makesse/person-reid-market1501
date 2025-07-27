# Person Re‑Identification on Market‑1501

This repository implements a two‑stage person Re‑ID pipeline on the Market‑1501 dataset:  
1. A baseline initial approach (see `first_approach.md`).  
2. A fine‑tuned ResNet50 classifier followed by feature extraction and evaluation.

## Repository Structure

- `first_approach.md`  
  Detailed description of the baseline approach using ImageNet‑pretrained ResNet50 and cosine similarity.
- `market1501_training.ipynb`  
  Colab notebook for dataset download, fine‑tuning ResNet50 on Market‑1501 IDs, backbone saving, feature extraction, and evaluation.
- `resnet50_market1501_backbone.pth`  
  Trained backbone weights (no classification head).
- `README.md`  
  This file.

## Fine‑Tuning Approach

1. **Dataset Preparation**  
   - Download via Kaggle API.  
   - Verify `bounding_box_train`, `query`, `bounding_box_test` folders.

2. **Model Architecture**  
   - ResNet50 pretrained on ImageNet.  
   - Replace the final fully‑connected layer with a new linear layer for 751 classes (Market‑1501 IDs).

3. **Training Configuration**  
   - Input images resized to 128×64.  
   - Cross‑Entropy loss, Adam optimizer (lr=1e‑4), batch size 32.  
   - OOM handling reduces batch size on GPU memory errors.

4. **Backbone Extraction**  
   - After training, remove the classification head and save the convolutional backbone:
     ```python
     backbone = nn.Sequential(*list(model.children())[:-1])
     torch.save(backbone.state_dict(), "resnet50_market1501_backbone.pth")
     ```

## Evaluation

1. **Feature Extraction**  
   - Load saved backbone weights into ResNet50 architecture.  
   - Extract 2048‑D embeddings for all query and gallery images.

2. **Distance Computation**  
   - Compute pairwise cosine distances batch‑wise to avoid OOM.

3. **Metrics**  
   - **Rank‑1 Accuracy**: percentage of queries whose nearest gallery match has the same ID (different camera).  
   - **mAP**: mean Average Precision over all query instances, excluding same‑camera and junk images.

**Final results after fine‑tuning**  
  Rank‑1 Accuracy: 63.93%
  mAP: 41.01%

## Visualization

The notebook includes a section to display, for five random queries, the query image alongside its top‑5 gallery matches, annotated with their cosine distances.

## Usage

1. Clone this repository.  
2. Place `kaggle.json` in the root or use Colab to upload.  
3. Open and run `notebooks/market1501_training.ipynb` on Colab or a GPU‑enabled environment.  
4. Fine‑tune the model, extract features, evaluate metrics, and visualize matches.

## Next Steps

- Implement triplet or contrastive loss for improved embedding separation.  
- Experiment with specialized Re‑ID architectures (OSNet, PCB).  
- Deploy the backbone in an API (FastAPI or Streamlit) for live Re‑ID demonstrations.  
