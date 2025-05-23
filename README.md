[![DOI](https://zenodo.org/badge/988950682.svg)](https://doi.org/10.5281/zenodo.15496546)
# Enhancing Participatory Mapping through AI: Detecting Hand-drawn Markings Using Siamese YOLOv9e
This Repository contains the model code used to train our custom siamese YOLOv9e, based on a fork from the [Ultralytics](https://github.com/ultralytics/ultralytics) package.
See [Ultralytics Fork for Mulitspectral & Change Detection](#ultrlytics-fork-for-mulispectral-&-change-detection) for general information how to setup a training using our custom YOLOv9e Changedetection.

## Abstract

Participatory Mapping empowers communities to contribute localized spatial knowledge vital for urban planning, disaster preparedness, and environmental risk assessment. These valuable inputs are often captured in analogue formats—such as Sketch Maps—to bridge the digital gap and include local population. However, these analogue maps pose significant challenges for digital interpretation due to visual variability, scanning artefacts, and complex backgrounds. The Sketch Map Tool (SMT) addresses this through a multi-stage deep learning pipeline that extracts annotations from scanned maps. We enhance the SMT by replacing its object detection module with a Siamese YOLOv9e architecture. Our dual-input approach processes both clean and annotated versions of the same map, using feature-level fusion to isolate user-added content. 
Trained on a large-scale dataset of synthetic and real-world Sketch Maps, our approach improves recall, precision, and mean average precision. Experiments across OpenStreetMap and satellite imagery basemaps demonstrate improved robustness and generalization. This focused upgrade makes the SMT pipeline more scalable for automated Participatory Mapping, while keeping it easy to understand and practical to use in real-world field settings. This ensures communities can meaningfully contribute to spatial planning through inclusive, data-driven insights.

## Manuscript
The Manuscript has been submitted to the ACM GoodIT 2025 Conference. We will provide a link to the manuscript later on.
## Model Checkpoints

The model checkpoints for the siamese yolov9e are availabe under the following links through the sketchmaptool.

| Task                 | Model Name | Modification    |Purpose | URL Link                                                           |
|----------------------|------------|-----------------|---------|--------------------------------------------------------------------|
| Object Detection     | YOLO_OSM   | 6-Channel Input siamese YOLOv9 |Detects sketches on OSM | [download](https://sketch-map-tool.heigit.org/weights/SMT-OSM.pt)  |
| Object Detection     | YOLO_ESRI  | 6-Channel Input siamese YOLOv9 | Detects sketches on ESRI maps | [download](https://sketch-map-tool.heigit.org/weights/SMT-ESRI.pt) |

## Data
We are currently evaluating our options to provide the Public with our Annotated Datasets used for model Training, under the consideration of ToU and Privacy Laws.
Additionaly we aim to provide the code necessary to create Synthetic SketchMaps for Training.

# Ultralytics Fork for Multispectral & Change-Detection

This repository is a fork of the [Ultralytics](https://github.com/ultralytics/ultralytics) package, extended to support:

- **Multispectral & Remote-Sensing Inputs**  
  – Full compatibility with images having arbitrary channel counts (e.g. RGB+NIR, multi-band satellite)
    - supporting tiff files with n bands as a new file type
    – Missing input-channel weights are initialized via Xavier uniform when loading a pretrained checkpoint  


- **Siamese / Dual-Stream Architectures**  
  – Built-in support for Siamese YOLO models geared toward bi-temporal change detection  
  – Example config included for a Siamese YOLOv9e; additional dual-stream configs coming soon  

- **Tested Models & Tasks**  
  – **Classification & Detection** on multispectral data: YOLOv8, YOLOv9, and YOLOv10  
  – Other tasks (segmentation, pose) may work but are currently untested or brokene
  - **Siamese** object level change detection for YOLOv9e. 

- **Usage Modes**  
  – Train from scratch on your own multispectral dataset  
  – Fine-tune from any Ultralytics pretrained checkpoint, automatically adapting input-layer dimensions  

> **Warning:** This fork is under active development and subject to breaking changes.  
> To avoid unexpected issues, pin your project to a specific release tag or commit hash.

---

## What Is YOLO?

**You Only Look Once (YOLO)** is a family of single-stage object detectors that perform localization and classification in one forward pass through a convolutional network.  
- The input image is divided into a grid; each cell predicts a set of bounding boxes and class probabilities simultaneously.  
- Modern YOLO versions (v8, v9, v10) use advanced backbones, feature-pyramid necks, and optimized heads to balance speed (dozens of FPS) with high accuracy.  
- Ultralytics’ implementation provides an easy CLI/API, out-of-the-box pretrained weights, and a consistent hyperparameter configuration.

---

## What Are Siamese Networks?

A **Siamese network** consists of two identical subnetworks with shared weights, processing two inputs in parallel. The key idea is to learn feature embeddings that highlight similarities or differences between the inputs, making Siamese architectures well suited for tasks such as:

- **Verification & Matching**: e.g., face or signature verification  
- **Change Detection**: comparing bi-temporal imagery to isolate new or altered features  

In this fork, we combine a Siamese backbone with YOLO’s detection head to focus on _change cues_ between a reference image (e.g., a base map) and a query image (e.g., a newly annotated or temporally updated scene).

---

## Simplified Comparison: YOLO vs. Siamese YOLO

![siamese_schema.png](siamese_schema.png)

| Feature                | Standard YOLO            | Siamese YOLO                       |
|------------------------|--------------------------|------------------------------------|
| Input                  | Single image             | Pair of images (reference + query) |
| Feature extractor      | One backbone             | Two synchronized backbones        |
| Fusion mechanism       | —                        | Difference or attention fusion     |
| Detection focus        | All objects              | Changes between inputs             |
| Use case               | Object detection         | Change detection & comparison      |

---

## Quick Start

1. **Install**  
   ```bash
   pip install git+https://github.com/your-org/ultralytics-multispectral.git
   ```
## configuration

### mulitlayer 

### 3. Configuring the Model for 6-Channel Input

**A) Fine-tune from pretrained RGB weights**  
```python
from ultralytics_MB import YOLO

# 1. Load a pretrained YOLOv8n (.pt) checkpoint
model = YOLO('yolov8n.pt')

# 2. Tell it to expect 6 input channels instead of 3
#    and only adjust the first conv layer’s weights
model.train(
    data='data.yaml',
    channels=6,           # input now has 6 bands (RGB@t1 + RGB@t2)
    adjust_layers=[0],    # remaps layer 0 weights from 3→6 channels
    bands = [0,1,2,3,4,5], # defines which bands should be loaded 

    # … other args (imgsz, epochs, batch, etc.)
)
```
- **`channels=6`** re-builds the very first conv kernel to accept 6 bands.  
- **`adjust_layers=[0]`** loads the RGB weights into both halves of that new 6-channel kernel, so you retain all pretrained features for timestamp 1 & 2.
- ** bands = [0,1,2,3,4,5]** defines which bands should be loaded 

---

**B) Train from scratch on 6 channels**  
```python
from ultralytics_MB import YOLO

# 1. Load an untrained YOLOv8n architecture (.yaml)
model = YOLO('custom_colo.yaml')

# 2. Directly train with 6 inputs—no weight remapping needed
model.train(
    data='data.yaml',
    channels=6,  
    bands = [0,1,2,3,4,5]
)
```
- Using the `.yaml` spec builds your network “from scratch” with the requested number of channels baked in.

### 3. Configuring the Siamese YOLOv9e-s (ES) for 6-Channel Dual-Stream Input

```python
from ultralytics_MB import YOLO

# 1. Load the Siamese-enabled “ES” spec — NOT the standard yolov9e.yaml!
model = YOLO('yolov9es.yaml', task='detect')

# 2. Tell it you have 6 input bands (RGB@t1 + RGB@t2)
#    and enable the Siamese dual-stream head
model.train(
    data='data.yaml',
    channels=6,         # network expects 6-band inputs
    bands=[0,1,2,3,4,5], # indexes of the band that should be loaded, first 
    dual_stream=True,   # turn on Siamese twin-stream processing
    # … other hyperparams go below
)
```

- **`yolov9es.yaml`** is the special “ES” version with the extra dual-stream modules baked in.  
- **`channels=6`** defines your two RGB timestamps as six input planes.  
- **`bands`** = [0,1,2,3,4,5] defines which bands should be loaded, first half corresponds to image one and the second to image 2 
- **`dual_stream=True`** wires the Siamese twin backbones and fusion head for change detection.

- **`imgsz`**, **`epochs`**, **`batch`**, **`single_cls`** are your usual YOLO training flags.  
- **`channels`** and **`dual_stream`** are the only extras to spin up the Siamese network on multi-band imagery.  


### `FUSION_METHOD`

The `FeatureFusionBlock` picks its fusion strategy from the `FUSION_METHOD` environment variable. If you don’t set it, it defaults to **`diff`** (element-wise subtraction).

#### Available modes

- **`add`**  
  Element-wise addition:  
  ```python
  out = x1 + x2
  ```

- **`diff`**  
  Element-wise difference:  
  ```python
  out = x1 - x2
  ```

- **`multiply`**  
  Element-wise multiplication:  
  ```python
  out = x1 * x2
  ```

- **`weighted`**  
  Learnable weighted sum. You must pass two floats via `params=[w1, w2]` when you construct the block:  
  ```python
  block = FeatureFusionBlock(params=[0.7, 0.3], in_channels=128)
  # out = 0.7*x1 + 0.3*x2
  ```

- **`attention`**  
  Squeeze-and-Excitation channel attention on (x1 + x2).

- **`cross_attention`**  
  QKV-style cross-attention between x1 (queries) and x2 (keys, values).

- **`cbam`**  
  CBAM (Channel + Spatial) attention on (x1 + x2).

- **`cbam_cross_attention`**  
  CBAM followed by QKV cross-attention.

#### How to set

Before creating your fusion block, export the choice in your shell:

```bash
export FUSION_METHOD=attention
```


# Apretiation
We gratefully acknowledge the support of the Klaus Tschira Stiftung
(KTS) and the German Foreign Office, whose funding made this
research possible. We also thank the German Red Cross (GRC) for
their collaboration and practical insights throughout the project.
Special thanks go to Dr. Carolin Klonner for guiding first prototypes
and inspiring this work. We thank our colleagues and student assis-
tants at HeiGIT for their crucial contributions to dataset creation,
annotation, and technical implementation, which were instrumen-
tal to the success of this work. We thank Heidelberg University’s
Computing Centre for Accesses to the SDS@hd hot-data storage.
Support by the state of Baden-Württemberg through bwHPC (Helix
Cluster) and the German Research Foundation (DFG) through grant
INST 35/1597-1 FUGG is greatfully aknowledged.
This Work is based on the work of the ultralytics team, which layed the foundation.

