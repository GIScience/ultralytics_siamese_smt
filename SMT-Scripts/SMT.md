# Run Training Scripts

This directory contains scripts to train and evaluate models using our custom **Sketch Map Tool (SMT)** datasets.

---

## `train.py` — Model training entrypoint

`train.py` is a thin training wrapper around an Ultralytics-based YOLO model (imported as `ultralytics_MB.YOLO`).  
It is configured **entirely via environment variables**, making it easy to run locally or on HPC/Slurm without editing code.

**What it does**
- Reads training/model configuration from environment variables (e.g., `MODEL_PATH`, `EPOCHS`, `BATCH_SIZE`, …)
- Prints all resolved parameters for reproducibility
- Selects an output/logging backend (**local** vs. **Neptune**)
- Instantiates the model from a YAML definition
- Starts training via `model.train(...)`

---

## `run.sh` — SLURM submission script

`run.sh` is a SLURM batch script to run `train.py` on a computing cluster.

**What it does**
- Requests a **single GPU** job (e.g., A100) and sets runtime/memory/CPU resources
- Configures training through the same environment variables used by `train.py`
- Loads CUDA/conda modules
- Creates a fresh conda environment (as written) and installs:
  - PyTorch + CUDA dependencies
  - the local `ultralytics_multiband_support` fork
  - additional Python packages required for training/logging
- Launches training with:
  - `python ./train.py`

---

## `analyse_model.ipynb` — Model analysis notebook

`analyse_model.ipynb` contains code to analyze trained models and visualize results **quantitatively and qualitatively**.  
It produces tables and figures that complement the standard Ultralytics YOLO outputs.

### Included analyses

**Dataset analysis**
- Counts of files, instances, and class distributions (synthetic + real-world)
- Analysis of failed clipping cases and filtering steps

**Model performance visualization**
- Detection error analysis (False Positives / False Negatives)
- Visualization of predictions vs. ground truth
- Training metrics plots (loss, learning rate, metrics over epochs)

**Data visualization**
- Sample visualizations from synthetic datasets (ESRI & OSM)
- Sample visualizations from real-world datasets
- Class Activation Map (CAM) plots showing model feature activation
