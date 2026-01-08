#!/bin/sh
#SBATCH --job-name smt-train                                          
#SBATCH --time=25:00:00                                            
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=50gb                                                 
#SBATCH --cpus-per-task=8
#SBATCH -o output_train_diff.out                                     
#SBATCH --mail-user your_mail@here.com
#SBATCH --mail-type ALL
# % ALL will alert you of job beginning, completion, failure etc


export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
#export LD_LIBRARY_PATH= 
#export XLA_FLAGS='--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.6.1/'
export NEPTUNE_API_TOKEN="eyJhcGlfYW...jgzNWEifQ==" #should look like this
export NEPTUNE_PROJECT="Org/project"
export NEPTUNE_MODEL="SMT-OSM"
export MODEL_PATH="yolov9es.yaml"
export DATASET_PATH="Path/to/ds"
export EPOCHS=200
export BATCH_SIZE=8
export IMGSZ=1024
export CHANNELS=6
export ADJUST_LAYERS=""
export DUAL_STREAM=True
export BANDS=[0,1,2,3,4,5]
export FUSION_METHOD="diff"
export STORE="local"
export STORE_PATH="path/to/ds" #adjust path to ur destination

module load devel/miniconda/3
#module load lib/cudnn/8.5.0-cuda-11.6
module load devel/cuda

source $MINICONDA_HOME/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

conda remove -n smt-train --all -y 
conda create -n smt-train -y

conda activate smt-train
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
cd ultralytics_multiband_support
pip install -e .
cd ..
pip install neptune tifffile imagecodecs
pip install albumentations==1.0.3
pip install --upgrade torch torchvision
python ./train.py
