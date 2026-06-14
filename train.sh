#!/bin/bash

echo "Git cloning the repository..."

git clone https://github.com/Mithun-033/SharpGPT.git

echo "Repository cloned successfully."

#------------------------------------------------------------------------------------#

echo "Downloading required torch version..."

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "PyTorch version: $TORCH_VERSION"

CUDA_VERSION=$(python -c "import torch; print("cu"+str(int(float(torch.version.cuda)*10)))")
echo "CUDA version: $CUDA_VERSION"

echo "Uninstalling existing torch packages..."
pip uninstall -y torch torchvision torchaudio 

echo "Installing torch packages with specific CUDA version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_VERSION
echo "Torch packages installed successfully."

#------------------------------------------------------------------------------------#

pip install datasets tokenizers torchinfo
echo "All dependencies installed successfully."

#------------------------------------------------------------------------------------#

cd SharpGPT
# fill the lightning cp to download pre train data / download data to Pre_train_data dir path

#--------------------------------------------------------------------------------#

echo "Starting the training process..."
python Model_dir/train.py



