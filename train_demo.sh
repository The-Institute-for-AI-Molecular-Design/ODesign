#!/bin/bash

#######################################################################
# ODesign Training Demo Script
# ---------------------------------------------------------------------
# Modify the arguments below, then simply run:
#     bash train_demo.sh
#######################################################################

# 1. Choose training model name (REQUIRED)
#     Options:
#        - odesign_base_prot_flex
#        - odesign_base_prot_rigid
#        - odesign_base_ligand_rigid
#        - odesign_base_na_rigid
model_name="odesign_base_prot_flex"

# 2. Data root directory (default: ./data)
data_root_dir="./data"

# 3. Checkpoint root directory (default: ./ckpt)
ckpt_root_dir="./ckpt"

# 4. Custom experiment name for this training run (optional)
#     If empty -> auto generate: train_${model_name}
exp_name=""

# If exp_name is empty, create default experiment name
if [[ -z "$exp_name" ]]; then
    exp_name="train_${model_name}"
fi

# 5. CUDA Device Setup (default: 0)
export CUDA_VISIBLE_DEVICES=0



#######################################################################
# Config Summary
#######################################################################

echo "-----------------------------------------------------------"
echo "🚀 Start ODesign Training"
echo "-----------------------------------------------------------"
echo "Model                   : $model_name"
echo "Experiment Name         : $exp_name"
echo "Data Root Dir           : $data_root_dir"
echo "Checkpoint Root Dir     : $ckpt_root_dir"
echo "CUDA_VISIBLE_DEVICES    : $CUDA_VISIBLE_DEVICES"
echo "-----------------------------------------------------------"
echo ""



#######################################################################
# Single-GPU Training Command
# ---------------------------------------------------------------------
# You don't need to modify the following command
#######################################################################

# Launch training
python ./scripts/train.py \
    exp="train_${model_name}" \
    data_root_dir="$data_root_dir" \
    ckpt_root_dir="$ckpt_root_dir" \
    exp.exp_name="$exp_name"



#######################################################################
# 🔥 Optional: Distributed Multi-GPU Training Example
# ---------------------------------------------------------------------
# Uncomment and modify the following block if you want to train
# with multiple GPUs using torchrun.
#######################################################################

# NPROC=2             # Number of GPUs used
# NODE_RANK=0         # 0 for single-machine multi-GPU
# NNODES=1            # Number of nodes (keep 1 if only one machine)

# # Auto-generate random port
# MASTER_PORT=$((10000 + RANDOM % 50000))
# MASTER_ADDR=$(hostname -I | awk '{print $1}')

# # Launch training
# torchrun \
#     --nproc_per_node=$NPROC \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     ./scripts/train.py \
#         exp="train_${model_name}" \
#         data_root_dir="$data_root_dir" \
#         ckpt_root_dir="$ckpt_root_dir" \
#         exp.exp_name="$exp_name"

