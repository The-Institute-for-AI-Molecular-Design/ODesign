#!/bin/bash

#######################################################################
# ODesign Checkpoint Download Script
# ---------------------------------------------------------------------
# This script downloads all available model checkpoints for ODesign.
#
# Usage:
#     bash get_odesign_ckpt.sh [ckpt_root_dir]
#
# Parameter:
#     1. ckpt_root_dir: User-specified pre-trained ckpt root directory
#                       (default: ./ckpt)
#######################################################################

# Set default argument
DEFAULT_CKPT_ROOT_DIR="./ckpt"

# Parse command line argument
ckpt_root_dir="${1:-$DEFAULT_CKPT_ROOT_DIR}"

# Available checkpoints
CKPTS=(
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/ab.pt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_prot_rigid.pt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_ligand_rigid.pt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign/resolve/main/ckpt/odesign_base_na_rigid.pt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_protein.ckpt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_ligand.ckpt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_dna.ckpt?download=true"
    # "https://huggingface.co/The-Institute-for-AI-Molecular-Design/OInvFold/resolve/main/oinvfold_rna.ckpt?download=true"
    "https://huggingface.co/The-Institute-for-AI-Molecular-Design/ODesign-AB/resolve/main/ab.pt?download=true"
)


echo "-----------------------------------------------------------"
echo "🚀 Start ODesign Checkpoint Download"
echo "-----------------------------------------------------------"
echo "Checkpoint Root Directory: $ckpt_root_dir"
echo "-----------------------------------------------------------"
echo ""

# Create checkpoint directory if it doesn't exist
mkdir -p "$ckpt_root_dir"

# Function to download file with error handling
download_file() {
    local file_name=$1
    local file_url=$2
    local output_path="$ckpt_root_dir/$file_name"
    
    echo "📥 Downloading: $file_name"
    echo "   From: $file_url"
    echo "   To: $output_path"
    
    # Check if file already exists
    if [[ -f "$output_path" ]]; then
        echo "   ⚠️  File already exists, skipping download."
        return 0
    fi
    
    # Download file using wget or curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output_path" "$file_url"
    elif command -v curl &> /dev/null; then
        curl -L -# -o "$output_path" "$file_url"
    else
        echo "   ❌ Error: Neither wget nor curl is available. Please install one of them."
        return 1
    fi
    
    # Check if download was successful
    if [[ $? -eq 0 ]] && [[ -f "$output_path" ]]; then
        echo "   ✅ Successfully downloaded: $file_name"
        return 0
    else
        echo "   ❌ Failed to download: $file_name"
        return 1
    fi
}

for file_url in "${CKPTS[@]}"; do
    file_name=$(basename "$file_url" | sed 's/?download=true//')
    download_file "$file_name" "$file_url"
    echo ""
done

echo "-----------------------------------------------------------"
echo "🎉 SUCCESS: All available ckpts have been downloaded!"
echo "-----------------------------------------------------------"
