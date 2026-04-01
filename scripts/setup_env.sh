#!/bin/bash
# Environment setup for VQA Hallucination Reduction project
# Run on A100 server

set -e

echo "=== Step 1: Install LLaMA-Factory ==="
cd ~
if [ ! -d "LlamaFactory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
fi
cd LlamaFactory
pip install -e ".[torch,metrics]"

echo "=== Step 2: Install additional dependencies ==="
pip install gradio

echo "=== Step 3: Verify installation ==="
llamafactory-cli version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
python -c "from transformers import AutoProcessor; print('transformers OK')"
python -c "from peft import PeftModel; print('peft OK')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit scripts/config.sh to set MODEL_PATH and COCO paths"
echo "  2. Download model and data: see DOWNLOAD_GUIDE.md"
echo "  3. Run: bash scripts/download_data.sh"
