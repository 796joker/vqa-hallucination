---
library_name: peft
license: other
base_model: ../downloads/models/Qwen3-VL-8B-Instruct
tags:
- base_model:adapter:../downloads/models/Qwen3-VL-8B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sft_lora_r32
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_lora_r32

This model is a fine-tuned version of [../downloads/models/Qwen3-VL-8B-Instruct](https://huggingface.co/../downloads/models/Qwen3-VL-8B-Instruct) on the llava_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8516

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.91          | 0.1633 | 500  | 0.8853          |
| 0.8861        | 0.3265 | 1000 | 0.8757          |
| 0.8636        | 0.4898 | 1500 | 0.8697          |
| 0.8658        | 0.6531 | 2000 | 0.8646          |
| 0.8815        | 0.8163 | 2500 | 0.8617          |
| 0.8503        | 0.9796 | 3000 | 0.8572          |
| 0.8283        | 1.1427 | 3500 | 0.8573          |
| 0.8217        | 1.3060 | 4000 | 0.8562          |
| 0.7981        | 1.4692 | 4500 | 0.8546          |
| 0.8048        | 1.6325 | 5000 | 0.8525          |
| 0.8099        | 1.7958 | 5500 | 0.8519          |
| 0.8357        | 1.9590 | 6000 | 0.8517          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2