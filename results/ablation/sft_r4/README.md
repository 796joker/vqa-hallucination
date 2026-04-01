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
- name: sft_r4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_r4

This model is a fine-tuned version of [../downloads/models/Qwen3-VL-8B-Instruct](https://huggingface.co/../downloads/models/Qwen3-VL-8B-Instruct) on the llava_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8615

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
| 0.9182        | 0.1633 | 500  | 0.8932          |
| 0.8924        | 0.3265 | 1000 | 0.8817          |
| 0.8709        | 0.4898 | 1500 | 0.8760          |
| 0.8729        | 0.6531 | 2000 | 0.8717          |
| 0.8909        | 0.8163 | 2500 | 0.8694          |
| 0.8614        | 0.9796 | 3000 | 0.8666          |
| 0.8646        | 1.1427 | 3500 | 0.8654          |
| 0.8601        | 1.3060 | 4000 | 0.8642          |
| 0.8406        | 1.4692 | 4500 | 0.8630          |
| 0.8475        | 1.6325 | 5000 | 0.8620          |
| 0.8539        | 1.7958 | 5500 | 0.8616          |
| 0.8774        | 1.9590 | 6000 | 0.8614          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2