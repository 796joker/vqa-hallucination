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
- name: sft_data25k
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_data25k

This model is a fine-tuned version of [../downloads/models/Qwen3-VL-8B-Instruct](https://huggingface.co/../downloads/models/Qwen3-VL-8B-Instruct) on the llava_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8719

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
| 0.898         | 0.3265 | 500  | 0.8908          |
| 0.9106        | 0.6531 | 1000 | 0.8818          |
| 0.9007        | 0.9796 | 1500 | 0.8766          |
| 0.8417        | 1.3056 | 2000 | 0.8746          |
| 0.8417        | 1.6322 | 2500 | 0.8725          |
| 0.8579        | 1.9587 | 3000 | 0.8718          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2