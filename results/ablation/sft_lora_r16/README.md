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
- name: sft_lora_r16
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_lora_r16

This model is a fine-tuned version of [../downloads/models/Qwen3-VL-8B-Instruct](https://huggingface.co/../downloads/models/Qwen3-VL-8B-Instruct) on the llava_sft dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8541

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
| 0.9129        | 0.1633 | 500  | 0.8876          |
| 0.8877        | 0.3265 | 1000 | 0.8771          |
| 0.8657        | 0.4898 | 1500 | 0.8712          |
| 0.8683        | 0.6531 | 2000 | 0.8663          |
| 0.8843        | 0.8163 | 2500 | 0.8636          |
| 0.8541        | 0.9796 | 3000 | 0.8597          |
| 0.8439        | 1.1427 | 3500 | 0.8589          |
| 0.8383        | 1.3060 | 4000 | 0.8579          |
| 0.8167        | 1.4692 | 4500 | 0.8564          |
| 0.8244        | 1.6325 | 5000 | 0.8548          |
| 0.8292        | 1.7958 | 5500 | 0.8542          |
| 0.854         | 1.9590 | 6000 | 0.8540          |


### Framework versions

- PEFT 0.18.1
- Transformers 4.57.6
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2