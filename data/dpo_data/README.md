---
dataset_info:
  features:
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: chosen
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: rejected
    struct:
    - name: from
      dtype: string
    - name: value
      dtype: string
  - name: images
    list:
      dtype: image
license: cc-by-nc-4.0
task_categories:
- text-generation
- question-answering
- visual-question-answering
language:
- en
tags:
- llama-factory
size_categories:
- 1K<n<10K
---

Borrowed from: https://huggingface.co/datasets/openbmb/RLHF-V-Dataset

You can use it in [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) by specifying `dataset: rlhf_v`.
