---
library_name: peft
license: other
base_model: .model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master
tags:
- base_model:adapter:.model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: persona_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# persona_lora

This model is a fine-tuned version of [.model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master](https://huggingface.co/.model/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master) on the persona_toy dataset.

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
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 2.0

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2