# Supervised Fine-Tuning

## Index

- [1️⃣ Chat Templates](#1️⃣-chat-templates)
- [2️⃣ Supervised Fine-Tuning](#2️⃣-supervised-fine-tuning)
- [3️⃣ Low Rank Adaptation (LoRA)](#3️⃣-low-rank-adaptation-lora)
- [4️⃣ Evaluation](#4️⃣-evaluation)
- [References](#references)

## 1️⃣ Chat Templates

Chat templates **structure interactions** between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.


## 2️⃣ Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is a critical process for **adapting pre-trained language models to specific tasks**. It involves training the model on a task-specific dataset with labeled examples. For a detailed guide on SFT, including key steps and best practices, see [The supervised fine-tuning section of the TRL documentation](https://huggingface.co/docs/trl/en/sft_trainer).


## 3️⃣ Low Rank Adaptation (LoRA)

Low Rank Adaptation (LoRA) is a technique for fine-tuning language models by adding low-rank matrices to the model’s layers. This allows for **efficient fine-tuning** while preserving the model’s pre-trained knowledge. One of the key benefits of LoRA is the significant memory savings it offers, making it possible to fine-tune large models on hardware with limited resources.


## 4️⃣ Evaluation

Evaluation is a crucial step in the fine-tuning process. It allows us to **measure the performance** of the model on a task-specific dataset.


## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [SFTTrainer in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://github.com/huggingface/alignment-handbook)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)