# Unsloth

> Easily finetune & train LLMs, get faster with [unsloth](https://unsloth.ai).

Models with  `-unsloth` suffix are [dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) quantized models, which are only slightly larger than non suffix models, but much better.

## Index

- [Fine-tuning Guide](#fine-tuning-guide)
- [Datasets 101](#datasets-101)


## Fine-tuning Guide

By fine-tuning a pre-trained model on a specialized dataset, you can:
- **Update Knowledge**: Introduce new domain-specific information.
- **Customize Behavior**: Adjust the model’s tone, personality, or response style.
- **Optimize for Tasks**: Improve accuracy and relevance for specific use cases.

Example usecases:
- Train LLM to predict if a headline impacts a company positively or negatively.
- Use historical customer interactions for more accurate and custom responses.
- Fine-tune LLM on legal texts for contract analysis, case law research, and compliance.

You can think of a fine-tuned model as a specialized agent designed to do specific tasks more effectively and efficiently.

> [FAQ + Is Fine-tuning Right For Me?](https://docs.unsloth.ai/get-started/beginner-start-here/faq-+-is-fine-tuning-right-for-me)

### Choose the Right Model + Method

It is best to start with a small instruct model like Llama 3.1 (8B) and experiment from there. You'll also need to decide between QLoRA and LoRA training:

- **LoRA**: Fine-tunes small, trainable matrices in 16-bit without updating all model weights.  
- **QLoRA**: Combines LoRA with 4-bit quantization to handle very large models with minimal resources. 

You can change the model name to **whichever model you like** by matching it with model's name on Hugging Face e.g. `'unsloth/llama-3.1-8b-bnb-4bit'`.

There are 3 other settings which you can toggle:
- `max_seq_length = 2048` – Controls context length. While Llama-3 supports 8192, we recommend 2048 for testing. Unsloth enables 4× longer context fine-tuning.
- `dtype = None` – Defaults to None; use torch.float16 or torch.bfloat16 for newer GPUs.
- `load_in_4bit = True` – Enables 4-bit quantization, reducing memory use 4× for fine-tuning on 16GB GPUs. Disabling it on larger GPUs (e.g., H100) slightly improves accuracy (1–2%).

**We recommend starting with QLoRA**, as it is one of the most accessible and effective methods for training models. Our [dynamic 4-bit](https://unsloth.ai/blog/phi4) quants, the accuracy loss for QLoRA compared to LoRA is now largely recovered.

### Understand Model Parameters

The goal is to change these numbers to increase accuracy, but also counteract over-fitting. Over-fitting is when you make the language model memorize a dataset, and not be able to answer novel new questions. We want to a final model to answer unseen questions, and not do memorization. Here are the key parameters:

- **Learning Rate**: Defines how much the model’s weights adjust per training step. Higher Learning Rates: Faster training, risk of overfitting. Lower Learning Rates: More stable training, may require more epochs.
Typical Range: 1e-4 (0.0001) to 5e-5 (0.00005).
- **Epochs**: Number of times the model sees the full training dataset. Recommended: 1-3 epochs (anything more than 3 is generally not optimal unless you want your model to have much less hallucinations but also less creativity). More Epochs: Better learning, higher risk of overfitting. Fewer Epochs: May undertrain the model.

For more additional parameters check [LoRA Parameters Encyclopedia](https://docs.unsloth.ai/get-started/beginner-start-here/lora-parameters-encyclopedia#lora-configuration-parameters).

To avoid overfitting & underfitting:
- **Overfitting (Too Specialized)**: The model memorizes training data, failing to generalize to unseen inputs. Solution: Reduce learning rate. Lower the number of training epochs. Increase dropout rate to introduce regularization.
- **Underfitting (Too Generic)**: The model fails to learn from training data, providing responses similar to the base model. Solution: Increase learning rate. Train for more epochs. Use a more domain-relevant dataset.

> Fine-tuning has no single "best" approach, only best practices. **Experimentation is key** to finding what works for your needs.


### Training

Your job is to set parameters to make training loss go to as close to 0.5 as possible! If your finetune is not reaching 1, 0.8 or 0.5, you might have to adjust some numbers. If your loss goes to 0, that's probably not a good sign as well!

We generally recommend keeping the default settings unless you need longer training or larger batch sizes.
- `per_device_train_batch_size = 2` – Increase for better GPU utilization but beware of slower training due to padding. Instead, increase gradient_accumulation_steps for smoother training.
- `gradient_accumulation_steps = 4` – Simulates a larger batch size without increasing memory usage.
- `max_steps = 60` – Speeds up training. For full runs, replace with num_train_epochs = 1 (1–3 epochs recommended to avoid overfitting).
- `learning_rate = 2e-4` – Lower for slower but more precise fine-tuning. Try values like 1e-4, 5e-5, or 2e-5.


## Datasets 101

> [Learn](https://docs.unsloth.ai/basics/datasets-101) all the essentials of creating a dataset for fine-tuning!

For LLMs, datasets are collections of data that can be used to train our models. In order to be useful for training, text data needs to be **in a format that can be tokenized**.

To enable the process of tokenization, datasets need to be in a format that can be read by a tokenizer.

|    Format    | Description                                                                                                        |         Training Type        |
|:------------:|--------------------------------------------------------------------------------------------------------------------|:----------------------------:|
| Raw Corpus   | Raw text from a source such as a website, book, or article.                                                        | Continued Pretraining        |
| Instruct     | Instructions for the model to follow and an example of the output to aim for.                                      | Supervised fine-tuning (SFT) |
| Conversation | Multiple-turn conversation between a user and an AI assistant.                                                     | Supervised fine-tuning (SFT) |
| RLHF         | Conversation between a user and an AI assistant, with the assistant's responses being ranked by a human evaluator. | Reinforcement Learning       |

> It's worth noting that different styles of format exist for each of these types. 

We want to identify the following: 
1. **Purpose of dataset**: Knowing the purpose of the dataset will help us determine **what data we need and format** to use.
The purpose could be, adapting a model to a new task such as summarization or improving a model's ability to role-play a specific character. 
2. **Style of output**: The style of output will let us know **what sources of data we will use** to reach our desired output.
For example, the type of output you want to achieve could be JSON, HTML, text or code. Or perhaps you want it to be Spanish, English or German etc. 
3. **Data source**: When we know the purpose and style of the data we need, we can look for a **data source to collect** our data from.
The Source of data can be a CSV file, PDF or even a website. You can also synthetically generate data but extra care is required to make sure each example is **high quality and relevant**. 

If you have multiple datasets for fine-tuning, you can either:
- Standardize the format of all datasets, combine them into a single dataset, and fine-tune on this unified dataset.
- Use the [Multiple Datasets notebook](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing) to fine-tune on multiple datasets directly.

You can fine-tune an already fine-tuned model multiple times, but it's best to combine all the datasets and perform the fine-tuning in a single process instead. Training an already fine-tuned model can potentially alter the quality and knowledge acquired during the previous fine-tuning process.