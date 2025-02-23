# Unsloth

> Easily finetune & train LLMs, get faster with [unsloth](https://unsloth.ai).

Models with  `-unsloth` suffix are [dynamic 4-bit](https://unsloth.ai/blog/dynamic-4bit) quantized models, which are only slightly larger than non suffix models, but much better.


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