# Causal Language Modeling

Causal language modeling is the task of predicting the next token in a sentence. This is the task that most language models are trained on. We can use it for pretraining a model from scratch or, adapting a pretrained model to a new domain, or to learn a new behavior such as following instructions.

We can use any **plain text dataset** with a language model tokenizer.

# Index

- [Key Techniques](#key-techniques)
- [Common Tasks](#common-tasks)
- [Example Projects](#example-projects)
- [Evaluation Metrics](#evaluation-metrics)
- [Projects](#projects)
  - [Causal Language Modeling from Scratch](#causal-language-modeling-from-scratch)

## Key Techniques

The key techniques for causal language modeling are:
- Decoder-only architectures: Causal language modeling is usually done with decoder-only architectures, which means that the model only has access to the tokens that come before the token it is trying to predict. This is in contrast to encoder-decoder architectures, which have access to both the tokens that come before and after the token they are trying to predict.
- LoRA & PEFT: LoRA is widely used when fine-tuning a language model on in-domain data. Also other more recent Parameter Efficient Fine-Tuning (PEFT) techniques can be used to fine-tune a language model on in-domain data.
- [Unsloth](https://github.com/unslothai/unsloth): UnslothAI is a parameter-efficient fine-tuning library for LLMs that accelerates fine-tuning by 2-5 times while using 70% less memory.
- Ollama: Ollama is a library that provides a simple library that provides LLMs models.

## Common Tasks

- **Model Adaptation**: By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once! This process of fine-tuning a pretrained language model on in-domain data is usually called *domain adaptation*.
- **Instruction Following**: You can train a language model to follow instructions by training it on a dataset of instructions and then fine-tuning it on a dataset of instructions with a different style or domain.

## Example Projects

- **Python Completion**: You can train a language model to predict the next token in a Python code snippet, which can be useful for code completion.
- **Instruction Following**: You can train a language model to follow instructions by training it on a dataset of instructions and then fine-tuning it on a dataset of instructions with a different style or domain. You can also add guardrails to the model to ensure that it follows the instructions correctly and doesn't generate unsafe or incorrect output.

## Evaluation Metrics

Some potential metrics could be:

- Perplexity: The perplexity of a language model on a test set is the exponentiated average negative log-likelihood of the test set, normalized by the number of words.

## Projects

### Causal Language Modeling from Scratch

In this example project we see how to load a big dataset for coding. We load a dataset based on GitHub repositories and filter it in the fly to only keep Python *data science* files.

> ⚠️ **Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels. Specifically, the shift happens in the loss function. You can check in the Pytorch code how we implement a custom loss function and the label shifting is handled there.**