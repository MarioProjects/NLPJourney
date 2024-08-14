# Masked Language Modeling

Masked Language modeling is the task of masking some tokens in a sentence and having the model predict the masked tokens.
We can use any **plain text dataset** with a language model tokenizer.

# Index

- [Key Techniques](#key-techniques)
- [Common Tasks](#common-tasks)
- [Example Projects](#example-projects)
- [Evaluation Metrics](#evaluation-metrics)
- [Projects](#projects)
  - [Domain Adaptation](#domain-adaptation)

## Key Techniques

- BERT (Bidirectional Encoder Representations from Transformers)
- RoBERTa (Robustly Optimized BERT Approach)
- ALBERT (A Lite BERT)

## Common Tasks

- **Model Adaptation**: By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once! This process of fine-tuning a pretrained language model on in-domain data is usually called *domain adaptation*.

## Example Projects

| NA

## Evaluation Metrics

Some potential metrics could be:

- Perplexity: The perplexity of a language model on a test set is the exponentiated average negative log-likelihood of the test set, normalized by the number of words.
- Accuracy: The percentage of correctly predicted tokens.

## Projects

### Domain Adaptation

There are a few cases where you’ll want to first fine-tune the language models on your data, **before training a task-specific head**. For example, if your dataset contains legal contracts or scientific articles, a vanilla Transformer model like BERT will typically treat the domain-specific words in your corpus as rare tokens, and the resulting performance may be less than satisfactory. By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once!

This process of fine-tuning a pretrained language model on in-domain data is usually called ***domain adaptation. ***

**For both auto-regressive and masked language modeling**, a common preprocessing step is to **concatenate all the examples** and then **split the whole corpus into chunks of equal size**. This is quite different from our usual approach, where we simply tokenize individual examples. Why concatenate everything together? The reason is that individual examples might get truncated if they’re too long, and that would result in losing information that might be useful for the language modeling task!

> Note that **using a small chunk size can be detrimental** in real-world scenarios, so you should use a size that corresponds to the use case you will apply your model to.

In masked language modeling the objective is to predict randomly masked tokens in the input batch, and by creating a labels column we provide the ground truth for our language model to learn from. So initially, the labels must be set as a a copy of the `input_ids`.


Pending issues:
- [ ] Add Next Sentence Prediction (NSP) task - Maybe if we are dealing with reviews it has not much sense to predict the next sentence, but it could be useful for other tasks.