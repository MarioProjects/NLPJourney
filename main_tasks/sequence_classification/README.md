# Sequence Classification

Token classification is a generic task encompasses any problem that can be formulated as “attributing a label to each token in a sentence,”.

# Index

- [Key Techniques](#key-techniques)
- [Common Tasks](#common-tasks)
- [Example Projects](#example-projects)
- [Evaluation Metrics](#evaluation-metrics)
- [Projects](#projects)
  - [MRPC - GLUE](#mrpc---glue)


## Key Techniques

The key techniques we are going to use are **Transformer-based models**: Pre-trained models like BERT, RoBERTa, and their variants have become state-of-the-art for many sequence classification tasks.

## Common Tasks

- **Entailment**: Determining whether a sentence entails, contradicts, or is neutral with respect to another sentence.
- **Classification**: Assigning a category or label to a sequence of tokens. For example for document classification, or spam detection.
- **Sentiment Analysis**: Determining the sentiment (positive, negative, neutral) of a sentence or document.

## Example Projects

| NA
 
## Evaluation Metrics

The common evaluation metrics for token classification tasks are:

- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall.
- **Accuracy**: The proportion of correct predictions among all predictions.

## Projects

### MRPC - GLUE

The Microsoft Research Paraphrase Corpus (MRPC) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

Fine-tune a model (BERT) on a Sequence Classification task, which will then be able to compute predictions.
