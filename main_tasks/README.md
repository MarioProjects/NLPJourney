# Main NLP Taks

Here you can find at each subfolder a specific NLP task with some projects and examples.
Each section can be taken independently and usually will show you how to train or use a model with the Hugging Face `Trainer API` or with your own training loop, using Hugging Face Accelerate,for example. Feel free to skip either part and focus on the one that interests you the most: the Trainer API is great for fine-tuning or training your model without worrying about what’s going on behind the scenes, while the training loop with Accelerate will let you customize any part you want more easily.

## Index

- [Masked Language Modeling](masked_language_modeling): Masked language modeling is the task of masking some tokens in a sentence and having the model predict the masked tokens. Masked language modeling fine-tuning on in-domain data you can help boost the performance of many downstream tasks (such as the others described in this index), which means you usually only have to do this step once! This process of fine-tuning a pretrained language model on in-domain data is usually called *domain adaptation*.
- [Sequence Classification](sequence_classification): Sequence classification is a generic task that encompasses any problem that can be formulated as classifying a sequence of tokens into a given number of classes.
- [Token Classification](token_classification): Token classification is a generic task that encompasses any problem that can be formulated as “attributing a label to each token in a sentence.”
- [Causal Language Modeling](causal_language_modeling): Causal language modeling is the task of predicting the next token in a sentence. This is the task that most language models are trained on. We can use it for pretraining a model from scratch or, adapting a pretrained model to a new domain, or to learn a new behavior such as following instructions.
- [Semantic Search](semantic_search): Semantic search is a search technique that takes the meaning of words into account rather than just the presence of the words themselves. This is done by embedding the words into a high-dimensional space where the distance between the vectors is a measure of the semantic similarity between the words.
