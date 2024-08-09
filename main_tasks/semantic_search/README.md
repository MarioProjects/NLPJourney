# Semantic Search

Here we will explore semantic search problems. Semantic search is a search technique that takes the meaning of words into account rather than just the presence of the words themselves. This is done by embedding the words into a high-dimensional space where the distance between the vectors is a measure of the semantic similarity between the words.

Generally we will need:
- A model to generate embeddings for the text
- A search algorithm to find the most similar embeddings


## HF FAISS

Basic usage of [***FAISS index***](https://faiss.ai/) with Hugging Face Transformers. In this Notebook we:
1. Load a dataset from the Hugging Face Datasets library about GitHub issues
2. Filter and preprocess the dataset
3. Load a pretrained sentence-transformers model for [**asymmetric semantic search**](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search).
4. Use the model to generate embeddings for the text from the `[CLS]` token
5. Build a FAISS index from the embeddings
6. Use the index to find the most similar issues to a given query
