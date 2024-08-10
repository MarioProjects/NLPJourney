# Semantic Search


Semantic search aims to improve search accuracy by understanding the searcher's intent and the contextual meaning of terms. The advantages are:

- Better handling of synonyms and context.
- Improved search relevance.
- Ability to find related concepts, not just exact matches.

Here we will explore semantic search problems. Semantic search is a search technique that takes the meaning of words into account rather than just the presence of the words themselves. This is done by embedding the words into a high-dimensional space where the distance between the vectors is a measure of the semantic similarity between the words.

Generally we will need:
- A model to generate embeddings for the text
- A search algorithm to find the most similar embeddings

# Index

- [Key Techniques](#key-techniques)
- [Common Tasks](#common-tasks)
- [Example Projects](#example-projects)
- [Evaluation Metrics](#evaluation-metrics)
- [Projects](#projects)
  - [HF FAISS](#hf-faiss)

## Key Techniques

- **Word Embeddings**: e.g., Word2Vec, GloVe.
- **Sentence Embeddings**: e.g., BERT, USE (Universal Sentence Encoder).
- **Dense Retrieval**: Using neural networks for matching queries and documents.
- **Approximate Nearest Neighbor (ANN) Search**:
  - FAISS (Facebook AI Similarity Search).
  - Annoy (Spotify).
  - HNSW (Hierarchical Navigable Small World).

## Common Tasks

- **Document retrieval**: Finding relevant documents based on a query
- **Question answering**: Providing answers to questions based on a document corpus.
- **Information extraction**: Extracting structured information from unstructured text.
- **Text classification**: Assigning categories or labels to text data.
- **Clustering**: Grouping similar documents together.
- **Chatbots**: Conversational agents that can understand and respond to user queries.

## Example Projects

- Building a semantic search engine for a large document corpus.
- Implementing a chatbot with improved query understanding.
- Creating a recommendation system based on content similarity.

## Evaluation Metrics

- **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of the first relevant document.
- **Precision@K**: Precision at a given rank K.
- **Recall@K**: Recall at a given rank K.


## Projects

### HF FAISS

Basic usage of [***FAISS index***](https://faiss.ai/) with Hugging Face Transformers. In this Notebook we:
1. Load a dataset from the Hugging Face Datasets library about GitHub issues
2. Filter and preprocess the dataset
3. Load a pretrained sentence-transformers model for [**asymmetric semantic search**](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search).
4. Use the model to generate embeddings for the text from the `[CLS]` token
5. Build a FAISS index from the embeddings
6. Use the index to find the most similar issues to a given query
