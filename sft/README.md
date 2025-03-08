# Supervised Fine-Tuning

## Index

- [1️⃣ Chat Templates](#1️⃣-chat-templates)
- [2️⃣ Supervised Fine-Tuning](#2️⃣-supervised-fine-tuning)
- [3️⃣ Low Rank Adaptation (LoRA)](#3️⃣-low-rank-adaptation-lora)
- [4️⃣ Evaluation](#4️⃣-evaluation)
- [References](#references)

## 1️⃣ Chat Templates

Chat templates **structure interactions** between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.

Checkout the Jupyter Notebook [here](chat_templates.ipynb) for a detailed guide on creating chat templates using Hugging Face's `transformers` library.

## 2️⃣ Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is a critical process for **adapting pre-trained language models to specific tasks**. It involves training the model on a task-specific dataset with labeled examples. For a detailed guide on SFT, including key steps and best practices, see [The supervised fine-tuning section of the TRL documentation](https://huggingface.co/docs/trl/en/sft_trainer).

Check out the Jupyter Notebook [here](sfttrainer.ipynb) for a detailed guide on supervised fine-tuning using Hugging Face's `transformers` library.


## 3️⃣ Low Rank Adaptation (LoRA)

Low Rank Adaptation (LoRA) is a technique for fine-tuning language models by adding low-rank matrices to the model’s layers. This allows for **efficient fine-tuning** while preserving the model’s pre-trained knowledge. One of the key benefits of LoRA is the significant memory savings it offers, making it possible to fine-tune large models on hardware with limited resources.

Checkout the Jupyter Notebook [here](lora.ipynb) for a detailed guide on using LoRA for fine-tuning language models with Hugging Face's `transformers` library.

> Note: We should ensure that and `<eos>` is placed at the end of each sample in the dataset!


## 4️⃣ Evaluation

Evaluation is a crucial step in the fine-tuning process. It allows us to **measure the performance** of the model on a task-specific dataset. As machine learning engineers you should maintain a suite of relevant evaluations for your targeted domain of interest.

### Automatic Benchmarks

Automatic benchmarks typically consist of curated datasets with **predefined tasks and evaluation metrics**. These benchmarks aim to assess various aspects of model capability, from basic language understanding to complex reasoning. The key advantage of using automatic benchmarks is their standardization - they allow for consistent comparison across different models and provide reproducible results.

However, it’s crucial to understand that benchmark performance doesn’t always translate directly to real-world effectiveness. A model that excels at academic benchmarks **may still struggle with specific domain applications** or practical use cases.

- **General Knowledge Benchmarks**: [MMLU](https://huggingface.co/datasets/cais/mmlu) (Massive Multitask Language Understanding) tests knowledge across 57 subjects, from science to humanities. While comprehensive, it may not reflect the depth of expertise needed for specific domains. TruthfulQA evaluates a model’s tendency to reproduce common misconceptions, though it can’t capture all forms of misinformation.
- **Reasoning Benchmarks**: [BBH](https://huggingface.co/datasets/lukaemon/bbh) (Big Bench Hard) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) focus on complex reasoning tasks. BBH tests logical thinking and planning, while GSM8K specifically targets mathematical problem-solving. These benchmarks help assess analytical capabilities but may not capture the nuanced reasoning required in real-world scenarios.
- **Language Understanding**: [HELM](https://github.com/stanford-crfm/helm) provides a holistic evaluation framework. Benchmarks like HELM offer insights into language processing capabilities on aspects like commonsense, world knowledge, and reasoning. But may not fully represent the complexity of natural conversation or domain-specific terminology.
- **Math benchmarks**: [The MATH benchmark](https://huggingface.co/papers/2103.03874) is another important evaluation tool for mathematical reasoning. It consists of 12,500 problems from mathematics competitions, covering algebra, geometry, number theory, counting, probability, and more. What makes MATH particularly challenging is that it requires multi-step reasoning, formal mathematical notation understanding, and the ability to generate step-by-step solutions. Unlike simpler arithmetic tasks, MATH problems often demand sophisticated problem-solving strategies and mathematical concept applications.
- **Coding**: The [HumanEval Benchmark](https://github.com/openai/human-eval) is a coding-focused evaluation dataset consisting of 164 programming problems. The benchmark tests a model’s ability to generate functionally correct Python code that solves the given programming tasks. What makes HumanEval particularly valuable is that it evaluates both code generation capabilities and functional correctness through actual test case execution, rather than just superficial similarity to reference solutions. The problems range from basic string manipulation to more complex algorithms and data structures.
- **Instruction-following benchmarks**: [Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/) is an automated evaluation framework designed to assess the quality of instruction-following language models. It uses GPT-4 as a judge to evaluate model outputs across various dimensions including helpfulness, honesty, and harmlessness. The framework includes a dataset of 805 carefully curated prompts and can evaluate responses against multiple reference models like Claude, GPT-4, and others. What makes Alpaca Eval particularly useful is its ability to provide consistent, scalable evaluations without requiring human annotators, while still capturing nuanced aspects of model performance that traditional metrics might miss.

We can implement evaluation for our finetuned model. We can use [lighteval](https://github.com/huggingface/lighteval) to evaluate our finetuned model on standard benchmarks, which contains a wide range of tasks built into the library. We just need to define the tasks we want to evaluate and the parameters for the evaluation.

LightEval tasks are defined using a specific format:

```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

|   Parameter  |                                    Description                                   |
|:------------:|:--------------------------------------------------------------------------------:|
| suite        | The benchmark suite (e.g., ‘mmlu’, ‘truthfulqa’)                                 |
| task         | Specific task within the suite (e.g., ‘abstract_algebra’)                        |
| num_few_shot | Number of examples to include in prompt (0 for zero-shot)                        |
| auto_reduce  | Whether to automatically reduce few-shot examples if prompt is too long (0 or 1) |

Example: `"mmlu|abstract_algebra|0|0"` evaluates on MMLU’s abstract algebra task with zero-shot inference.


Let’s set up an evaluation pipeline for our finetuned model. We will evaluate the model on set of sub tasks that relate to the domain of medicine.

Here’s a complete example of evaluating on automatic benchmarks relevant to one specific domain using Lighteval with the VLLM backend:

```bash
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results" \
    --save_generations true
```

Lighteval also include a python API for more detailed evaluation tasks, which is useful for manipulating the results in a more flexible way. Check out the [Lighteval documentation](https://huggingface.co/docs/lighteval/using-the-python-api) for more information.


### Alternative Evaluation Approaches

Many organizations have developed alternative evaluation methods to address the limitations of standard benchmarks:

- **LLM-as-Judge**: Using one language model to evaluate another’s outputs has become **increasingly popular**. This approach can provide more nuanced feedback than traditional metrics, though it comes with its own biases and limitations.
- **Evaluation Arenas**: Evaluation arenas like [Chatbot Arena](https://lmarena.ai/) offer a unique approach to LLM assessment through crowdsourced feedback. In these platforms, users engage in anonymous “battles” between two LLMs, asking questions and voting on which model provides better responses. This approach captures real-world usage patterns and preferences through diverse, challenging questions, with studies showing strong agreement between crowd-sourced votes and expert evaluations. While powerful, these platforms have limitations including potential user base bias, skewed prompt distributions, and a primary focus on helpfulness rather than safety considerations.


### Custom Evaluation

While standard benchmarks provide a useful baseline, they shouldn’t be your only evaluation method. Here’s how to develop a more comprehensive approach:

1. Start with relevant standard benchmarks to establish a baseline and enable comparison with other models.

2. Identify the specific requirements and challenges of your use case. What tasks will your model actually perform? What kinds of errors would be most problematic?

4. Develop custom evaluation datasets that reflect your actual use case. This might include:
  - Real user queries from your domain
  - Common edge cases you’ve encountered
  - Examples of particularly challenging scenarios

5. Consider implementing a multi-layered evaluation strategy:
  - Automated metrics for quick feedback
  - Human evaluation for nuanced understanding
  - Domain expert review for specialized applications
  - A/B testing in controlled environments



## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [SFTTrainer in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://github.com/huggingface/alignment-handbook)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)