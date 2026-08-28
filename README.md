# autollm
Easy data distillation of LLMs for text classification, information extraction, and open-ended tasks

`autollm` works like any AutoML libraries you would expect: input data, you get a trained model. 
The twist is that you don’t have to provide annotations along with the data, 
it is annotated automatically by an LLM (e.g. ChatGPT). 
The second twist is that you don’t even have to provide data: 
a meta-task of `autollm` is to detect relevant documents from large corpora (e.g. CommonCrawl).

# Motivation
Since the release of ChatGPT, a large body of work in academia and industry have revolved around 
distilling general-purpose but compute-intensive LLMs (e.g. ChatGPT) 
into task-specific but compact classifiers (e.g. [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) or LLMs (e.g. [NuExtract3](https://huggingface.co/numind/NuExtract3)).

The industrial benefit is obvious, your company pays a monthly subscription for six months, 
collecting more than enough data to train a compact model that will cost a 100th of the price while keeping all data local.

# Similar libraries
TODO

- [DistillKit](https://github.com/arcee-ai/DistillKit) not "autoML" style, more of a technical toolkit/framework for knowledge distillation (from logits)

# Roadmap
TODO link each with issues

- annotate: annotate text with an OpenAI-compatible API
  - the task might be to detect relevant texts for the downstream task from a large-scale corpora 
    (e.g. [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
  - benchmark: 
    - accuracy of annotation against existing human annotations
      - sklearn metrics?
      - hf metrics?
      - lightning metrics?
  - for information extraction: again use https://developers.openai.com/api/docs/guides/structured-outputs ? so single ClassifierIE class?
  - for open-ended tasks: use https://developers.openai.com/api
  - also provide alternative models
    - claude
    - gemini
    - https://docs.vllm.ai/en/latest/api/vllm/index.html
- distill: train a compact model on the annotated dataset
  - implement using one of the following framework:
    - lightning
    - transformers
    - KTransformers
  - for text classification: pick encoder models
    - get list from transformers automodel?
    - option for language (defaults multimodal)
  - for sequence tagging: also encoder but token-wise classifier
  - for open-ended: most likely decoder-only (or encoder-decoder?)
  - also accept pre-annotated dataset (not from `autollm`)
  - benchmark to compare against a human-engineered fine-tuning
    - GLUE
- GUI
- deploy on server so that user doesn't need local compute
