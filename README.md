# autollm
Easy data distillation of LLMs for text classification, information extraction, and open-ended tasks

`autollm` works like any AutoML libraries you would expect: input data, you get a trained model. 
The twist is that you don’t have to provide annotations along with the data, 
it is annotated automatically by an LLM (e.g. ChatGPT). 
The second twist is that you don’t even have to provide data: 
a meta-task of `autollm` is to detect relevant documents from large corpora (e.g. CommonCrawl).

## Installation

### via pip
`pip install automl-llm`

### via uv
`uv add automl-llm`

### editable
```bash
git clone https://github.com/PaulLerner/autollm.git
cd autollm
uv sync
```

## Annotation
```bash
OPENAI_API_KEY="your-api-key" python -m autollm.annotate --dataset_path=nyu-mll/glue --dataset_name=wnli --input_columns+=sentence1 --input_columns+=sentence2 --classes+=entailment --classes+=not_entailment
```

## Distillation
```bash
python -m autollm.distill glue --input_columns+=sentence1 --input_columns+=sentence2 --classes+=entailment --classes+=not_entailment
```

## Motivation
Since the release of ChatGPT, a large body of work in academia and industry have revolved around 
distilling general-purpose but compute-intensive LLMs (e.g. ChatGPT) 
into task-specific but compact classifiers (e.g. [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) or LLMs (e.g. [NuExtract3](https://huggingface.co/numind/NuExtract3)).

The industrial benefit is obvious, your company pays a monthly subscription for six months, 
collecting more than enough data to train a compact model that will cost a 100th of the price while keeping all data local.

## Similar libraries

- [DistillKit](https://github.com/arcee-ai/DistillKit) not "autoML" style, more of a technical toolkit/framework for knowledge distillation (from logits)

## Roadmap
TODO link each with issues

- annotate: annotate text with an OpenAI-compatible API
  - the task might be to detect relevant texts for the downstream task from a large-scale corpora 
    (e.g. [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
  - benchmark: 
    - accuracy of annotation against existing human annotations using sklearn metrics
  - for information extraction: again use https://developers.openai.com/api/docs/guides/structured-outputs ? so single ClassifierIE class?
  - for open-ended tasks: use https://developers.openai.com/api
  - also provide alternative models
    - claude
    - gemini
    - https://docs.vllm.ai/en/latest/api/vllm/index.html
- distill: train a compact model on the annotated dataset
  - get list from transformers automodel?
  - option for language (defaults multimodal)
  - for sequence tagging: also encoder but token-wise classifier
  - for open-ended: most likely decoder-only (or encoder-decoder?)
  - also accept pre-annotated dataset (not from `autollm`)
  - benchmark to compare against a human-engineered fine-tuning
    - GLUE
- GUI
- deploy on server so that user doesn't need local compute

## Contributing
Feel free to open an issue or PR to contribute. 
The roadmap will probably never happen without your help :)

### Building
Use:
- `uv version --bump patch` for `1.2.3 => 1.2.4`
- `uv version --bump minor` for `1.2.3 => 1.3.0`
- `uv version --bump major` for `1.2.3 => 2.0.0`

Then
```bash
uv build
uv publish --token=<TOKEN>
```
