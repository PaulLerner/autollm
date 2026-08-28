import copy
from pathlib import Path
from typing import List
from jsonargparse import CLI

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding


def main(dataset_path: Path, input_columns: List[str], classes: List[str], model_name: str = "jhu-clsp/mmBERT-small", 
         training_args: TrainingArguments = TrainingArguments()):
    # TODO list available models
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer)

    dataset = datasets.load_from_disk(dataset_path)
    useless_columns = copy.deepcopy(dataset.column_names)
    dataset = dataset.map(tokenizer, batched=True, input_columns=input_columns)
    # TODO store in model config?
    label2id = {label: i for i, label in enumerate(classes)}
    dataset = dataset.map(lambda label: {"labels": label2id[label]}, input_columns="autollm_annotation")
    for subset_name, subset in dataset.items():
        dataset[subset_name] = subset.remove_columns(useless_columns[subset_name])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
        #compute_metrics=TODO
    )

    # TODO hyperparam https://huggingface.co/docs/transformers/hpo_train
    trainer.train()


if __name__ == "__main__":
    CLI(main, as_positional=False)