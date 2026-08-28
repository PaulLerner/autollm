import os
from typing import Literal, List, Union
from jsonargparse import CLI
from pathlib import Path

import datasets

import pydantic
import openai


openai.api_key = os.environ["OPENAI_API_KEY"]


class Classifier:
    def __init__(self, classes: List[str], model_name: str = "gpt-5.6-luna", system_prompt: str = "Classify each text into exactly one category."):

        class ClassificationResult(pydantic.BaseModel):
            category: Literal[*classes] # type: ignore

        self.text_format = ClassificationResult
        self.client = openai.OpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt
        
    def annotate(self, *texts: List[str]):
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": "\n".join(texts)
                }
            ],
            text_format=self.text_format,
        )
        return {"autollm_annotation": response.output_parsed.category}


def main(dataset_path: str, input_columns: List[str], classes: List[str], 
         output_path: Path = None, sample: int = None, dataset_name: str = None, split: str = None,
         model_name: str = "gpt-5.6-luna", 
         system_prompt: str = "Classify each text into exactly one category."
    ):
    dataset = datasets.load_dataset(dataset_path, name=dataset_name, split=split)
    if sample is not None:        
        dataset = dataset.shuffle()
        if isinstance(dataset,datasets.DatasetDict):
            for subset_name, subset in dataset.items():
                dataset[subset_name] = subset.select(range(sample))
        else:
            dataset = dataset.select(range(sample))
    # TODO list available models
    classifier = Classifier(classes, model_name=model_name, system_prompt=system_prompt)
    dataset = dataset.map(classifier.annotate, input_columns=input_columns)
    if output_path is not None:
        dataset.save_to_disk(output_path)
    else:
        dataset.save_to_disk(dataset_path.split("/")[-1])
    # TODO option to push_to_hub


if __name__ == "__main__":
    CLI(main, as_positional=False)