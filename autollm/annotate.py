import os
from typing import Literal, List, Union
from jsonargparse import CLI
from pathlib import Path

import datasets

import pydantic
import openai


openai.api_key = os.environ["OPENAI_API_KEY"]


class Classifier:
    def __init__(self, classes: List[str], system_prompt: str = "Classify each text into exactly one category.", model_name: str = "gpt-5.6-luna"):

        class ClassificationResult(pydantic.BaseModel):
            category: Literal[*classes] # type: ignore

        self.text_format = ClassificationResult
        self.client = openai.OpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt
        
    def annotate(self, text: str):
        # TODO batch https://developers.openai.com/api/docs/guides/batch
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            text_format=self.text_format,
        )
        return {"autollm_annotation": response.output_parsed.category}


def main(dataset_name: str, input_column: str, classes: List[str], 
         output_path: Path = None, sample: int = None, split: str = "train",
         model_name: str = "gpt-5.6-luna", 
         system_prompt: str = "Classify each text into exactly one category."
    ):
    dataset = datasets.load_dataset(dataset_name, split=split)
    if sample is not None:
        dataset = dataset.shuffle().select(range(sample))
    # TODO list available models
    classifier = Classifier(classes, system_prompt=system_prompt, model_name=model_name)
    dataset = dataset.map(classifier.annotate, input_columns=input_column)
    if output_path is not None:
        dataset.save_to_disk(output_path)
    else:
        dataset.save_to_disk(dataset_name)


if __name__ == "__main__":
    CLI(main, as_positional=False)