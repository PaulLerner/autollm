import os
import openai
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

openai.api_key = os.environ["OPENAI_API_KEY"]




class Classification(BaseModel):
    category: Literal[
        "billing",
        "technical_support",
        "sales",
        "other"
    ]

def classify(text: str) -> Classification:
    response = client.responses.parse(
        model="gpt-5.6-luna",
        #gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": """
You are a text classifier.

Classify each customer message into exactly one category:

- billing: invoices, payments, refunds, charges, subscriptions
- technical_support: bugs, errors, login problems, technical issues
- sales: pricing questions, buying, plans, upgrades
- other: anything that doesn't fit the categories

Return the most appropriate category, a confidence score from 0 to 1,
and a short reason.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        text_format=Classification,
    )

    return response.output_parsed


client = OpenAI()
result = classify("I was charged twice for my subscription.")

print(result.category)