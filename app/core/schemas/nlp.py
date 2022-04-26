from pydantic import BaseModel
from typing import Union, List

class QuestionAnsweringInput(BaseModel):
    question: str
    context: str

class ZeroShotInput(BaseModel):
    documents: Union[List[str], str]
    topics: Union[List[str], str]

class BinaryZeroShotInput(BaseModel):
    documents: Union[List[str], str]
    topic: str