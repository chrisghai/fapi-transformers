import os

from sys import prefix
from fastapi import APIRouter
from typing import (
    Union,
    List,
    Dict,
)

from app.core.config import settings
from app.core.schemas.nlp import (
    BinaryZeroShotInput,
    QuestionAnsweringInput, 
    ZeroShotInput,
)

from app.core.utils.nlp import (
    question_answering as _question_answering,
    zero_shot as _zero_shot,
    binary_zero_shot as _binary_zero_shot,
    ner as _ner,
    translate as _translate,
)

router = APIRouter(
    prefix="/nlp",
    tags=["nlp"],
)


@router.get("/models")
async def list_models():
    models = os.listdir(settings.NLP_ROOT)
    return models


@router.post("/translate/{src}-{tgt}")
async def translate(
    document: str,
    src: str,
    tgt: str,
    num_beams: int = 50,
) -> str:
    translation = _translate(
        document,
        src=src,
        tgt=tgt,
        num_beams=num_beams,
    )
    return translation


@router.post("/entity-recognition")
async def ner(
    documents: List[str],
) -> List[Dict]:
    results = _ner(documents)
    return results


@router.post("/question-answering")
async def question_answering(
    qa_input: QuestionAnsweringInput,
) -> Dict:
    results = _question_answering(dict(qa_input))
    return results


@router.post("/zero-shot")
async def zero_shot(
    zs_input: ZeroShotInput,
    format_results: bool = True,
) -> List[Dict]:
    results = _zero_shot(
        dict(zs_input),
        format_results=format_results,
    )
    return results


@router.post("/binary-zero-shot")
async def binary_zero_shot(
    zs_input: BinaryZeroShotInput,
    raw_scores: bool = False,
    threshold: float = 0.5,
) -> Union[List[float], List[bool]]:
    results = _binary_zero_shot(
        dict(zs_input),
        raw_scores,
        threshold,
    )
    return results