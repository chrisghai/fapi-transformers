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
    TranslationInput,
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
async def list_models() -> List[str]:
    models = os.listdir(settings.NLP_ROOT)
    return models


#@router.post("/translate/{src}-{tgt}")
@router.post("/translate/{model}")
async def translate(
    document: TranslationInput,
    #src: str,
    #tgt: str,
    model: str,
    num_beams: int = 50,
) -> Union[dict, str]:
    translation = _translate(
        document=document.text,
        #src=src,
        #tgt=tgt,
        model=model,
        num_beams=num_beams,
    )
    if translation is None:
        return {"message": "Model not loaded."}
    return translation


@router.post("/entity-recognition")
async def ner(
    documents: List[str],
) -> Union[dict, List[Dict]]:
    results = _ner(documents)
    if results is None:
        return {"message": "Model not loaded."}
    return results


@router.post("/question-answering")
async def question_answering(
    qa_input: QuestionAnsweringInput,
) -> Dict:
    results = _question_answering(dict(qa_input))
    if results is None:
        return {"message": "Model not loaded."}
    return results


@router.post("/zero-shot")
async def zero_shot(
    zs_input: ZeroShotInput,
    format_results: bool = True,
) -> Union[dict, List[Dict]]:
    results = _zero_shot(
        dict(zs_input),
        format_results=format_results,
    )
    if results is None:
        return {"message": "Model not loaded."}
    return results


@router.post("/binary-zero-shot")
async def binary_zero_shot(
    zs_input: BinaryZeroShotInput,
    raw_scores: bool = False,
    threshold: float = 0.5,
) -> Union[dict, List[float], List[bool]]:
    results = _binary_zero_shot(
        dict(zs_input),
        raw_scores,
        threshold,
    )
    if results is None:
        return {"message": "Model not loaded."}
    return results