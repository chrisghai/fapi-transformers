import os

from typing import (
    Union,
    List,
    Dict,
)

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

from app.core.config import settings

QA_MODEL = settings.QA_MODEL
ZS_MODEL = settings.ZS_MODEL
NER_MODEL = settings.NER_MODEL

LOADED_MODELS = {
    "ner": pipeline(
        "ner",
        model=AutoModelForTokenClassification.from_pretrained(NER_MODEL["path"]),
        tokenizer=AutoTokenizer.from_pretrained(NER_MODEL["path"]),
        aggregation_strategy="simple",
    ) if (
        os.path.isdir(NER_MODEL["path"])
        and NER_MODEL["load"]
    ) else None,

    "qa": pipeline(
        task="question-answering",
        model=AutoModelForQuestionAnswering.from_pretrained(QA_MODEL["path"]),
        tokenizer=AutoTokenizer.from_pretrained(QA_MODEL["path"]),
    ) if (
        os.path.isdir(QA_MODEL["path"])
        and QA_MODEL["load"]
    ) else None,

    "zero-shot": pipeline(
        task="zero-shot-classification",
        model=AutoModelForSequenceClassification.from_pretrained(ZS_MODEL["path"]),
        tokenizer=AutoTokenizer.from_pretrained(ZS_MODEL["path"]),
    ) if (
        os.path.isdir(ZS_MODEL["path"])
        and ZS_MODEL["load"]
    ) else None,
}

def ner(
    documents: List[str],
) -> List[Dict]:
    classifier = LOADED_MODELS.get("ner")
    if classifier is None:
        return {"message": "Model is not loaded."}

    results = classifier(documents)
    for result in results:
        for r in result:
            r["score"] = float(r["score"])
    if results:
        results = format_ner_results(results)

    return results

def format_ner_results(
    results: List[Dict],
) -> List[Dict]:
    new_results = []
    for result in results:
        if result:
            for i, r in zip(range(len(result)), reversed(result)):
                if (
                    i + 1 < len(result)
                    and r["entity_group"] == result[i + 1]["entity_group"]
                    and r["start"] == result[i + 1]["end"]
                ):
                    result[i + 1]["end"] = r["end"]
                    result[i + 1]["word"] += r["word"]
                    result[i + 1]["score"] = max(
                        r["score"],
                        result[i + 1]["score"],
                    )
                    r["remove"] = True
            new_results.append(
                [
                    r for r in result if not r.get("remove")
                ]
            )

    return new_results

def question_answering(
    qa_input: Dict,
) -> Dict:
    classifier = LOADED_MODELS.get("qa")
    if classifier is None:
        return {"message": "Model is not loaded."}
        
    result = classifier(qa_input)
    return result

def zero_shot(
    zs_input: Dict,
    format_results: bool = True,
) -> List[Dict]:
    classifier = LOADED_MODELS.get("zero-shot")
    if classifier is None:
        return {"message": "Model is not loaded."}
    
    documents, topics = (
        zs_input["documents"], 
        zs_input["topics"],
    )

    results = classifier(documents, topics)
    if results and format_results:
        results = format_zs_results(results)

    return results

def format_zs_results(results: List[Dict]) -> List[Dict]:
    return [
        {
            l: s for l, s in zip(
                _result["labels"], 
                _result["scores"]
            )
        } for _result in results
    ]

def binary_zero_shot(
    zs_input: Dict,
    raw_scores: bool = False,
    threshold: float = 0.5,
) -> List[Dict]:
    classifier = LOADED_MODELS.get("zero-shot")
    if classifier is None:
        return {"message": "Model is not loaded."}

    documents, topic = (
        zs_input["documents"],
        zs_input["topic"],
    )

    results = classifier(documents, topic)
    if results:
        results = format_bzs_results(
            results,
            raw_scores=raw_scores,
            threshold=threshold,
        )

    return results

def format_bzs_results(
    results: List[Dict],
    raw_scores: bool,
    threshold: float,
) -> Union[List[float], List[bool]]:
    return [
        (
            r["scores"][0] > threshold if not raw_scores
            else r["scores"][0]
        )
        for r in results
    ]