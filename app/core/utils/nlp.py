import os
import ftfy
import torch

from emoji import get_emoji_regexp

from typing import (
    Union,
    List,
    Dict,
)

from transformers import (
    pipeline,
    Pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

from app.core.config import settings

QA_MODEL = settings.QA_MODEL
ZS_MODEL = settings.ZS_MODEL
NER_MODEL = settings.NER_MODEL
RU_EN_MODEL = settings.RU_EN_MODEL
RU_EN_ENSEMBLE = settings.RU_EN_ENSEMBLE

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

    "ru_en": pipeline(
        task="translation_ru_to_en",
        model=AutoModelForSeq2SeqLM.from_pretrained(RU_EN_MODEL["path"]),
        tokenizer=AutoTokenizer.from_pretrained(RU_EN_MODEL["path"]),
    ) if (
        os.path.isdir(RU_EN_MODEL["path"])
        and RU_EN_MODEL["load"]
    ) else None,
}


try:
    from fairseq.hub_utils import GeneratorHubInterface
    from fairseq.models.transformer import TransformerModel
    
    LOADED_MODELS["ru_en_ensemble"] = TransformerModel.from_pretrained(
        RU_EN_ENSEMBLE["path"],
        checkpoint_file=":".join(
            sorted(
                [
                    f for f in os.listdir(RU_EN_ENSEMBLE["path"])
                    if ".pt" in f
                ]
            )
        ),
        bpe="fastbpe",
        bpe_codes=os.path.join(RU_EN_ENSEMBLE["path"], "bpecodes"),
    ) if (
        os.path.isdir(RU_EN_ENSEMBLE["path"])
        and RU_EN_ENSEMBLE["load"]
    ) else None

except:
    pass


def clean_text(
    document: str,
) -> str:
    document = ftfy.fix_text(document)
    document = get_emoji_regexp().sub(
        r'', 
        document.encode().decode("utf-8"),
    )
    return document


def chunk(
    document: str,
) -> List[str]:
    return [
        s.strip() for s in document.split("\n")
        if s.strip()
    ]


def translate(
    document: str,
    #src: str = "ru",
    #tgt: str = "en",
    model: str,
    num_beams: int = 50,
) -> str:
    #classifier = LOADED_MODELS.get(f"{src}_{tgt}")
    classifier = LOADED_MODELS.get(model)
    if classifier is None:
        return None

    document = clean_text(document)
    chunked_document = chunk(document)
    translation = ""
    for i, sentence in enumerate(chunked_document):
        if isinstance(classifier, Pipeline):
            translation += classifier(
                sentence,
                num_beams=num_beams,
            )[0]["translation_text"]
        
        else:
            classifier.eval()
            with torch.no_grad():
                translation += classifier.translate(
                    sentence,
                    beams=num_beams,
                )

        if i + 1 < len(chunked_document):
            translation += "\n\n"

    return translation


def ner(
    documents: List[str],
) -> List[Dict]:
    classifier = LOADED_MODELS.get("ner")
    if classifier is None:
        return None

    results = classifier(documents)
    for result in results:
        for r in result:
            r["score"] = float(r["score"])
    if results:
        results = format_ner(results)

    return results


def format_ner(
    results: List[Dict],
) -> List[Dict]:
    new_results = []
    for result in results:
            for i, r in enumerate(reversed(result)):
                r["score"] = float(r["score"])
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
        return None
        
    result = classifier(qa_input)
    return result


def zero_shot(
    zs_input: Dict,
    format_results: bool = True,
) -> List[Dict]:
    classifier = LOADED_MODELS.get("zero-shot")
    if classifier is None:
        return None
    
    documents, topics = (
        zs_input["documents"], 
        zs_input["topics"],
    )

    results = classifier(documents, topics)
    if results and format_results:
        results = format_zs(results)

    return results


def format_zs(results: List[Dict]) -> List[Dict]:
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
        return None

    documents, topic = (
        zs_input["documents"],
        zs_input["topic"],
    )

    results = classifier(documents, topic)
    if results:
        results = format_bzs(
            results,
            raw_scores=raw_scores,
            threshold=threshold,
        )

    return results


def format_bzs(
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