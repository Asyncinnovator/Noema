"""Transformer-based relation extraction, used as a fallback for sentences
the rule-based extractor (parser.relations) finds nothing in. Uses
Babelscape/rebel-large (REBEL), a seq2seq model that generates structured
relation triplets as text, e.g.
"<triplet> Google <subj> Larry Page <obj> founded by". The triplet parser
below is REBEL's own published parsing function from its model card.
"""

from __future__ import annotations

from functools import lru_cache

from noema.parser.entities import _load_model as _load_spacy_model
from noema.parser.relations import extract_relations
from noema.schema import Document, Entity, Relation, SourceSpan

DEFAULT_MODEL = "Babelscape/rebel-large"

# Lower than the rule-based extractor's default (1.0) -- REBEL's
# generated triplets are a less-validated source.
TRANSFORMER_RELATION_CONFIDENCE = 0.75

_GENERATE_KWARGS = {"max_length": 256, "length_penalty": 0, "num_beams": 3}


@lru_cache(maxsize=1)
def _load_rebel(model_name: str = DEFAULT_MODEL):
    # Imported lazily: a much heavier install than the rest of the parser needs.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _generate_raw_triplet_text(text: str, model_name: str = DEFAULT_MODEL) -> str:
    tokenizer, model = _load_rebel(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    output_ids = model.generate(inputs["input_ids"], **_GENERATE_KWARGS)
    return tokenizer.decode(output_ids[0], skip_special_tokens=False)


def _extract_triplets(text: str) -> list[dict[str, str]]:
    """Parse REBEL's raw generated <triplet>/<subj>/<obj> text into dicts."""
    triplets = []
    relation, subject, object_ = "", "", ""
    text = text.strip()
    current = "x"
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append({"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()})
    return triplets


def _normalize_predicate(relation_text: str) -> str:
    """Normalize a REBEL relation type ("headquarters location") to the
    same lowercase snake_case convention parser.relations uses.
    """
    return "_".join(relation_text.lower().split())


def _match_entity(mention_text: str, entities: list[Entity]) -> Entity | None:
    """Resolve a triplet's head/tail text to an already-extracted Entity.

    Deliberately conservative: case-insensitive exact match against an
    entity's name or aliases only, since REBEL's free-text output has no
    character-offset grounding. A triplet that doesn't match a known
    entity is dropped rather than used to invent a new one.
    """
    target = mention_text.strip().lower()
    for entity in entities:
        if entity.name.lower() == target:
            return entity
        if any(alias.lower() == target for alias in entity.aliases):
            return entity
    return None


def extract_relations_transformer(
    document: Document,
    entities: list[Entity],
    text: str | None = None,
    start_char: int = 0,
    model_name: str = DEFAULT_MODEL,
) -> list[Relation]:
    """Run REBEL over `text` (defaulting to the whole document) and return
    Relations for every triplet whose head AND tail both match a known
    entity. `text`/`start_char` let callers scope extraction to a single
    sentence while recording a correct document-relative mention span.
    """
    scoped_text = document.text if text is None else text
    raw = _generate_raw_triplet_text(scoped_text, model_name)
    triplets = _extract_triplets(raw)

    mention = SourceSpan(
        doc_id=document.id,
        start_char=start_char,
        end_char=start_char + len(scoped_text),
        text=scoped_text,
    )

    relations = []
    for triplet in triplets:
        subject = _match_entity(triplet["head"], entities)
        obj = _match_entity(triplet["tail"], entities)
        if subject is None or obj is None or subject.id == obj.id:
            continue
        relations.append(
            Relation(
                subject_id=subject.id,
                predicate=_normalize_predicate(triplet["type"]),
                object_id=obj.id,
                mentions=[mention],
                confidence=TRANSFORMER_RELATION_CONFIDENCE,
            )
        )
    return relations


def extract_relations_hybrid(
    document: Document, entities: list[Entity], model_name: str = DEFAULT_MODEL
) -> list[Relation]:
    """Rule-based extraction first (fast, precise), transformer extraction
    only for sentences the rule-based pass found zero relations in.
    """
    rule_based = extract_relations(document, entities)
    covered_sentence_starts = {mention.start_char for r in rule_based for mention in r.mentions}

    nlp = _load_spacy_model()
    doc = nlp(document.text)
    missed_sentences = [sent for sent in doc.sents if sent.start_char not in covered_sentence_starts]

    if not missed_sentences:
        return rule_based

    transformer_relations: list[Relation] = []
    for sent in missed_sentences:
        transformer_relations.extend(
            extract_relations_transformer(
                document, entities, text=sent.text, start_char=sent.start_char, model_name=model_name
            )
        )

    return rule_based + transformer_relations