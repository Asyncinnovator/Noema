"""Transformer-based relation extraction, used as a targeted fallback for
sentences the rule-based extractor (parser.relations) finds nothing in.

Phase 5 scope, per the project plan: "Swap/augment relation extraction with
a transformer-based model for sentences rule-based patterns miss." This
module does the "augment" half -- it does NOT replace parser.relations.
The rule-based extractor stays the default because it's fast, has no model
download, and is precise by construction (it only fires on four specific,
well-understood clause shapes). This module exists for everything outside
those four shapes: comparatives, multi-clause sentences, appositives,
relations without a clean verb, and so on -- the "sentences rule-based
patterns miss" the plan calls out.

Uses Babelscape/rebel-large (REBEL), a BART-based seq2seq model that
performs end-to-end relation extraction by generating structured triplets
directly as text, e.g. "<triplet> Google <subj> Larry Page <obj> founded by"
for more than 200 relation types (EMNLP 2021, Babelscape). The triplet
parser below is Babelscape's own published parsing function from the
model card (https://huggingface.co/Babelscape/rebel-large), not a
reimplementation -- REBEL's output format is exact and undocumented
deviations from the official parser are a common source of silently
dropped triplets.

Honesty note: this sandbox's network doesn't reach huggingface.co, so the
real model could not be run here. The triplet PARSER is tested directly
against REBEL's real, documented output format (verified from the model
card, not guessed), and the surrounding pipeline (entity matching, predicate
normalization, hybrid fallback logic) is tested with the generation step
mocked. Run this against the live model on your own machine to confirm
real extraction quality on your data.
"""

from __future__ import annotations

from functools import lru_cache

from noema.parser.entities import _load_model as _load_spacy_model
from noema.parser.relations import extract_relations
from noema.schema import Document, Entity, Relation, SourceSpan

DEFAULT_MODEL = "Babelscape/rebel-large"

# REBEL's output is generated at lower confidence than the rule-based
# extractor's hand-written, precision-first patterns (which default to
# 1.0) -- this reflects that a general-purpose generative model's triplets
# are a genuinely less-validated source, not an arbitrary number.
TRANSFORMER_RELATION_CONFIDENCE = 0.75

_GENERATE_KWARGS = {"max_length": 256, "length_penalty": 0, "num_beams": 3}


@lru_cache(maxsize=1)
def _load_rebel(model_name: str = DEFAULT_MODEL):
    # Imported lazily -- this is the only module in the codebase that needs
    # a seq2seq transformer, and it's a much heavier install than anything
    # else the parser package requires.
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
    """Parse REBEL's raw generated text into triplet dicts.

    This is Babelscape's own published parsing function (see module
    docstring), copied rather than reimplemented from scratch -- REBEL's
    <triplet>/<subj>/<obj> output format is exact, and small "obvious
    simplifications" to this state machine are a documented way to
    silently drop or corrupt triplets.
    """
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
    """REBEL relation types are Wikidata-property-like phrases ("headquarters
    location", "chief executive officer") -- normalize to the same
    lowercase snake_case convention parser.relations uses.
    """
    return "_".join(relation_text.lower().split())


def _match_entity(mention_text: str, entities: list[Entity]) -> Entity | None:
    """Resolve a triplet's head/tail text to an already-extracted Entity.

    Deliberately conservative: only case-insensitive exact match against an
    entity's name or one of its known aliases. REBEL generates free text,
    not offsets into the source document, so there's no character-span
    grounding available the way parser.relations has -- accepting a fuzzy
    substring match here would risk attaching a transformer-generated fact
    to the wrong entity. A triplet whose head or tail doesn't match a known
    entity is dropped rather than used to silently invent a new,
    ungrounded one (see extract_relations_transformer).
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
    Relations for every triplet whose head AND tail both match an entity
    already in `entities`.

    `text`/`start_char` let callers scope extraction to a single sentence
    (see extract_relations_hybrid) while still recording a correct
    document-relative mention span.
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

    Rule-based relations always record their mention as the full sentence
    span (see parser.relations), so "which sentences did rule-based
    coverage miss" is answerable directly from the relations it returned,
    without re-parsing anything to find out.
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