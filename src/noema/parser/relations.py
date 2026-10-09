"""Text -> Relation extraction using spaCy's dependency parse.

Rule-based patterns over the dependency tree (see parser.transformer_relations
for a learned alternative). Recognizes four clause shapes per sentence root:
active SVO, passive+agent, passive+prep, and copular "role of". Anything
else is skipped. Requires entities already extracted for the document
(parser.entities.extract_entities) to resolve subject/object tokens to
Entity ids via character-offset overlap.
"""

from __future__ import annotations

import re

from spacy.tokens import Token

from noema.parser.entities import _load_model
from noema.schema import Document, Entity, Relation, SourceSpan

DEFAULT_MODEL = "en_core_web_sm"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_predicate(text: str) -> str:
    """'CEO' -> 'ceo', 'Chief Executive Officer' -> 'chief_executive_officer'."""
    return _NON_ALNUM.sub("_", text.strip().lower()).strip("_")


def _span_index(entities: list[Entity], doc_id: str) -> list[tuple[int, int, Entity]]:
    """Character-offset index of this document's entity mentions, for
    resolving a dependency-parse token back to the Entity it belongs to.
    """
    index = []
    for entity in entities:
        for m in entity.mentions:
            if m.doc_id == doc_id:
                index.append((m.start_char, m.end_char, entity))
    return index


def _resolve_entity(token: Token, index: list[tuple[int, int, Entity]]) -> Entity | None:
    """Map a single dependency-parse token to the Entity whose mention span
    contains it. A token's char offset falls inside its full entity span
    even when the entity is multi-word (e.g. token 'Page' inside 'Larry Page'),
    so containment -- not equality -- is the right check.
    """
    token_start, token_end = token.idx, token.idx + len(token.text)
    for start, end, entity in index:
        if token_start >= start and token_end <= end:
            return entity
    return None


def _expand_conjuncts(token: Token) -> list[Token]:
    """'Larry Page and Sergey Brin' -> both head tokens, so 'X founded Google
    and Yahoo' or 'X and Y founded Google' each yield one relation per pair.
    """
    return [token, *token.conjuncts]


def _add_or_merge(
    relations: dict[tuple[str, str, str], Relation],
    subject: Entity,
    predicate: str,
    obj: Entity,
    mention: SourceSpan,
) -> None:
    key = (subject.id, predicate, obj.id)
    existing = relations.get(key)
    if existing is not None:
        existing.mentions.append(mention)
        return
    relation = Relation(subject_id=subject.id, predicate=predicate, object_id=obj.id)
    relation.mentions.append(mention)
    relations[key] = relation


def extract_relations(
    document: Document,
    entities: list[Entity],
    model_name: str = DEFAULT_MODEL,
) -> list[Relation]:
    """Extract subject-predicate-object relations from a document's text,
    given the entities already extracted for that same document.
    """
    nlp = _load_model(model_name)
    doc = nlp(document.text)
    index = _span_index(entities, document.id)

    relations: dict[tuple[str, str, str], Relation] = {}

    for sent in doc.sents:
        root = next((t for t in sent if t.dep_ == "ROOT"), None)
        if root is None or root.pos_ not in ("VERB", "AUX"):
            continue

        subj_token = next(
            (c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None
        )
        if subj_token is None:
            continue
        is_passive = subj_token.dep_ == "nsubjpass"

        mention = SourceSpan(
            doc_id=document.id,
            start_char=sent.start_char,
            end_char=sent.end_char,
            text=sent.text,
        )

        if is_passive:
            agent = next((c for c in root.children if c.dep_ == "agent"), None)
            if agent is not None:
                # "Google was founded by Larry Page" -> founded(Larry Page, Google)
                pobj = next((c for c in agent.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                predicate = root.lemma_.lower()
                for s in _expand_conjuncts(pobj):
                    for o in _expand_conjuncts(subj_token):
                        s_ent, o_ent = _resolve_entity(s, index), _resolve_entity(o, index)
                        if s_ent and o_ent and s_ent.id != o_ent.id:
                            _add_or_merge(relations, s_ent, predicate, o_ent, mention)
                continue

            prep = next((c for c in root.children if c.dep_ == "prep"), None)
            if prep is not None:
                # "Google is headquartered in Mountain View" -> headquartered_in
                pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                if pobj is None:
                    continue
                predicate = f"{root.lemma_.lower()}_{prep.text.lower()}"
                for s in _expand_conjuncts(subj_token):
                    for o in _expand_conjuncts(pobj):
                        s_ent, o_ent = _resolve_entity(s, index), _resolve_entity(o, index)
                        if s_ent and o_ent and s_ent.id != o_ent.id:
                            _add_or_merge(relations, s_ent, predicate, o_ent, mention)
            continue

        # Active voice
        dobj = next((c for c in root.children if c.dep_ == "dobj"), None)
        if dobj is not None:
            # "Larry Page founded Google" -> founded(Larry Page, Google)
            predicate = root.lemma_.lower()
            for s in _expand_conjuncts(subj_token):
                for o in _expand_conjuncts(dobj):
                    s_ent, o_ent = _resolve_entity(s, index), _resolve_entity(o, index)
                    if s_ent and o_ent and s_ent.id != o_ent.id:
                        _add_or_merge(relations, s_ent, predicate, o_ent, mention)

            # "Google Inc. named Sundar Pichai as CEO" -> ceo_of(Sundar Pichai, Google Inc.)
            # Emitted in addition to the plain SVO relation above, not instead of it.
            as_prep = next(
                (c for c in root.children if c.dep_ == "prep" and c.text.lower() == "as"), None
            )
            if as_prep is not None:
                role_pobj = next((c for c in as_prep.children if c.dep_ == "pobj"), None)
                if role_pobj is not None:
                    role_predicate = f"{_normalize_predicate(role_pobj.text)}_of"
                    for s in _expand_conjuncts(dobj):
                        for o in _expand_conjuncts(subj_token):
                            s_ent, o_ent = _resolve_entity(s, index), _resolve_entity(o, index)
                            if s_ent and o_ent and s_ent.id != o_ent.id:
                                _add_or_merge(relations, s_ent, role_predicate, o_ent, mention)
            continue

        attr = next((c for c in root.children if c.dep_ == "attr"), None)
        if attr is not None:
            # "Sundar Pichai is the CEO of Google" -> ceo_of(Sundar Pichai, Google)
            prep = next((c for c in attr.children if c.dep_ == "prep"), None)
            if prep is None:
                continue
            pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
            if pobj is None:
                continue
            predicate = f"{_normalize_predicate(attr.text)}_of"
            for s in _expand_conjuncts(subj_token):
                for o in _expand_conjuncts(pobj):
                    s_ent, o_ent = _resolve_entity(s, index), _resolve_entity(o, index)
                    if s_ent and o_ent and s_ent.id != o_ent.id:
                        _add_or_merge(relations, s_ent, predicate, o_ent, mention)

    return list(relations.values())