"""Text -> entities/relations extraction."""

from noema.parser.entities import extract_entities
from noema.parser.relations import extract_relations

__all__ = ["extract_entities", "extract_relations"]