"""Persistent .noema bundle format and the KnowledgeBase class.

For knowledge bases too large for a single bundle, GraphBackend is the
interface an external store implements; concrete backends live in their own
modules (noema.storage.neo4j_backend, noema.storage.rdf_backend) since each
needs an optional third-party package, mirroring noema.integrations.
"""

from noema.storage.backends import GraphBackend, read_graph, write_graph
from noema.storage.bundle import LoadedBundle, load_bundle, save_bundle
from noema.storage.knowledge_base import IngestReport, KnowledgeBase

__all__ = [
    "GraphBackend",
    "IngestReport",
    "KnowledgeBase",
    "LoadedBundle",
    "load_bundle",
    "read_graph",
    "save_bundle",
    "write_graph",
]