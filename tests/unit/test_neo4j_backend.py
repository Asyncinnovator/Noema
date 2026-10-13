"""Tests for noema.storage.neo4j_backend.Neo4jBackend.

No real Neo4j server is reachable in this sandbox, so the `neo4j` driver
itself is mocked throughout -- these tests verify the Cypher-building and
row-mapping logic, not a live database.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from noema.schema import Entity, EntityType, Event, Relation
from noema.storage import neo4j_backend


@pytest.fixture
def mock_driver_module():
    with patch.object(neo4j_backend, "_neo4j") as mock_module:
        mock_module.GraphDatabase.driver.return_value = MagicMock()
        yield mock_module


@pytest.fixture
def backend(mock_driver_module):
    return neo4j_backend.Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "pw"))


def _mock_session(backend) -> MagicMock:
    """The mocked session context manager returned by backend._session()."""
    return backend._driver.session.return_value.__enter__.return_value


class TestConstruction:
    def test_missing_driver_package_raises_with_install_hint(self):
        with (
            patch.object(neo4j_backend, "_neo4j", None),
            pytest.raises(ImportError, match="neo4j"),
        ):
            neo4j_backend.Neo4jBackend("bolt://localhost:7687")

    def test_driver_created_with_given_uri_and_auth(self, mock_driver_module):
        neo4j_backend.Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "pw"))
        mock_driver_module.GraphDatabase.driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "pw")
        )


class TestUpsertEntities:
    def test_empty_list_does_not_open_a_session(self, backend):
        backend.upsert_entities([])
        backend._driver.session.assert_not_called()

    def test_entities_are_sent_as_one_batched_query(self, backend):
        entities = [
            Entity(name="Larry Page", type=EntityType.PERSON),
            Entity(name="Google", type=EntityType.ORG),
        ]
        session = _mock_session(backend)

        backend.upsert_entities(entities)

        session.run.assert_called_once()
        (query,), kwargs = session.run.call_args
        assert "MERGE" in query
        assert len(kwargs["rows"]) == 2
        assert kwargs["rows"][0]["name"] == "Larry Page"


class TestUpsertRelations:
    def test_relation_props_include_subject_and_object_id(self, backend):
        relation = Relation(subject_id="ent_a", predicate="found", object_id="ent_b")
        session = _mock_session(backend)

        backend.upsert_relations([relation])

        _, kwargs = session.run.call_args
        assert kwargs["rows"][0]["subject_id"] == "ent_a"
        assert kwargs["rows"][0]["object_id"] == "ent_b"
        assert kwargs["rows"][0]["predicate"] == "found"

    def test_empty_list_does_not_open_a_session(self, backend):
        backend.upsert_relations([])
        backend._driver.session.assert_not_called()


class TestUpsertEvents:
    def test_event_params_include_participant_ids(self, backend):
        event = Event(label="ipo", participant_ids=["ent_a", "ent_b"], time="2004-08-19")
        session = _mock_session(backend)

        backend.upsert_events([event])

        _, kwargs = session.run.call_args
        assert kwargs["participant_ids"] == ["ent_a", "ent_b"]
        assert kwargs["label"] == "ipo"


class TestGetEntity:
    def test_found_record_is_mapped_to_entity(self, backend):
        session = _mock_session(backend)
        record = {
            "e": {
                "id": "ent_1",
                "name": "Google",
                "type": "ORG",
                "aliases": ["Google Inc."],
                "confidence": 1.0,
                "mentions": [],
            }
        }
        session.run.return_value.single.return_value = record

        entity = backend.get_entity("ent_1")

        assert entity is not None
        assert entity.name == "Google"
        assert entity.type == EntityType.ORG

    def test_missing_record_returns_none(self, backend):
        session = _mock_session(backend)
        session.run.return_value.single.return_value = None

        assert backend.get_entity("ent_missing") is None


class TestAllEntities:
    def test_iterates_every_row(self, backend):
        session = _mock_session(backend)
        session.run.return_value = [
            {"e": {"id": "ent_1", "name": "Larry Page", "type": "PERSON",
                    "aliases": [], "confidence": 1.0, "mentions": []}},
            {"e": {"id": "ent_2", "name": "Google", "type": "ORG",
                    "aliases": [], "confidence": 1.0, "mentions": []}},
        ]

        entities = list(backend.all_entities())

        assert [e.name for e in entities] == ["Larry Page", "Google"]


class TestAllRelations:
    def test_iterates_every_row(self, backend):
        session = _mock_session(backend)
        session.run.return_value = [
            {
                "subject_id": "ent_a",
                "object_id": "ent_b",
                "rel": {"id": "rel_1", "predicate": "found", "confidence": 1.0, "mentions": []},
            }
        ]

        relations = list(backend.all_relations())

        assert len(relations) == 1
        assert relations[0].subject_id == "ent_a"
        assert relations[0].predicate == "found"


class TestEntityCount:
    def test_returns_count_from_query(self, backend):
        session = _mock_session(backend)
        session.run.return_value.single.return_value = {"n": 42}
        assert backend.entity_count() == 42

    def test_no_record_returns_zero(self, backend):
        session = _mock_session(backend)
        session.run.return_value.single.return_value = None
        assert backend.entity_count() == 0


class TestClearAndClose:
    def test_clear_runs_delete_query(self, backend):
        session = _mock_session(backend)
        backend.clear()
        (query,), _ = session.run.call_args
        assert "DETACH DELETE" in query

    def test_close_closes_the_driver(self, backend):
        backend.close()
        backend._driver.close.assert_called_once()