"""Regression tests for crash-safe saves (H4) and hardened bundle loading (H5).

The embedding model is mocked (huggingface.co isn't reachable in CI-less
sandboxes); these tests are about file handling and validation, not
embedding quality.
"""

from __future__ import annotations

import glob
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from typer.testing import CliRunner

import noema
from noema.cli import app
from noema.graph.builder import SemanticGraph
from noema.schema import Entity, EntityType, Relation
from noema.storage import bundle
from noema.storage._atomic import atomic_write
from noema.storage.bundle import BundleError
from noema.storage.knowledge_base import KnowledgeBase


class _FakeEmbeddingModel:
    def __init__(self, dim: int = 6) -> None:
        self._dim = dim

    def encode(self, texts, normalize_embeddings=True):
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(len(texts), self._dim)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def _graph() -> SemanticGraph:
    graph = SemanticGraph()
    page = Entity(name="Larry Page", type=EntityType.PERSON)
    google = Entity(name="Google", type=EntityType.ORG)
    graph.add_entity(page)
    graph.add_entity(google)
    graph.add_relation(Relation(subject_id=page.id, predicate="found", object_id=google.id))
    return graph


@pytest.fixture
def good_bundle(tmp_path) -> str:
    path = str(tmp_path / "kb.noema")
    with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
        bundle.save_bundle(path, _graph())
    return path


def _parts(path: str) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def _rewrite(path: str, **overrides) -> str:
    """Copy `path` to a sibling bundle with some entries replaced (bytes) or
    removed (None). Manifest overrides can be passed as a dict via `manifest`.
    """
    parts = _parts(path)
    manifest_patch = overrides.pop("manifest", None)
    if manifest_patch is not None:
        manifest = json.loads(parts["manifest.json"])
        manifest.update(manifest_patch)
        parts["manifest.json"] = json.dumps(manifest).encode()
    for name, data in overrides.items():
        entry = name.replace("_", ".", 1)
        if data is None:
            parts.pop(entry, None)
        else:
            parts[entry] = data
    out = path + ".tampered"
    with zipfile.ZipFile(out, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return out


# -- H4: atomic writes ----------------------------------------------------


class TestAtomicWrite:
    def test_success_replaces_destination(self, tmp_path):
        target = tmp_path / "f.bin"
        target.write_bytes(b"old")
        with atomic_write(target) as fh:
            fh.write(b"new")
        assert target.read_bytes() == b"new"
        assert glob.glob(str(tmp_path / "*.tmp")) == []

    def test_exception_leaves_original_untouched_and_cleans_up(self, tmp_path):
        target = tmp_path / "f.bin"
        target.write_bytes(b"old")
        with pytest.raises(RuntimeError), atomic_write(target) as fh:
            fh.write(b"partial")
            raise RuntimeError("boom")
        assert target.read_bytes() == b"old"
        assert glob.glob(str(tmp_path / "*.tmp")) == []

    def test_exception_with_no_existing_file_creates_nothing(self, tmp_path):
        target = tmp_path / "f.bin"
        with pytest.raises(RuntimeError), atomic_write(target) as fh:
            fh.write(b"partial")
            raise RuntimeError("boom")
        assert list(tmp_path.iterdir()) == []


class TestSaveBundleIsAtomic:
    def test_failure_while_serializing_keeps_previous_bundle_loadable(self, good_bundle):
        before = Path(good_bundle).read_bytes()
        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()), patch.object(
            bundle.faiss, "serialize_index", side_effect=RuntimeError("disk full")
        ), pytest.raises(RuntimeError):
            bundle.save_bundle(good_bundle, _graph())
        assert Path(good_bundle).read_bytes() == before
        assert len(bundle.load_bundle(good_bundle).graph.entities) == 2

    def test_failure_midway_through_the_zip_keeps_previous_bundle_and_leaves_no_temp(
        self, good_bundle
    ):
        before = Path(good_bundle).read_bytes()
        real_writestr = zipfile.ZipFile.writestr
        calls = {"n": 0}

        def flaky(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("no space left on device")
            return real_writestr(self, *a, **k)

        with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()), patch.object(
            zipfile.ZipFile, "writestr", flaky
        ), pytest.raises(OSError):
            bundle.save_bundle(good_bundle, _graph())
        assert Path(good_bundle).read_bytes() == before
        assert glob.glob(good_bundle + ".*.tmp") == []

    def test_checkpoint_survives_a_failed_later_save(self, good_bundle):
        kb = KnowledgeBase.load(good_bundle)
        with patch.object(
            bundle.faiss, "serialize_index", side_effect=RuntimeError("crash")
        ), pytest.raises(RuntimeError):
            kb.save(good_bundle)
        assert len(KnowledgeBase.load(good_bundle).graph.entities) == 2


class TestRDFSaveIsAtomic:
    def test_unserializable_predicate_does_not_destroy_existing_file(self, tmp_path):
        pytest.importorskip("rdflib")
        from noema.storage.rdf_backend import RDFBackend

        path = str(tmp_path / "kg.ttl")
        graph = _graph()
        backend = RDFBackend(path=path)
        backend.upsert_entities(graph.entities)
        backend.upsert_relations(graph.relations)
        backend.save()
        assert len(list(RDFBackend(path=path).all_entities())) == 2

        bad = Relation(
            subject_id=graph.entities[0].id, predicate="has ceo <x>", object_id=graph.entities[1].id
        )
        backend.upsert_relations([bad])
        with pytest.raises(Exception):  # noqa: B017 - rdflib's own serializer error
            backend.save()

        assert len(list(RDFBackend(path=path).all_entities())) == 2
        assert glob.glob(path + ".*.tmp") == []


# -- H5: hardened loading -------------------------------------------------


class TestLoadBundleValidation:
    def test_good_bundle_still_loads(self, good_bundle):
        loaded = bundle.load_bundle(good_bundle)
        assert loaded.index.ntotal == len(loaded.fact_ids) == 1

    def test_not_a_zip_raises_bundle_error(self, tmp_path):
        path = tmp_path / "junk.noema"
        path.write_bytes(b"this is not a zip")
        with pytest.raises(BundleError):
            bundle.load_bundle(str(path))

    @pytest.mark.parametrize("missing", ["graph_json", "manifest_json", "embeddings_bin", "index_faiss"])
    def test_missing_entry_raises_bundle_error(self, good_bundle, missing):
        tampered = _rewrite(good_bundle, **{missing: None})
        with pytest.raises(BundleError, match="missing entry"):
            bundle.load_bundle(tampered)

    def test_decompression_bomb_is_rejected_before_it_is_read(self, tmp_path, good_bundle):
        bomb = str(tmp_path / "bomb.noema")
        parts = _parts(good_bundle)
        with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("graph.json", b"[" + b"0," * (5 * 1024 * 1024) + b"0]")  # ~10 MB of JSON
            for name in ("manifest.json", "embeddings.bin", "index.faiss"):
                z.writestr(name, parts[name])
        assert os.path.getsize(bomb) < 100_000
        with pytest.raises(BundleError, match="limit"):
            bundle.load_bundle(bomb, max_json_bytes=1024 * 1024)

    def test_oversized_binary_entry_is_rejected(self, good_bundle):
        with pytest.raises(BundleError, match="limit"):
            bundle.load_bundle(good_bundle, max_binary_bytes=10)

    def test_invalid_json_raises_bundle_error(self, good_bundle):
        with pytest.raises(BundleError, match="not valid JSON"):
            bundle.load_bundle(_rewrite(good_bundle, graph_json=b"{not json"))

    def test_json_that_is_not_an_object_raises_bundle_error(self, good_bundle):
        with pytest.raises(BundleError, match="JSON object"):
            bundle.load_bundle(_rewrite(good_bundle, graph_json=b"[1, 2, 3]"))

    def test_malformed_graph_raises_bundle_error(self, good_bundle):
        tampered = _rewrite(good_bundle, graph_json=json.dumps({"entities": [{"nope": 1}]}).encode())
        with pytest.raises(BundleError, match="graph.json"):
            bundle.load_bundle(tampered)

    @pytest.mark.parametrize(
        "patch_",
        [
            {"schema_version": "999"},
            {"schema_version": None},
            {"embedding_dim": 0},
            {"embedding_dim": -4},
            {"embedding_dim": 10**9},
            {"embedding_dim": "6"},
            {"embedding_dim": True},
            {"fact_ids": "rel_a"},
            {"fact_ids": [1, 2]},
            {"embedding_model": 5},
            {"spacy_model": ["x"]},
        ],
    )
    def test_invalid_manifest_values_raise_bundle_error(self, good_bundle, patch_):
        with pytest.raises(BundleError):
            bundle.load_bundle(_rewrite(good_bundle, manifest=patch_))

    def test_fact_ids_claiming_more_facts_than_the_index_holds(self, good_bundle):
        # The tampered bundle from the audit: 3 fact_ids, empty index.
        tampered = _rewrite(good_bundle, manifest={"fact_ids": ["rel_a", "rel_b", "rel_c"]})
        with pytest.raises(BundleError):
            bundle.load_bundle(tampered)

    def test_duplicate_fact_ids_rejected(self, good_bundle):
        fid = json.loads(_parts(good_bundle)["manifest.json"])["fact_ids"][0]
        with pytest.raises(BundleError, match="duplicates"):
            bundle.load_bundle(_rewrite(good_bundle, manifest={"fact_ids": [fid, fid]}))

    def test_embeddings_length_must_match_manifest(self, good_bundle):
        with pytest.raises(BundleError, match="embeddings.bin"):
            bundle.load_bundle(_rewrite(good_bundle, embeddings_bin=b"\x00" * 7))

    def test_index_dimension_must_match_manifest(self, good_bundle):
        wrong = faiss.serialize_index(faiss.IndexFlatIP(3)).tobytes()
        with pytest.raises(BundleError, match="index.faiss"):
            bundle.load_bundle(_rewrite(good_bundle, index_faiss=wrong))

    def test_non_inner_product_index_rejected(self, good_bundle):
        dim = json.loads(_parts(good_bundle)["manifest.json"])["embedding_dim"]
        wrong = faiss.serialize_index(faiss.IndexFlatL2(dim)).tobytes()
        with pytest.raises(BundleError, match="inner-product"):
            bundle.load_bundle(_rewrite(good_bundle, index_faiss=wrong))

    def test_garbage_index_bytes_raise_bundle_error(self, good_bundle):
        with pytest.raises(BundleError):
            bundle.load_bundle(_rewrite(good_bundle, index_faiss=b"garbage bytes"))

    def test_bundle_error_is_a_value_error_and_exported(self):
        assert issubclass(BundleError, ValueError)
        assert noema.BundleError is BundleError


# -- H5: model names read from a file are not blindly followed -------------


def _saved_kb(tmp_path, **kb_kwargs) -> str:
    kb = KnowledgeBase(**kb_kwargs)
    kb._graph = _graph()
    path = str(tmp_path / "kb.noema")
    with patch.object(bundle, "_load_model", return_value=_FakeEmbeddingModel()):
        kb.save(path)
    return path


class TestLoadDoesNotFollowManifestModels:
    def test_default_models_round_trip_unchanged(self, tmp_path):
        kb = KnowledgeBase.load(_saved_kb(tmp_path))
        assert kb._embedding_model == bundle.DEFAULT_EMBEDDING_MODEL
        assert kb._spacy_model == "en_core_web_sm"

    def test_unknown_embedding_model_in_manifest_is_refused(self, tmp_path):
        path = _saved_kb(tmp_path, embedding_model="attacker/some-hf-repo")
        with pytest.raises(BundleError, match="embedding_model="):
            KnowledgeBase.load(path)

    def test_explicit_embedding_model_is_accepted(self, tmp_path):
        path = _saved_kb(tmp_path, embedding_model="my-org/my-model")
        kb = KnowledgeBase.load(path, embedding_model="my-org/my-model")
        assert kb._embedding_model == "my-org/my-model"

    def test_trust_flag_restores_the_manifest_embedding_model(self, tmp_path):
        path = _saved_kb(tmp_path, embedding_model="my-org/my-model")
        kb = KnowledgeBase.load(path, trust_manifest_models=True)
        assert kb._embedding_model == "my-org/my-model"

    @pytest.mark.parametrize("evil", ["/tmp/evil_pipeline", "../x", "https://evil.example/m", "evil.module"])
    def test_non_package_spacy_name_is_ignored_with_a_warning(self, tmp_path, evil):
        path = _saved_kb(tmp_path, spacy_model=evil)
        with pytest.warns(UserWarning, match="Ignoring spaCy model"):
            kb = KnowledgeBase.load(path)
        assert kb._spacy_model == "en_core_web_sm"

    @pytest.mark.parametrize("name", ["en_core_web_md", "de_dep_news_trf", "xx_ent_wiki_sm"])
    def test_package_style_spacy_name_is_still_restored(self, tmp_path, name):
        kb = KnowledgeBase.load(_saved_kb(tmp_path, spacy_model=name))
        assert kb._spacy_model == name

    def test_explicit_spacy_model_always_wins(self, tmp_path):
        path = _saved_kb(tmp_path, spacy_model="/tmp/evil_pipeline")
        kb = KnowledgeBase.load(path, spacy_model="en_core_web_lg")
        assert kb._spacy_model == "en_core_web_lg"

    def test_trust_flag_restores_a_non_package_spacy_name(self, tmp_path):
        path = _saved_kb(tmp_path, spacy_model="/opt/my_pipeline")
        kb = KnowledgeBase.load(path, trust_manifest_models=True)
        assert kb._spacy_model == "/opt/my_pipeline"


# -- CLI ------------------------------------------------------------------


class TestCliBundleErrors:
    runner = CliRunner()

    def test_inspect_on_corrupt_bundle_exits_cleanly(self, tmp_path):
        path = tmp_path / "bad.noema"
        path.write_bytes(b"not a zip")
        result = self.runner.invoke(app, ["inspect", str(path)])
        assert result.exit_code == 1
        assert "Error" in result.output
        assert "Traceback" not in result.output

    def test_query_on_untrusted_embedding_model_exits_cleanly(self, tmp_path):
        path = _saved_kb(tmp_path, embedding_model="attacker/some-hf-repo")
        result = self.runner.invoke(app, ["query", path, "who founded google?"])
        assert result.exit_code == 1
        assert "--trust-bundle-models" not in result.output or "embedding_model" in result.output
        assert "Traceback" not in result.output

    def test_query_output_is_printed_literally_not_as_rich_markup(self, tmp_path):
        path = tmp_path / "kb.noema"
        path.write_bytes(b"placeholder")
        fake_kb = MagicMock()
        fake_kb.retrieve.return_value = SimpleNamespace(
            text="#1 Foo [/bar] :ORG", token_count=3, truncated=False
        )
        with patch.object(noema.KnowledgeBase, "load", return_value=fake_kb):
            result = self.runner.invoke(app, ["query", str(path), "q"])
        assert result.exit_code == 0, result.output
        assert "[/bar]" in result.output