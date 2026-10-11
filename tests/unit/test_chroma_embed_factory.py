from unittest.mock import MagicMock, patch

from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function


def test_trust_remote_code_passed_top_level_not_inside_model_kwargs():
    captured: dict = {}

    def capture_init(**kw):
        captured.update(kw)
        return MagicMock()

    with patch(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
        side_effect=capture_init,
    ):
        build_chroma_embedding_function(
            {
                "backend": "sentence_transformers",
                "model_name": "dummy/model",
                "st_model_kwargs": {"trust_remote_code": True, "foo": "bar"},
            }
        )

    assert captured.get("trust_remote_code") is True
    mk = captured.get("model_kwargs") or {}
    assert "trust_remote_code" not in mk
    assert mk.get("foo") == "bar"
