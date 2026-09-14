import json

from agent.curated_memory import search_curated_memory


def test_curated_memory_fails_closed_for_missing_or_invalid_bundle(tmp_path):
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version": 2, "documents": []}', encoding="utf-8")

    assert search_curated_memory(missing, "Rodrigo") == "(memoria curada no disponible)"
    assert search_curated_memory(invalid, "Rodrigo") == "(memoria curada no disponible)"


def test_curated_memory_ranks_matches_and_honors_limit(tmp_path):
    bundle = tmp_path / "curated.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "documents": [
                    {
                        "slug": "facts/one",
                        "title": "Agenda",
                        "content": "cardiologia cardiologia control",
                        "timeline": "",
                    },
                    {
                        "slug": "facts/two",
                        "title": "Control",
                        "content": "cardiologia",
                        "timeline": "mañana",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = search_curated_memory(bundle, "cardiologia", limit=1)

    assert "[facts/one] Agenda" in result
    assert "facts/two" not in result


def test_curated_memory_rejects_empty_query(tmp_path):
    bundle = tmp_path / "curated.json"
    bundle.write_text('{"schema_version": 1, "documents": []}', encoding="utf-8")

    assert search_curated_memory(bundle, "!") == "(consulta de memoria vacía)"


def test_curated_memory_fails_closed_for_malformed_document(tmp_path):
    bundle = tmp_path / "curated.json"
    bundle.write_text(
        '{"schema_version": 1, "documents": ["not-an-object"]}',
        encoding="utf-8",
    )

    assert search_curated_memory(bundle, "agenda") == "(memoria curada no disponible)"
