"""Smoke tests: package skeleton imports and project metadata sanity."""


def test_packages_importable():
    import ai  # noqa: F401
    import ai.ingestion  # noqa: F401
    import ai.reasoning  # noqa: F401
    import ai.retrieval  # noqa: F401
    import ai.speech  # noqa: F401
    import ai.translate  # noqa: F401
    import ai.verification  # noqa: F401
    import backend  # noqa: F401


def test_project_metadata():
    import tomllib
    from pathlib import Path

    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert data["project"]["name"] == "setu"
    assert data["project"]["requires-python"] == ">=3.12,<3.13"
