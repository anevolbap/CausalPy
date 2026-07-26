#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Tests for resource-safe documentation notebook selection."""

from pathlib import Path

import pytest

from scripts.run_notebooks import runner


@pytest.fixture
def notebook_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    gallery = tmp_path / "docs" / "source" / "notebooks"
    knowledgebase = tmp_path / "docs" / "source" / "knowledgebase"
    gallery.mkdir(parents=True)
    knowledgebase.mkdir(parents=True)
    for path in (
        gallery / "alpha-pymc.ipynb",
        gallery / "beta-sklearn.ipynb",
        gallery / "other.ipynb",
        gallery / "skipped.ipynb",
        knowledgebase / "concepts.ipynb",
    ):
        path.touch()

    collections = {"gallery": gallery, "knowledgebase": knowledgebase}
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "NOTEBOOK_COLLECTIONS", collections)
    monkeypatch.setattr(runner, "SKIP_NOTEBOOKS", {"gallery/skipped.ipynb"})
    return collections


def test_default_selection_uses_gallery_and_mock_skip_policy(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks()

    assert [notebook.name for notebook in notebooks] == [
        "alpha-pymc.ipynb",
        "beta-sklearn.ipynb",
        "other.ipynb",
    ]


def test_named_knowledgebase_collection_is_discoverable(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(collections=["knowledgebase"])

    assert [notebook.name for notebook in notebooks] == ["concepts.ipynb"]


def test_multiple_exact_notebooks_can_cross_collections(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(
        notebook_paths=[
            "docs/source/knowledgebase/concepts.ipynb",
            "docs/source/notebooks/alpha-pymc.ipynb",
        ]
    )

    assert notebooks == [
        notebook_collections["knowledgebase"] / "concepts.ipynb",
        notebook_collections["gallery"] / "alpha-pymc.ipynb",
    ]


def test_exact_mock_incompatible_notebook_requires_full_mode(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="Use --full"):
        runner.get_notebooks(notebook_paths=["docs/source/notebooks/skipped.ipynb"])

    notebooks = runner.get_notebooks(
        notebook_paths=["docs/source/notebooks/skipped.ipynb"], full=True
    )
    assert [notebook.name for notebook in notebooks] == ["skipped.ipynb"]


def test_collection_full_mode_does_not_bypass_skip_policy(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(full=True)

    assert "skipped.ipynb" not in [notebook.name for notebook in notebooks]


def test_exact_notebook_must_be_inside_documentation_collections(
    notebook_collections: dict[str, Path],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.ipynb"
    outside.touch()

    with pytest.raises(ValueError, match="outside configured"):
        runner.get_notebooks(notebook_paths=[str(outside)])


def test_pattern_and_exclusion_filters_compose(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(pattern="*-pymc.ipynb")
    assert [notebook.name for notebook in notebooks] == ["alpha-pymc.ipynb"]

    notebooks = runner.get_notebooks(exclude_patterns=["pymc"])
    assert [notebook.name for notebook in notebooks] == [
        "beta-sklearn.ipynb",
        "other.ipynb",
    ]

    notebooks = runner.get_notebooks(exclude_patterns=["pymc", "sklearn"])
    assert [notebook.name for notebook in notebooks] == ["other.ipynb"]


def test_workflow_shards_are_exhaustive_and_non_overlapping(
    notebook_collections: dict[str, Path],
) -> None:
    shards = [
        runner.get_notebooks(pattern="*-pymc.ipynb"),
        runner.get_notebooks(pattern="*-sklearn.ipynb"),
        runner.get_notebooks(exclude_patterns=["pymc", "sklearn"]),
        runner.get_notebooks(collections=["knowledgebase"]),
    ]
    selected = [notebook for shard in shards for notebook in shard]
    expected = [
        notebook_collections["gallery"] / "alpha-pymc.ipynb",
        notebook_collections["gallery"] / "beta-sklearn.ipynb",
        notebook_collections["gallery"] / "other.ipynb",
        notebook_collections["knowledgebase"] / "concepts.ipynb",
    ]

    assert len(selected) == len(set(selected))
    assert sorted(selected) == sorted(expected)


def test_explicit_notebooks_reject_other_selectors(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        runner.get_notebooks(
            collections=["gallery"],
            notebook_paths=["docs/source/notebooks/alpha-pymc.ipynb"],
        )


def test_collection_symlink_cannot_escape_allowed_root(
    notebook_collections: dict[str, Path],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.ipynb"
    outside.touch()
    (notebook_collections["gallery"] / "escape.ipynb").symlink_to(outside)

    with pytest.raises(ValueError, match="outside configured"):
        runner.get_notebooks()


def test_main_list_mode_never_executes_notebooks(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        runner,
        "run_notebook",
        lambda path, *, full=False: executed.append((path, full)),
    )

    assert runner.main(["--list"]) == 0
    assert executed == []


def test_main_executes_exact_notebooks_serially_and_forwards_full(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        runner,
        "run_notebook",
        lambda path, *, full=False: executed.append((path, full)),
    )

    assert (
        runner.main(
            [
                "--full",
                "--notebook",
                "docs/source/notebooks/alpha-pymc.ipynb",
                "--notebook",
                "docs/source/knowledgebase/concepts.ipynb",
            ]
        )
        == 0
    )
    assert executed == [
        (notebook_collections["gallery"] / "alpha-pymc.ipynb", True),
        (notebook_collections["knowledgebase"] / "concepts.ipynb", True),
    ]


def test_main_rejects_empty_selection(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="No notebooks matched"):
        runner.main(["--pattern", "does-not-exist-*.ipynb"])


def test_parallel_option_is_not_supported(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--parallel"])


def test_unknown_collection_is_rejected(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="Unknown notebook collection"):
        runner.get_notebooks(collections=["unknown"])
