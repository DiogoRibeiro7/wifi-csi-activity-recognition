"""Keep the requirements files installable.

`requirements-dev.txt` carried `pdb++>=0.10.0` for a long time. `+` is not
permitted in a PEP 508 requirement name, so `pip install -r requirements-dev.txt`
failed outright for anyone following the contributor setup, and the Dependency
Graph workflow errored on every run. CI never noticed because it installs the
project via its extras (`.[dev,docs,torch]`) rather than through these files.
"""

from pathlib import Path

import pytest

packaging_requirements = pytest.importorskip("packaging.requirements")

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILES = ["requirements.txt", "requirements-dev.txt"]


def _requirement_lines(path: Path) -> list[tuple[int, str]]:
    """Return (line number, requirement) pairs, ignoring comments and flags."""
    lines = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#")[0].strip()
        # Skip blanks and pip flags such as `-r requirements.txt`.
        if not line or line.startswith("-"):
            continue
        lines.append((number, line))
    return lines


@pytest.mark.unit
@pytest.mark.parametrize("filename", REQUIREMENTS_FILES)
def test_every_requirement_is_valid_pep508(filename: str) -> None:
    """Every pinned requirement must parse, or pip rejects the whole file."""
    path = REPO_ROOT / filename
    assert path.exists(), f"{filename} is missing"

    invalid = []
    for number, line in _requirement_lines(path):
        try:
            packaging_requirements.Requirement(line)
        except packaging_requirements.InvalidRequirement as exc:
            invalid.append(f"{filename}:{number}: {line!r} -> {exc}")

    assert not invalid, "pip cannot parse these requirements:\n" + "\n".join(invalid)


@pytest.mark.unit
@pytest.mark.parametrize("filename", REQUIREMENTS_FILES)
def test_requirement_names_are_installable(filename: str) -> None:
    """Guard the specific shape that broke: punctuation PyPI does not allow.

    PEP 503 normalizes names over ``[A-Za-z0-9._-]`` only, so a name containing
    ``+`` can never resolve to a real distribution.
    """
    path = REPO_ROOT / filename

    offenders = []
    for number, line in _requirement_lines(path):
        try:
            name = packaging_requirements.Requirement(line).name
        except packaging_requirements.InvalidRequirement:
            continue  # reported by the test above
        if not all(char.isalnum() or char in "._-" for char in name):
            offenders.append(f"{filename}:{number}: {name!r}")

    assert not offenders, (
        "these names contain characters PyPI does not permit:\n" + "\n".join(offenders)
    )
