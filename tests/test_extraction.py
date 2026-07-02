import pytest

from gamegen.extraction import (ExtractionError, extract_json,
                                extract_labeled_python, extract_markdown)


def test_takes_last_json_block():
    text = (
        "thinking...\n```json\n{\"draft\": 1}\n```\nmore thinking\n"
        "```json\n{\"final\": 2}\n```\n"
    )
    assert extract_json(text) == {"final": 2}


def test_unlabeled_fallback():
    assert extract_json("```\n{\"a\": 1}\n```") == {"a": 1}


def test_no_block_raises():
    with pytest.raises(ExtractionError, match="fenced"):
        extract_json("just prose, no fences")


def test_invalid_json_raises_with_hint():
    with pytest.raises(ExtractionError, match="not valid JSON"):
        extract_json("```json\n{'single': 'quotes',}\n```")


def test_labeled_python_sections():
    text = (
        "reasoning\n### ENGINE\n```python\nclass E:\n    pass\n```\n"
        "### TESTS\n```python\ndef test_x():\n    assert True\n```\n"
    )
    out = extract_labeled_python(text, ("ENGINE", "TESTS"))
    assert out["ENGINE"].startswith("class E")
    assert out["TESTS"].startswith("def test_x")


def test_missing_section_raises():
    with pytest.raises(ExtractionError, match="TESTS"):
        extract_labeled_python(
            "### ENGINE\n```python\nx = 1\n```", ("ENGINE", "TESTS")
        )


def test_markdown_block_or_whole_text():
    assert extract_markdown("```markdown\n# Title\n```") == "# Title"
    assert extract_markdown("# Plain\n\nbody") == "# Plain\n\nbody"
