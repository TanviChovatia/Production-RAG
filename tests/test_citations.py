from rag_chain import citation_coverage, enforce_citations, extract_citations


class DummyDoc:
    metadata = {}
    page_content = "Example"


def test_extract_citations() -> None:
    assert extract_citations("Paris is in France. [S1] It is a capital. [S2]") == ["S1", "S2"]


def test_enforce_citations_adds_fallback() -> None:
    text = enforce_citations("A supported answer.", [DummyDoc()])
    assert "[S1]" in text


def test_citation_coverage() -> None:
    value = citation_coverage("A fact. [S1] Another fact. [S2]")
    assert value == 1.0
