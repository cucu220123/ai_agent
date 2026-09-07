from app.llm.contracts import extract_json_object


def test_extract_json_from_markdown():
    value = extract_json_object("```json\n{\"candidate_algorithms\":[\"logistic_regression\"]}\n```")
    assert value == {"candidate_algorithms": ["logistic_regression"]}


def test_bounded_literal_correction_preserves_quoted_evidence():
    assert extract_json_object('{"depth": None, "flag": True, "evidence": "None is a quoted word"}') == {"depth": None, "flag": True, "evidence": "None is a quoted word"}
    assert extract_json_object('{"x": __import__("os").system("id")}') is None
