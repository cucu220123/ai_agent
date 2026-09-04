from app.llm.contracts import extract_json_object


def test_extract_json_from_markdown():
    value = extract_json_object("```json\n{\"candidate_algorithms\":[\"logistic_regression\"]}\n```")
    assert value == {"candidate_algorithms": ["logistic_regression"]}

