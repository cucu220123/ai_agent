from app.knowledge.extractor import CapabilityExtractor


def test_extract_markdown_and_python(tmp_path):
    md = tmp_path / "material.md"
    md.write_text("# Churn\n| age | number |\nROC-AUC and F1", encoding="utf-8")
    result = CapabilityExtractor().extract_markdown(md)
    assert result["headings"] == ["Churn"]
    assert "ROC-AUC" in result["metrics"]
    py = tmp_path / "algorithm.py"
    py.write_text("def train(a,b): pass\ndef predict(a,b): pass\ndef evaluate(a,b,c): pass", encoding="utf-8")
    assert CapabilityExtractor().extract_python(py)["has_algorithm_interface"]

