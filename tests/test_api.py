from fastapi.testclient import TestClient

from app.api import app


def test_api_read_endpoints():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/graph/summary").status_code == 200
    assert client.get("/plugins").status_code == 200
    assert client.get("/knowledge/search?q=客户流失").status_code == 200
    assert client.get("/ui").status_code == 200
    assert client.get("/runs").status_code == 200
    assert client.post("/knowledge/ingest", json={"path": "data/business_material.md", "provider": "mock"}).status_code == 200


def test_read_views_never_load_model_and_sensitive_paths_are_rejected(monkeypatch):
    import app.api as api
    monkeypatch.setattr(api, "AlgorithmFactoryWorkflow", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("read view loaded LLM")))
    client = TestClient(app)
    for path in ("/reports", "/algorithms", "/capabilities", "/sources", "/catalog"):
        assert client.get(path).status_code == 200
    assert client.get("/run/not-a-run").status_code == 404
    assert client.post("/run", json={"description": "predict a binary label", "data_path": "/etc/passwd"}).status_code == 400
    from app.config import get_settings
    key_file = get_settings().project_root / "secret_test.txt"
    key_file.write_text("dummy test config")
    assert client.post("/knowledge/ingest", json={"path": "secret_test.txt"}).status_code == 400


def test_review_attachment_is_bound_to_original_report_content(tmp_path, monkeypatch):
    import hashlib, json
    import app.api as api
    from app.config import get_settings
    reports = get_settings().reports_dir
    run_id = 'abcdef012345'
    report = {'run_id': run_id, 'explanation': {'why_this_plan': 'original'}}
    (reports / f'{run_id}.json').write_text(json.dumps(report))
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    review_file = reports / f'{run_id}.explanation-review.json'
    review_file.write_text(json.dumps({'status': 'passed', 'source_report_semantic_sha256': digest, 'explanation': {'why_this_plan': 'reviewed'}}))
    result = api.run_detail(run_id)
    assert result['explanation']['why_this_plan'] == 'original'
    assert result['explanation_review']['explanation']['why_this_plan'] == 'reviewed'
    report['changed'] = True
    (reports / f'{run_id}.json').write_text(json.dumps(report))
    assert 'explanation_review' not in api.run_detail(run_id)
