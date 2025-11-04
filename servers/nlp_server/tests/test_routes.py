# tests/test_routes.py
from fastapi.testclient import TestClient
import pytest
from main import app

client = TestClient(app)

class TestRoutes:
    def test_health(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_annotate_success(self):
        data = {"texts": ["Hello world.", "Jet is a software developer."]}
        resp = client.post("/api/v1/annotate", json=data)
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) == 2

    def test_annotate_bad_request(self):
        data = {"texts": []}
        resp = client.post("/api/v1/annotate", json=data)
        assert resp.status_code == 400