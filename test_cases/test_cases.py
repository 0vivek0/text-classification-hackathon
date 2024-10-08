import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_check_api_health():
    response = client.get("/check")
    assert response.status_code == 200
    assert response.json() == {"status": "UP"}

def test_get_scope_for_given_query_success():
    response = client.post("/get_scope_for_given_query", data={"saved_model_name": "test_model", "query": "Sample query"})
    assert response.status_code == 200
    assert "Given query is Predicted Scope" in response.text  # Adjust based on expected output

def test_get_scope_for_given_query_empty_model_name():
    response = client.post("/get_scope_for_given_query", data={"saved_model_name": "", "query": "Sample query"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Model name cannot be empty."}

def test_get_scope_for_given_query_empty_query():
    response = client.post("/get_scope_for_given_query", data={"saved_model_name": "test_model", "query": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query cannot be empty."}

def test_get_scope_for_given_query_failure():
    import unittest.mock as mock
    with mock.patch('model_training_invocation.get_scope_for_query', side_effect=Exception("Error in prediction")):
        response = client.post("/get_scope_for_given_query", data={"saved_model_name": "test_model", "query": "Sample query"})
        assert response.status_code == 500
        assert "Failed to get the scope for given query" in response.json()["error"]
