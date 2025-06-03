from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_get_avis():
    response = client.get("/avis/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_kpi():
    response = client.get("/kpi/")
    assert response.status_code == 200
    data = response.json()
    assert 'note_moyenne' in data
    assert 'pourcentage_negatif' in data
    assert 'total_avis' in data

def test_get_motifs():
    response = client.get("/motifs/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
