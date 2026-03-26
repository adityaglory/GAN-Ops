# test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_endpoint_success():
    """
    Test 1: Ensure API returns 200 OK, correct number of images, 
    and the correct class label.
    """
    # Requesting 2 samples for class 1
    response = client.post("/generate?class_label=1&num_samples=2")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2
    assert "image_base64" in data["data"][0]
    assert data["data"][0]["class_label"] == 1

def test_generate_endpoint_invalid_samples():
    """
    Test 2: Ensure API handles too many requested samples.
    """
    response = client.post("/generate?class_label=0&num_samples=20")
    assert response.status_code == 400

def test_generate_endpoint_invalid_class():
    """
    Test 3: Ensure API handles non-existent class labels.
    """
    # Requesting class 99 (we only have 0, 1, 2)
    response = client.post("/generate?class_label=99&num_samples=1")
    assert response.status_code == 400