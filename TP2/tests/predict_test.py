import requests
host = "127.0.0.1"
port = "5027"
url_format = "http://{host}:{port}/{endpoint}"

def test_predict():
    data = {
        "data": [5.1, 3.5, 1.4, 0.2]
    }

    response = requests.post(url_format.format(host=host, port=port, endpoint="predict"), json=data)

    assert response.status_code == 200 
    assert response.headers["Content-Type"] == "application/json"

    data = response.json()  
    assert isinstance(data, dict)
    assert "y_pred" in data
    assert data["y_pred"] in [[0], [1], [2], [3]]

def test_load_model():
    model_name = "tracking-quickstart"
    model_version = "4"

    data = {
        "model_name": model_name,
        "model_version": model_version
    }

    response = requests.post(url_format.format(host=host, port=port, endpoint="update-model"), json=data)

    assert response.status_code == 200 
    assert response.headers["Content-Type"] == "application/json"

    data = response.json()  
    assert isinstance(data, dict)
    assert "message" in data
    assert data["message"] == "Modèle mis à jour vers models:/tracking-quickstart/4"

    data = {
        "data": [5.7, 3.8, 1.7, 0.3]
    }

    response = requests.post(url_format.format(host=host, port=port, endpoint="predict"), json=data)
    data = response.json()  

    assert response.status_code == 200 
    assert "y_pred" in data
    assert data["y_pred"] == [2]
