import pytest
from casiroli.app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    params={"Rank":"platinum-platinum",
            "Player 1":"Marth",
            "Player 2":"Fox",
            "Stage":"Pokémon Stadium",
            "Winner":"Player 1",
            "Rank_Difference":0,
            "P1_Rank_Value": 2,
            "P2_Rank_Value": 2}
    response = client.post("/infer", json={"name": "Alessandro", "param1":params})
    assert response.status_code == 200
    data = response.get_json()
    assert data["message"] == "Hello Alessandro!"
    assert data["inference"]== ["Player 1"]