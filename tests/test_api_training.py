"""Direct Function Test Suite for FastAPI Training & Inference Endpoints."""
import sys, os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import (
    health_check,
    get_training_status,
    get_sample_training_dataset,
    train_neural_network,
    reset_neural_weights
)

async def run_tests():
    print("=== Testing API Health ===")
    h = health_check()
    assert "neural_trainer" in h
    print("  Health endpoint OK:", h["neural_trainer"])

    print("=== Testing Training Status ===")
    status = get_training_status()
    print("  Status:", status["model_name"], f"({status['total_parameters']} params)")
    assert status["total_parameters"] > 0

    print("=== Testing Sample Training Dataset ===")
    sample_response = get_sample_training_dataset()
    import json
    sample_data = json.loads(sample_response.body.decode("utf-8"))
    assert len(sample_data["angles"]) == 4
    print(f"  Received {len(sample_data['angles'])} angles for demo crater dataset")

    print("=== Testing Train Neural Network Function ===")
    train_response = await train_neural_network(
        epochs=3,
        learning_rate=0.001,
        loss_type="contrastive",
        optimizer="adam",
        files=None
    )
    train_data = json.loads(train_response.body.decode("utf-8"))
    print(f"  Training completed: {train_data['epochs_completed']} epochs, initial loss: {train_data['initial_loss']:.4f} -> final loss: {train_data['final_loss']:.4f}")
    print(f"  Similarity gain: +{train_data['similarity_gain_pct']}%")
    assert train_data["epochs_completed"] == 3

    print("=== Testing Reset Function ===")
    reset_res = reset_neural_weights()
    print("  Reset response:", reset_res)
    assert reset_res["status"] == "success"

    print("ALL API TRAINING DIRECT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
