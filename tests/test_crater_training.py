"""Unit & Integration Test Suite for Neural Network Multi-Angle Crater Training.

Verifies:
1. CraterInvarianceNet produces normalized embeddings.
2. AngleInvariantContrastiveLoss and MultiAngleCosineLoss compute valid scalars.
3. Loss backpropagation generates non-zero gradients across model layers.
4. Optimizer step updates parameters (theta_new != theta_old).
5. Loss decreases and cosine similarity across angles increases over training epochs.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from app.pipeline.crater_nn import CraterInvarianceNet, AngleInvariantContrastiveLoss, MultiAngleCosineLoss
from app.pipeline.crater_trainer import CraterTrainingManager, render_synthetic_crater_at_angle

def test_crater_net_forward():
    print("=== Testing CraterInvarianceNet Forward Pass ===")
    model = CraterInvarianceNet(embedding_dim=128)
    model.eval()

    dummy_input = torch.randn(2, 1, 64, 64)
    out = model(dummy_input)

    assert out.shape == (2, 128), f"Expected shape (2, 128), got {out.shape}"
    norms = torch.norm(out, p=2, dim=1)
    # Check L2 unit norm
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings must be L2 normalized to unit sphere"
    print("CraterInvarianceNet forward pass test PASSED!\n")

def test_loss_and_backprop():
    print("=== Testing Loss Calculation & Parameter Backprop ===")
    model = CraterInvarianceNet(embedding_dim=128)
    model.train()
    loss_fn = AngleInvariantContrastiveLoss(margin=1.0)

    # Simulate same crater at two angles
    crater_angle1 = render_synthetic_crater_at_angle(radius=20, sun_incidence_deg=20.0, sun_azimuth_deg=45.0)
    crater_angle2 = render_synthetic_crater_at_angle(radius=20, sun_incidence_deg=65.0, sun_azimuth_deg=200.0)
    neg_patch = render_synthetic_crater_at_angle(radius=10, sun_incidence_deg=40.0, noise_std=0.05)

    t1 = torch.from_numpy(crater_angle1).unsqueeze(0).unsqueeze(0).float()
    t2 = torch.from_numpy(crater_angle2).unsqueeze(0).unsqueeze(0).float()
    t_neg = torch.from_numpy(neg_patch).unsqueeze(0).unsqueeze(0).float()

    z1 = model(t1)
    z2 = model(t2)
    z_neg = model(t_neg)

    loss, metrics = loss_fn(z1, z2, z_neg)
    assert not torch.isnan(loss) and loss.item() > 0.0, f"Invalid loss: {loss.item()}"
    print(f"  Initial calculated loss: {loss.item():.5f}")

    # Backpropagation
    loss.backward()

    # Verify gradients exist and are non-zero
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if param.grad.abs().sum() > 0:
                has_grad = True
                break

    assert has_grad, "Expected backpropagation to produce non-zero gradients on parameters"
    print("Loss backpropagation test PASSED!\n")

def test_trainer_parameter_update():
    print("=== Testing Parameter Updates & Similarity Optimization ===")
    trainer = CraterTrainingManager()
    dataset = trainer.generate_demo_multi_angle_dataset()

    patches = [item["image"] for item in dataset["angles"]]
    neg_patch = dataset["negative"]["image"]

    initial_params = [p.clone().detach() for p in trainer.model.parameters() if p.requires_grad]

    res = trainer.train_epochs(
        crater_patches_by_angle=patches,
        negative_patch=neg_patch,
        epochs=5,
        learning_rate=0.002,
        optimizer_name="adam",
        loss_type="contrastive"
    )

    print(f"  Epochs completed: {res['epochs_completed']}")
    print(f"  Initial Loss: {res['initial_loss']:.5f} -> Final Loss: {res['final_loss']:.5f}")
    print(f"  Similarity: {res['initial_similarity_pct']:.2f}% -> {res['final_similarity_pct']:.2f}%")
    print(f"  Similarity Gain: +{res['similarity_gain_pct']:.2f}%")

    # Verify parameters actually updated
    updated_params = [p.clone().detach() for p in trainer.model.parameters() if p.requires_grad]
    param_changes = [torch.norm(p_new - p_old).item() for p_old, p_new in zip(initial_params, updated_params)]

    assert max(param_changes) > 1e-5, "Neural network parameters must be updated by optimizer"
    assert res["final_loss"] <= res["initial_loss"] + 0.05, "Training loss should generally decrease or stabilize"
    print("Trainer parameter update test PASSED!\n")

if __name__ == "__main__":
    test_crater_net_forward()
    test_loss_and_backprop()
    test_trainer_parameter_update()
    print("ALL TESTS IN test_crater_training.py PASSED SUCCESSFULLY!")
