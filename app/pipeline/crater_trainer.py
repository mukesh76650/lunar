"""Lunar Crater Multi-Angle Neural Network Trainer.

Coordinates:
1. Multi-angle crater dataset synthesis and extraction (different solar incidence, azimuth, and tilts).
2. PyTorch forward pass, loss calculation (angle invariance), backpropagation, and parameter updates.
3. Telemetry tracking: loss curves, gradient norms, parameter weight update deltas, and similarity progression.
4. Feature map and descriptor activation rendering for dashboard visualizer.
"""
import os
import copy
import time
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional

from app.pipeline.crater_nn import (
    CraterInvarianceNet,
    AngleInvariantContrastiveLoss,
    MultiAngleCosineLoss
)

def render_synthetic_crater_at_angle(
    radius: int = 24,
    patch_size: int = 64,
    sun_incidence_deg: float = 45.0,
    sun_azimuth_deg: float = 45.0,
    tilt_deg: float = 0.0,
    noise_std: float = 0.02
) -> np.ndarray:
    """Render a physically grounded lunar impact crater viewed under a specific solar incidence and azimuth angle.
    
    Args:
        radius: Crater rim radius in pixels.
        patch_size: Square output patch dimension.
        sun_incidence_deg: Sun angle above horizon / zenith angle (0 = overhead, 90 = horizon).
        sun_azimuth_deg: Compass direction of incident sunlight in degrees (0 = from North, 90 = East, etc.).
        tilt_deg: Camera viewing slant / perspective tilt angle in degrees.
        noise_std: Lunar regolith speckle noise.
    """
    h = w = patch_size
    cx = w / 2.0
    cy = h / 2.0
    y, x = np.mgrid[0:h, 0:w]

    # Regolith base albedo
    patch = np.ones((h, w), dtype=np.float32) * 0.42

    # Perspective tilt matrix if requested
    if abs(tilt_deg) > 1e-3:
        # Skew/stretch x or y
        stretch_y = np.cos(np.radians(tilt_deg))
        y_eff = (y - cy) / max(0.2, stretch_y) + cy
    else:
        y_eff = y

    dist = np.sqrt((x - cx) ** 2 + (y_eff - cy) ** 2)

    # 1. Interior bowl depression
    inner_mask = dist < radius
    depth_profile = (1.0 - (dist / radius) ** 2)
    patch[inner_mask] -= 0.32 * depth_profile[inner_mask]

    # 2. Raised rim wall
    rim_mask = (dist >= radius * 0.75) & (dist <= radius * 1.25)
    rim_profile = 1.0 - np.abs(dist - radius) / (0.25 * radius)

    # 3. Directional solar illumination and shadowing
    sun_rad = np.radians(sun_azimuth_deg)
    # Gradient direction of crater wall points inward to center
    dx = (cx - x) / (dist + 1e-5)
    dy = (cy - y_eff) / (dist + 1e-5)
    
    # Sun direction vector
    sun_x = np.cos(sun_rad)
    sun_y = np.sin(sun_rad)

    # Dot product of slope normal with sun direction
    slope_sun = dx * sun_x + dy * sun_y

    # Grazing angle factor: lower incidence = deeper shadows & stronger rim crests
    inc_rad = np.radians(sun_incidence_deg)
    shadow_depth = float(np.sin(inc_rad))  # High incidence = longer shadows

    # Shadow on sun-opposing wall interior
    shadow_mask = inner_mask & (slope_sun > 0.05)
    patch[shadow_mask] -= 0.28 * shadow_depth * slope_sun[shadow_mask]

    # Rim crest illumination on sun-facing wall
    lit_mask = rim_mask & (slope_sun < -0.05)
    patch[lit_mask] += 0.25 * (1.0 + 0.3 * shadow_depth) * (-slope_sun[lit_mask]) * rim_profile[lit_mask]

    # Opposing rim shadow
    rim_shadow = rim_mask & (slope_sun > 0.15)
    patch[rim_shadow] -= 0.18 * shadow_depth * slope_sun[rim_shadow] * rim_profile[rim_shadow]

    # Regolith surface texture & sensor noise
    noise = np.random.normal(0, noise_std, (h, w)).astype(np.float32)
    patch = np.clip(patch + noise, 0.05, 0.95)

    return patch.astype(np.float32)

def img_to_b64(img: np.ndarray) -> str:
    """Helper to convert float32 [0, 1] or uint8 image to Base64 PNG."""
    if img.dtype != np.uint8:
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = img
    success, buf = cv2.imencode('.png', u8)
    if not success:
        return ""
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/png;base64,{b64}"

class CraterTrainingManager:
    """Manages the lifecycle, optimization, and training telemetry of CraterInvarianceNet."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = CraterInvarianceNet(embedding_dim=128).to(self.device)
        self.contrastive_loss_fn = AngleInvariantContrastiveLoss(margin=1.0, alpha_pos=1.0, alpha_neg=0.5)
        self.multi_angle_loss_fn = MultiAngleCosineLoss()

        # Cache baseline weights for easy reset
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        self.history: List[Dict[str, Any]] = []
        self.total_steps_executed = 0

    def reset_model(self):
        """Restore neural network parameters to fresh initial baseline state."""
        self.model.load_state_dict(copy.deepcopy(self.initial_state_dict))
        self.history.clear()
        self.total_steps_executed = 0

    def generate_demo_multi_angle_dataset(self) -> Dict[str, Any]:
        """Generate a standardized multi-angle dataset of the same crater viewed at distinct angles."""
        angles_config = [
            {"label": "Angle 1 (High Sun / 25°)", "incidence": 25.0, "azimuth": 45.0, "tilt": 0.0},
            {"label": "Angle 2 (Oblique Sun / 50°)", "incidence": 50.0, "azimuth": 120.0, "tilt": 12.0},
            {"label": "Angle 3 (Grazing Sun / 70°)", "incidence": 70.0, "azimuth": 210.0, "tilt": 22.0},
            {"label": "Angle 4 (Opposite Shadow / 55°)", "incidence": 55.0, "azimuth": 315.0, "tilt": 5.0}
        ]

        images = []
        for cfg in angles_config:
            patch = render_synthetic_crater_at_angle(
                radius=22,
                patch_size=64,
                sun_incidence_deg=cfg["incidence"],
                sun_azimuth_deg=cfg["azimuth"],
                tilt_deg=cfg["tilt"]
            )
            images.append({
                "label": cfg["label"],
                "incidence": cfg["incidence"],
                "azimuth": cfg["azimuth"],
                "tilt": cfg["tilt"],
                "image": patch,
                "preview_b64": img_to_b64(patch)
            })

        # Negative sample (different crater / mare flat)
        neg_patch = render_synthetic_crater_at_angle(
            radius=12,
            patch_size=64,
            sun_incidence_deg=40.0,
            sun_azimuth_deg=90.0,
            noise_std=0.04
        )
        # Shift crater away from center
        neg_patch = np.roll(neg_patch, shift=18, axis=0)

        return {
            "angles": images,
            "negative": {
                "label": "Non-Matching Lunar Terrain (Negative)",
                "image": neg_patch,
                "preview_b64": img_to_b64(neg_patch)
            }
        }

    def sample_from_moon_dataset(self, data_root: str = "D:/moon/moon") -> Dict[str, Any]:
        """Sample authentic multi-rotation lunar craters from the local dataset on disk."""
        import glob
        import re
        import random

        train_dir = os.path.join(data_root, "images", "train")
        if not os.path.isdir(train_dir):
            alt_dir = os.path.join(data_root, "moon", "images", "train")
            if os.path.isdir(alt_dir):
                train_dir = alt_dir
                data_root = os.path.join(data_root, "moon")
            else:
                return self.generate_demo_multi_angle_dataset()

        lbl_dir = os.path.join(data_root, "labels", "train")
        all_imgs = glob.glob(os.path.join(train_dir, "*.png"))
        if not all_imgs:
            all_imgs = glob.glob(os.path.join(train_dir, "*.jpg"))

        groups = {}
        for img_p in all_imgs[:1000]:  # quick sample from first 1000
            fname = os.path.basename(img_p)
            base = re.sub(r'_rotate_\d+\.(png|jpg)', '', fname)
            base = os.path.splitext(base)[0]
            if base not in groups:
                groups[base] = []
            groups[base].append(img_p)

        multi_groups = [g for g in groups.values() if len(g) >= 2]
        if not multi_groups:
            return self.generate_demo_multi_angle_dataset()

        chosen_group = random.choice(multi_groups)
        angles_list = []
        for i, img_path in enumerate(chosen_group[:4]):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape
            # Parse angle from filename if present
            m = re.search(r'_rotate_(\d+)', img_path)
            rot_deg = float(m.group(1)) if m else 0.0

            # Center crop or crater location
            cx, cy = w // 2, h // 2
            lbl_file = os.path.join(lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
            if os.path.exists(lbl_file):
                with open(lbl_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx = int(float(parts[1]) * w)
                            cy = int(float(parts[2]) * h)
                            break

            pad = 32
            x1, y1 = max(0, cx - pad), max(0, cy - pad)
            x2, y2 = min(w, cx + pad), min(h, cy + pad)
            crop = img[y1:y2, x1:x2]
            patch = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

            label_name = f"Angle {i+1} ({int(rot_deg)}° Rotation)" if rot_deg > 0 else "Base Angle (0° Reference)"
            angles_list.append({
                "label": label_name,
                "incidence": 30.0 + rot_deg * 0.1,
                "azimuth": rot_deg,
                "tilt": 0.0,
                "image": patch,
                "preview_b64": img_to_b64(patch)
            })

        # Negative patch from different group
        other_group = random.choice([g for g in multi_groups if g != chosen_group] or multi_groups)
        neg_img = cv2.imread(other_group[0], cv2.IMREAD_GRAYSCALE)
        if neg_img is not None:
            nh, nw = neg_img.shape
            neg_crop = neg_img[nh//4:nh//4+64, nw//4:nw//4+64]
            neg_patch = cv2.resize(neg_crop, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        else:
            neg_patch = np.random.uniform(0.3, 0.7, (64, 64)).astype(np.float32)

        return {
            "angles": angles_list,
            "negative": {
                "label": "Non-Matching Lunar Terrain (Negative from Split Tile)",
                "image": neg_patch,
                "preview_b64": img_to_b64(neg_patch)
            }
        }


    def compute_embedding(self, patch: np.ndarray) -> np.ndarray:
        """Compute 128-D descriptor vector for a single crater patch."""
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
            emb = self.model(t).cpu().numpy()[0]
        return emb

    def compute_feature_heatmap(self, patch: np.ndarray) -> str:
        """Render activation feature map heatmap overlay for the crater patch."""
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(self.device)
            fmap = self.model.extract_feature_map(t)  # (1, 256, 4, 4)
            # Channel energy map
            energy = torch.mean(torch.abs(fmap), dim=1).squeeze(0).cpu().numpy()
            energy_resized = cv2.resize(energy, (64, 64), interpolation=cv2.INTER_CUBIC)
            energy_norm = (energy_resized - energy_resized.min()) / (energy_resized.max() - energy_resized.min() + 1e-7)
            u8_heat = (energy_norm * 255).astype(np.uint8)
            colored = cv2.applyColorMap(u8_heat, cv2.COLORMAP_VIRIDIS)
            # Alpha blend with original
            orig_u8 = cv2.cvtColor(np.clip(patch * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(orig_u8, 0.45, colored, 0.55, 0)
            return img_to_b64(blended)

    def train_step(
        self,
        crater_patches_by_angle: List[np.ndarray],
        negative_patch: Optional[np.ndarray] = None,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        loss_type: str = "contrastive",
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Perform a single forward pass, calculate loss, execute backpropagation, and update parameters."""
        self.model.train()

        # Build optimizer if not passed
        if optimizer is not None:
            opt = optimizer
        elif optimizer_name.lower() == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name.lower() == "adamw":
            opt = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        else:
            opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Snapshot weights before optimization step
        old_params = [p.clone().detach() for p in self.model.parameters() if p.requires_grad]

        # Convert patches to batch tensors
        angle_tensors = [
            torch.from_numpy(p).unsqueeze(0).unsqueeze(0).float().to(self.device)
            for p in crater_patches_by_angle
        ]
        neg_tensor = None
        if negative_patch is not None:
            neg_tensor = torch.from_numpy(negative_patch).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # 1. Forward Pass
        embeddings = [self.model(t) for t in angle_tensors]
        neg_emb = self.model(neg_tensor) if neg_tensor is not None else None

        # Pre-step similarity between angle 0 and other angles
        pre_cos_sims = []
        for i in range(1, len(embeddings)):
            cos = F.cosine_similarity(embeddings[0], embeddings[i]).item()
            pre_cos_sims.append(float(cos))
        avg_pre_sim = float(np.mean(pre_cos_sims)) if pre_cos_sims else 1.0

        # 2. Compute Loss
        if loss_type == "multi_angle_cosine":
            loss, metrics = self.multi_angle_loss_fn(embeddings)
        else:
            # Contrastive across all angle combinations
            pos_losses = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    l_val, m_dict = self.contrastive_loss_fn(embeddings[i], embeddings[j], neg_emb)
                    pos_losses.append(l_val)
            loss = torch.mean(torch.stack(pos_losses))
            metrics = {
                "total_loss": float(loss.item()),
                "avg_positive_cosine": avg_pre_sim,
                "avg_negative_cosine": float(F.cosine_similarity(embeddings[0], neg_emb).item()) if neg_emb is not None else 0.0
            }

        # 3. Backpropagation: compute gradients
        opt.zero_grad()
        loss.backward()

        # Compute gradient norm: ||grad||
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad_norm = float(total_norm ** 0.5)

        # Clip gradients to avoid instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        # 4. Optimizer Step: UPDATE PARAMETERS OF NEURAL NETWORK
        opt.step()
        self.total_steps_executed += 1

        # 5. Measure Parameter Update Magnitude: ||theta_new - theta_old||
        new_params = [p.clone().detach() for p in self.model.parameters() if p.requires_grad]
        delta_norm_sq = 0.0
        layer_updates = []
        for i, (p_old, p_new) in enumerate(zip(old_params, new_params)):
            diff = (p_new - p_old).norm(2).item()
            delta_norm_sq += diff ** 2
            if i < 4:  # Collect top layer updates for logging
                layer_updates.append(round(diff, 6))

        total_param_delta = float(delta_norm_sq ** 0.5)

        # 6. Post-Step Evaluation
        self.model.eval()
        with torch.no_grad():
            post_embeddings = [self.model(t) for t in angle_tensors]
            post_cos_sims = []
            for i in range(1, len(post_embeddings)):
                cos = F.cosine_similarity(post_embeddings[0], post_embeddings[i]).item()
                post_cos_sims.append(float(cos))
            avg_post_sim = float(np.mean(post_cos_sims)) if post_cos_sims else 1.0

        step_report = {
            "step": self.total_steps_executed,
            "loss": round(float(loss.item()), 5),
            "grad_norm": round(total_grad_norm, 5),
            "param_delta": round(total_param_delta, 6),
            "layer_deltas_sample": layer_updates,
            "pre_similarity_pct": round(avg_pre_sim * 100.0, 2),
            "post_similarity_pct": round(avg_post_sim * 100.0, 2),
            "similarity_gain_pct": round((avg_post_sim - avg_pre_sim) * 100.0, 2)
        }
        return step_report

    def train_epochs(
        self,
        crater_patches_by_angle: List[np.ndarray],
        negative_patch: Optional[np.ndarray] = None,
        epochs: int = 10,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        loss_type: str = "contrastive"
    ) -> Dict[str, Any]:
        """Execute full training epoch loop with parameter optimization and metric collection."""
        t_start = time.time()
        epoch_history = []

        if optimizer_name.lower() == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_name.lower() == "adamw":
            opt = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        else:
            opt = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        initial_sim = None
        for ep in range(1, epochs + 1):
            report = self.train_step(
                crater_patches_by_angle,
                negative_patch=negative_patch,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                loss_type=loss_type,
                optimizer=opt
            )
            report["epoch"] = ep
            if initial_sim is None:
                initial_sim = report["pre_similarity_pct"]
            epoch_history.append(report)
            self.history.append(report)

        final_sim = epoch_history[-1]["post_similarity_pct"] if epoch_history else 0.0
        final_loss = epoch_history[-1]["loss"] if epoch_history else 0.0

        # Generate post-training heatmaps for visual comparison
        visualizations = []
        for idx, patch in enumerate(crater_patches_by_angle):
            hmap = self.compute_feature_heatmap(patch)
            visualizations.append({
                "angle_index": idx + 1,
                "preview_b64": img_to_b64(patch),
                "heatmap_b64": hmap
            })

        return {
            "epochs_completed": epochs,
            "total_steps": self.total_steps_executed,
            "initial_loss": epoch_history[0]["loss"] if epoch_history else 0.0,
            "final_loss": final_loss,
            "loss_reduction_pct": round(
                max(0.0, (epoch_history[0]["loss"] - final_loss) / (epoch_history[0]["loss"] + 1e-7) * 100.0), 2
            ) if epoch_history else 0.0,
            "initial_similarity_pct": initial_sim,
            "final_similarity_pct": final_sim,
            "similarity_gain_pct": round(final_sim - (initial_sim or 0.0), 2),
            "history": epoch_history,
            "visualizations": visualizations,
            "execution_time_sec": round(time.time() - t_start, 3)
        }
