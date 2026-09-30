"""Lunar Crater Detection System — FastAPI Application.

Implements the Web API for ingesting multi-image lunar datasets,
orchestrating the 10-stage pipeline, and delivering rich comparative metrics.
"""
import os
import io
import time
import base64
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import PipelineConfig
from app.pipeline.pds4_reader import PDS4Reader, LunarImageProduct, PDS4Metadata
from app.pipeline.manager import LunarPipelineManager, FullPipelineJobResult, PairPipelineResult

app = FastAPI(
    title="Lunar Crater Detection System API",
    description="Photometric correction, SuperPoint vs LoFTR registration, and crater detection pipeline.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
config = PipelineConfig()
pipeline_manager = LunarPipelineManager(config)

# Helper function to encode image to base64
def img_to_base64_png(img: np.ndarray) -> str:
    """Convert numpy array (float32 [0,1] or uint8) to Base64 PNG data URL."""
    if img.dtype != np.uint8:
        if img.ndim == 2:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = img

    success, buf = cv2.imencode('.png', u8)
    if not success:
        return ""
    b64 = base64.b64encode(buf).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def draw_correspondence_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 80
) -> str:
    """Draw side-by-side correspondence lines color-coded by inlier (green) vs outlier (red)."""
    if img_a is None or img_a.size == 0 or img_a.ndim < 2:
        return ""
    if img_b is None or img_b.size == 0 or img_b.ndim < 2:
        img_b = np.zeros_like(img_a)

    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    h = max(ha, hb)
    w = wa + wb

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    u8_a = np.clip(img_a * 255, 0, 255).astype(np.uint8)
    u8_b = np.clip(img_b * 255, 0, 255).astype(np.uint8)

    canvas[:ha, :wa] = cv2.cvtColor(u8_a, cv2.COLOR_GRAY2BGR) if u8_a.ndim == 2 else u8_a
    canvas[:hb, wa:] = cv2.cvtColor(u8_b, cv2.COLOR_GRAY2BGR) if u8_b.ndim == 2 else u8_b

    # Draw separator line
    cv2.line(canvas, (wa, 0), (wa, h), (80, 80, 80), 2)

    num_pts = len(pts_a)
    if num_pts == 0:
        return img_to_base64_png(canvas)

    indices = np.arange(num_pts)
    if num_pts > max_draw:
        # Prioritize showing inliers
        inlier_indices = indices[inlier_mask]
        outlier_indices = indices[~inlier_mask]
        num_in = min(len(inlier_indices), int(max_draw * 0.75))
        num_out = min(len(outlier_indices), max_draw - num_in)
        selected_idx = np.concatenate([
            np.random.choice(inlier_indices, num_in, replace=False) if len(inlier_indices) > 0 else [],
            np.random.choice(outlier_indices, num_out, replace=False) if len(outlier_indices) > 0 else []
        ]).astype(int)
    else:
        selected_idx = indices

    for idx in selected_idx:
        pt_a = (int(round(pts_a[idx][0])), int(round(pts_a[idx][1])))
        pt_b = (int(round(pts_b[idx][0])) + wa, int(round(pts_b[idx][1])))
        is_inlier = bool(inlier_mask[idx])

        if is_inlier:
            color = (0, 230, 115)  # Emerald green for inlier
            thick = 1
        else:
            color = (0, 60, 240)    # Red for outlier
            thick = 1

        cv2.circle(canvas, pt_a, 3, color, -1)
        cv2.circle(canvas, pt_b, 3, color, -1)
        cv2.line(canvas, pt_a, pt_b, color, thick, lineType=cv2.LINE_AA)

    return img_to_base64_png(canvas)

class MatchPairRequest(BaseModel):
    index_a: int = 0
    index_b: int = 1

def serialize_single_image(res) -> Dict[str, Any]:
    """Serialize an individual processed lunar image result."""
    return {
        "id": res.id,
        "name": res.name,
        "dimensions": {"height": int(res.image.shape[0]), "width": int(res.image.shape[1])},
        "preview_b64": img_to_base64_png(res.image),
        "annotated_features_b64": img_to_base64_png(res.annotated_features),
        "total_craters": len(res.craters),
        "total_features": len(res.keypoints),
        "execution_time_sec": res.execution_time_sec,
        "craters": [
            {
                "id": c.crater_id,
                "x": c.center_x,
                "y": c.center_y,
                "radius": c.radius,
                "diameter": c.diameter,
                "rim_contrast": c.rim_contrast,
                "confidence": c.confidence
            }
            for c in res.craters
        ]
    }

def serialize_graph_match(res) -> Dict[str, Any]:
    """Serialize two-image feature graph match results (labeled 1..n, no overlay)."""
    return {
        "image_a_id": res.image_a_id,
        "image_a_name": res.image_a_name,
        "image_b_id": res.image_b_id,
        "image_b_name": res.image_b_name,
        "total_matched_features": res.total_matched_features,
        "matched_craters_count": res.matched_craters_count,
        "inlier_count": res.inlier_count,
        "is_matched": getattr(res, "is_matched", res.total_matched_features > 0),
        "status_message": getattr(res, "status_message", ""),
        "photo_a_graph_b64": img_to_base64_png(res.photo_a_graph),
        "photo_b_graph_b64": img_to_base64_png(res.photo_b_graph),
        "side_by_side_graph_b64": img_to_base64_png(res.side_by_side_graph),
        "edges": res.edges,
        "execution_time_sec": res.execution_time_sec,
        "nodes": [
            {
                "label": n.label,
                "pt_a": [round(float(n.pt_a[0]), 1), round(float(n.pt_a[1]), 1)],
                "pt_b": [round(float(n.pt_b[0]), 1), round(float(n.pt_b[1]), 1)],
                "is_crater": n.is_crater,
                "crater_a_id": n.crater_a_id,
                "crater_b_id": n.crater_b_id,
                "diameter_a": n.diameter_a,
                "diameter_b": n.diameter_b,
                "rim_contrast_a": n.rim_contrast_a,
                "rim_contrast_b": n.rim_contrast_b,
                "similarity_probability": n.similarity_probability,
                "similarity_percentage": round(n.similarity_probability * 100.0, 1),
                "similarity_description": n.similarity_description
            }
            for n in res.nodes
        ]
    }

def serialize_pipeline_result(job_res: FullPipelineJobResult) -> Dict[str, Any]:
    """Serialize pipeline results into clean, self-contained JSON response."""
    anchor = job_res.anchor_item
    anchor_preview_b64 = img_to_base64_png(anchor.image)

    # Gather single image detection results strictly from current job items
    single_images_data = []
    processed_results = []
    for itm in job_res.processed_items:
        s_res = pipeline_manager.cached_single_images.get(itm.id)
        if s_res is not None:
            single_images_data.append(serialize_single_image(s_res))
            processed_results.append(s_res)

    # Compute graph match between designated reference anchor and first comparison image
    graph_match_data = None
    if len(processed_results) >= 2:
        anchor_s_res = pipeline_manager.cached_single_images.get(anchor.id)
        target_s_res = None
        for s in processed_results:
            if s.id != anchor.id:
                target_s_res = s
                break
        if anchor_s_res is None:
            anchor_s_res = processed_results[0]
            target_s_res = processed_results[1]
        elif target_s_res is None:
            target_s_res = processed_results[0] if processed_results[0].id != anchor.id else processed_results[1]

        try:
            gm = pipeline_manager.match_two_specific_images(anchor_s_res, target_s_res)
            graph_match_data = serialize_graph_match(gm)
        except Exception as e:
            print("Graph match calculation error:", e)

    pairs_data = []
    for pr in job_res.pair_results:
        # Resolve target image for pair
        target_item = next((itm for itm in job_res.processed_items if itm.id == pr.target_id), None)
        target_img = target_item.image if target_item is not None else anchor.image

        # Generate match comparison graphics
        sp_viz = draw_correspondence_matches(
            anchor.image,
            target_img,
            pr.superpoint_geom.anchor_inliers,
            pr.superpoint_geom.target_inliers,
            np.ones((len(pr.superpoint_geom.anchor_inliers),), dtype=bool)
        )
        loftr_viz = draw_correspondence_matches(
            anchor.image,
            target_img,
            pr.loftr_geom.anchor_inliers,
            pr.loftr_geom.target_inliers,
            np.ones((len(pr.loftr_geom.anchor_inliers),), dtype=bool)
        )

        warped_b64 = img_to_base64_png(pr.registration.warped_target)
        overlap_mask_b64 = img_to_base64_png(pr.overlap.overlap_mask)
        blended_b64 = img_to_base64_png(pr.overlap.blended_preview)
        craters_b64 = img_to_base64_png(pr.crater_detection.annotated_image)

        pairs_data.append({
            "pair_id": pr.pair_id,
            "target_id": pr.target_id,
            "target_name": pr.target_name,
            "execution_time_sec": pr.execution_time_sec,
            "algorithms": {
                "superpoint": {
                    "total_matches": pr.superpoint_matches,
                    "inlier_count": pr.superpoint_geom.inlier_count,
                    "inlier_ratio": pr.superpoint_eval.inlier_ratio,
                    "reprojection_rmse": pr.superpoint_eval.reprojection_rmse,
                    "spatial_score": pr.superpoint_eval.spatial.spatial_uniformity_score,
                    "hull_ratio": pr.superpoint_eval.spatial.convex_hull_ratio,
                    "grid_entropy": pr.superpoint_eval.spatial.grid_entropy,
                    "active_quadrants": pr.superpoint_eval.spatial.active_quadrants,
                    "is_plausible": pr.superpoint_eval.plausibility.is_plausible,
                    "condition_number": pr.superpoint_eval.plausibility.condition_number,
                    "determinant": pr.superpoint_eval.plausibility.determinant_2x2,
                    "composite_score": pr.superpoint_eval.composite_reliability_score,
                    "is_reliable": pr.superpoint_eval.is_reliable,
                    "summary": pr.superpoint_eval.summary,
                    "match_visualization_b64": sp_viz
                },
                "loftr": {
                    "total_matches": pr.loftr_matches,
                    "inlier_count": pr.loftr_geom.inlier_count,
                    "inlier_ratio": pr.loftr_eval.inlier_ratio,
                    "reprojection_rmse": pr.loftr_eval.reprojection_rmse,
                    "spatial_score": pr.loftr_eval.spatial.spatial_uniformity_score,
                    "hull_ratio": pr.loftr_eval.spatial.convex_hull_ratio,
                    "grid_entropy": pr.loftr_eval.spatial.grid_entropy,
                    "active_quadrants": pr.loftr_eval.spatial.active_quadrants,
                    "is_plausible": pr.loftr_eval.plausibility.is_plausible,
                    "condition_number": pr.loftr_eval.plausibility.condition_number,
                    "determinant": pr.loftr_eval.plausibility.determinant_2x2,
                    "composite_score": pr.loftr_eval.composite_reliability_score,
                    "is_reliable": pr.loftr_eval.is_reliable,
                    "summary": pr.loftr_eval.summary,
                    "match_visualization_b64": loftr_viz
                }
            },
            "selection": {
                "selected_algorithm": pr.decision.selected_algorithm,
                "confidence_level": pr.decision.confidence_level,
                "is_fallback_used": pr.decision.is_fallback_used,
                "rationale": pr.decision.selection_rationale,
                "homography": pr.decision.selected_homography.tolist() if pr.decision.selected_homography is not None else None
            },
            "registration": {
                "warped_target_b64": warped_b64,
                "anchor_shape": pr.registration.anchor_shape
            },
            "overlap": {
                "overlap_area_pixels": pr.overlap.overlap_area_pixels,
                "overlap_ratio_anchor": pr.overlap.overlap_ratio_anchor,
                "overlap_ratio_target": pr.overlap.overlap_ratio_target,
                "bounding_box": list(pr.overlap.bounding_box),
                "contour_vertices": pr.overlap.contour_points,
                "has_valid_overlap": pr.overlap.has_valid_overlap,
                "overlap_mask_b64": overlap_mask_b64,
                "blended_preview_b64": blended_b64
            },
            "crater_detection": {
                "total_detected": pr.crater_detection.total_craters_detected,
                "craters_in_overlap": pr.crater_detection.craters_in_overlap,
                "cross_verified_craters": pr.crater_detection.cross_verified_craters,
                "density_per_sq_kpix": pr.crater_detection.density_per_sq_kpix,
                "annotated_image_b64": craters_b64,
                "craters": [
                    {
                        "id": c.crater_id,
                        "x": c.center_x,
                        "y": c.center_y,
                        "radius": c.radius,
                        "diameter": c.diameter,
                        "confidence": c.confidence,
                        "in_overlap": c.in_overlap,
                        "cross_verified": c.cross_verified,
                        "rim_contrast": c.rim_contrast
                    }
                    for c in pr.crater_detection.craters[:50]
                ]
            }
        })

    return {
        "job_id": job_res.job_id,
        "status": job_res.status,
        "total_execution_time_sec": job_res.total_execution_time_sec,
        "anchor": {
            "id": anchor.id,
            "name": anchor.name,
            "metadata": anchor.metadata,
            "preview_b64": anchor_preview_b64
        },
        "processed_count": len(job_res.processed_items),
        "single_images": single_images_data,
        "graph_match": graph_match_data,
        "pairs": pairs_data
    }

@app.get("/api/health")
def health_check():
    """System health check and pipeline capability report."""
    return {
        "status": "healthy",
        "pipeline_stages": [
            "1. Web API & PDS4 / Raster Ingestion",
            "2. Radiometric Calibration & Parallel NLM & Percentile Norm",
            "3. Single-Image Feature & Crater Detection",
            "4. Parallel SuperPoint & LoFTR Correspondence Engines",
            "5. Independent RANSAC Geometric Verification",
            "6. Multi-Metric Reliability Evaluation",
            "7. Most Reliable Result Selection",
            "8. Two-Image Graph Matching (Labeled 1..n, No Overlay)",
            "9. Crater Similarity Description Probability Estimation",
            "10. Lunar Crater Feature Catalog"
        ],
        "device": str(pipeline_manager.superpoint.device),
        "neural_trainer": {
            "model": "CraterInvarianceNet",
            "embedding_dim": pipeline_manager.trainer.model.embedding_dim,
            "device": str(pipeline_manager.trainer.device)
        }
    }

# =========================================================================
# NEURAL NETWORK TRAINING ENDPOINTS (Multi-Angle Crater Loss & Optimization)
# =========================================================================

@app.get("/api/training/status")
def get_training_status():
    """Get neural network training status, parameters count, and execution history."""
    model = pipeline_manager.trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": "CraterInvarianceNet (Multi-Angle Deep Representation)",
        "embedding_dim": model.embedding_dim,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(pipeline_manager.trainer.device),
        "total_steps_executed": pipeline_manager.trainer.total_steps_executed,
        "history_count": len(pipeline_manager.trainer.history),
        "last_step": pipeline_manager.trainer.history[-1] if pipeline_manager.trainer.history else None
    }

@app.get("/api/training/sample_data")
def get_sample_training_dataset():
    """Provide a synthesized multi-angle crater dataset (varying solar incidence, azimuth, and tilts)."""
    dataset = pipeline_manager.trainer.generate_demo_multi_angle_dataset()
    # Serialize for JSON response
    return JSONResponse(content={
        "status": "success",
        "angles": [
            {
                "label": item["label"],
                "incidence_deg": item["incidence"],
                "azimuth_deg": item["azimuth"],
                "tilt_deg": item["tilt"],
                "preview_b64": item["preview_b64"]
            }
            for item in dataset["angles"]
        ],
        "negative": {
            "label": dataset["negative"]["label"],
            "preview_b64": dataset["negative"]["preview_b64"]
        }
    })

@app.get("/api/training/moon_data")
def get_moon_training_dataset():
    """Sample authentic multi-rotation crater pairs from local dataset at D:/moon/moon."""
    dataset = pipeline_manager.trainer.sample_from_moon_dataset("D:/moon/moon")
    return JSONResponse(content={
        "status": "success",
        "dataset_source": "D:/moon/moon",
        "angles": [
            {
                "label": item["label"],
                "incidence_deg": item["incidence"],
                "azimuth_deg": item["azimuth"],
                "tilt_deg": item["tilt"],
                "preview_b64": item["preview_b64"]
            }
            for item in dataset["angles"]
        ],
        "negative": {
            "label": dataset["negative"]["label"],
            "preview_b64": dataset["negative"]["preview_b64"]
        }
    })

@app.post("/api/training/train")
async def train_neural_network(
    epochs: int = Form(5),
    learning_rate: float = Form(0.001),
    loss_type: str = Form("contrastive"),
    optimizer: str = Form("adam"),
    dataset_source: str = Form("auto"),
    files: Optional[List[UploadFile]] = File(None)
):
    """Calculate multi-angle crater loss, run backpropagation, and update neural network parameters."""
    patches = []
    neg_patch = None

    if files and len(files) >= 2:
        for f in files:
            content = await f.read()
            arr = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Preprocess patch to float32 [0, 1] and 64x64
                img_f = img.astype(np.float32) / 255.0
                patch = cv2.resize(img_f, (64, 64), interpolation=cv2.INTER_AREA)
                patches.append(patch)

    if len(patches) < 2:
        # Check if local D:/moon/moon dataset is requested or present
        if dataset_source in ("moon", "auto") and os.path.exists("D:/moon/moon"):
            moon_data = pipeline_manager.trainer.sample_from_moon_dataset("D:/moon/moon")
            patches = [item["image"] for item in moon_data["angles"]]
            neg_patch = moon_data["negative"]["image"]
        else:
            # Fall back to synthetic / physics-grounded multi-angle crater dataset
            demo_data = pipeline_manager.trainer.generate_demo_multi_angle_dataset()
            patches = [item["image"] for item in demo_data["angles"]]
            neg_patch = demo_data["negative"]["image"]


    try:
        train_result = pipeline_manager.trainer.train_epochs(
            crater_patches_by_angle=patches,
            negative_patch=neg_patch,
            epochs=max(1, min(50, epochs)),
            learning_rate=max(1e-5, min(0.1, learning_rate)),
            optimizer_name=optimizer,
            loss_type=loss_type
        )
        return JSONResponse(content=train_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/api/training/reset")
def reset_neural_weights():
    """Reset neural network parameters back to initial baseline weights."""
    pipeline_manager.trainer.reset_model()
    return {
        "status": "success",
        "message": "CraterInvarianceNet parameters successfully reset to initial weights.",
        "total_steps": 0
    }

@app.post("/api/pipeline/match_pair")
def match_specific_pair(req: MatchPairRequest):
    """Compute feature & graph match between two specifically selected images (labeled 1..n, no overlay)."""
    cached_list = list(pipeline_manager.cached_single_images.values())
    if not cached_list or len(cached_list) < 2:
        raise HTTPException(status_code=400, detail="No processed images available. Run pipeline or demo first.")

    idx_a = max(0, min(len(cached_list) - 1, req.index_a))
    idx_b = max(0, min(len(cached_list) - 1, req.index_b))

    img_a = cached_list[idx_a]
    img_b = cached_list[idx_b]

    try:
        match_res = pipeline_manager.match_two_specific_images(img_a, img_b)
        return JSONResponse(content=serialize_graph_match(match_res))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pair matching error between Image {idx_a+1} and Image {idx_b+1}: {str(e)}")

@app.post("/api/pipeline/demo")
def run_demo_pipeline():
    """Run pipeline on pre-generated lunar PDS4 test products with single-image caching and dual graph matching."""
    demo_files = [
        "demo_data/LROC_NAC_M10001_ANCHOR.xml",
        "demo_data/LROC_NAC_M10002_TARGET1.xml",
        "demo_data/LROC_NAC_M10003_TARGET2.xml"
    ]
    products = []
    for xml_f in demo_files:
        if not os.path.exists(xml_f):
            raise HTTPException(status_code=404, detail=f"Demo file {xml_f} not found. Generate demo data first.")
        prod = PDS4Reader.load_pds4_pair(xml_f)
        products.append(prod)

    res = pipeline_manager.process_products(products, anchor_identifier=0)
    return JSONResponse(content=serialize_pipeline_result(res))

@app.post("/api/pipeline/process")
async def process_images(
    files: List[UploadFile] = File(...),
    anchor_index: int = Form(0),
    anchor_filename: Optional[str] = Form(None)
):
    """Receive multiple lunar images (PDS4 XML/raw pairs or PNG/JPEG/TIFF rasters) and execute the 10-stage pipeline."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 lunar images (Anchor + at least 1 Target) are required.")

    # Read uploaded files
    file_map = {}
    for f in files:
        contents = await f.read()
        file_map[f.filename] = contents

    # Separate XML labels and data files
    xml_files = {name: data for name, data in file_map.items() if name.lower().endswith('.xml')}
    products: List[LunarImageProduct] = []

    if xml_files:
        # PDS4 XML pairs
        for xml_name, xml_bytes in xml_files.items():
            meta = PDS4Reader.parse_xml_label(xml_bytes)
            base_name = os.path.splitext(xml_name)[0]
            # Find matching data payload in upload
            data_bytes = None
            data_ext = None
            for ext in [".raw", ".dat", ".img", ".bin", ".tif", ".png", ".jpg"]:
                cand = base_name + ext
                if cand in file_map:
                    data_bytes = file_map[cand]
                    data_ext = ext
                    break

            if data_bytes is None:
                # If only XML without binary, skip or error
                continue

            if data_ext in [".png", ".jpg", ".tif"]:
                prod = PDS4Reader.load_from_bytes(data_bytes, xml_name)
                prod.metadata = meta
            else:
                np_dtype = PDS4Reader.PDS4_DTYPE_MAP.get(meta.data_type, "<f4")
                arr = np.frombuffer(data_bytes[meta.byte_offset:], dtype=np_dtype)
                if meta.lines > 0 and meta.samples > 0:
                    raw_img = arr[:meta.lines * meta.samples].reshape((meta.lines, meta.samples)).astype(np.float32)
                else:
                    dim = int(np.sqrt(len(arr)))
                    raw_img = arr[:dim*dim].reshape((dim, dim)).astype(np.float32)
                
                img_scaled = raw_img * meta.scaling_factor + meta.scaling_offset
                vmin, vmax = img_scaled.min(), img_scaled.max()
                norm_img = (img_scaled - vmin) / (vmax - vmin + 1e-7)

                prod = LunarImageProduct(
                    name=xml_name,
                    image=norm_img,
                    raw_image=raw_img,
                    metadata=meta
                )
            products.append(prod)

    # Standard raster images
    for name, data in file_map.items():
        ext = os.path.splitext(name)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            # Only add if not already paired with PDS4 XML
            base = os.path.splitext(name)[0]
            if f"{base}.xml" not in xml_files:
                try:
                    prod = PDS4Reader.load_from_bytes(data, name)
                    products.append(prod)
                except Exception as e:
                    pass

    if len(products) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Parsed {len(products)} valid lunar products. Need at least 2 for anchor registration pairing."
        )

    # Determine anchor product index
    anchor_id_resolved = 0
    if anchor_filename:
        target_stem = os.path.splitext(anchor_filename)[0].lower()
        for idx, prod in enumerate(products):
            if target_stem in prod.name.lower():
                anchor_id_resolved = idx
                break
    elif 0 <= anchor_index < len(products):
        anchor_id_resolved = anchor_index

    # Run the 10-stage pipeline
    try:
        job_result = pipeline_manager.process_products(products, anchor_identifier=anchor_id_resolved)
        return JSONResponse(content=serialize_pipeline_result(job_result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

# Mount static folder for frontend dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the interactive web UI dashboard."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Lunar Crater Detection System API is running. UI is initializing...</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
