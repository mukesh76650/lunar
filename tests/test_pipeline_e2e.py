"""End-to-End Pipeline Automated Test."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
from app.main import app, run_demo_pipeline, health_check

def test_health_check_endpoint():
    res = health_check()
    assert res["status"] == "healthy"
    assert len(res["pipeline_stages"]) == 10
    print("Health check endpoint passed!")

def test_demo_pipeline_e2e():
    print("Testing End-to-End Pipeline on Demo PDS4 dataset...")
    response = run_demo_pipeline()
    assert response.status_code == 200
    
    # Parse JSON body
    body = json.loads(response.body.decode('utf-8'))
    assert body["status"] == "SUCCESS"
    assert "anchor" in body
    assert len(body["pairs"]) == 2

    # Check pair 1 details
    pair1 = body["pairs"][0]
    sp = pair1["algorithms"]["superpoint"]
    loftr = pair1["algorithms"]["loftr"]
    sel = pair1["selection"]
    reg = pair1["registration"]
    ovl = pair1["overlap"]
    cd = pair1["crater_detection"]

    print(f"Pair 1 ({pair1['pair_id']}):")
    print(f"  SuperPoint: {sp['inlier_count']} inliers, RMSE={sp['reprojection_rmse']}px, Score={sp['composite_score']}")
    print(f"  LoFTR:      {loftr['inlier_count']} inliers, RMSE={loftr['reprojection_rmse']}px, Score={loftr['composite_score']}")
    print(f"  Selected:   {sel['selected_algorithm']} (Confidence: {sel['confidence_level']})")
    print(f"  Overlap:    {ovl['overlap_ratio_anchor']:.1%} of anchor ({ovl['overlap_area_pixels']} px)")
    print(f"  Craters:    {cd['total_detected']} detected, {cd['cross_verified_craters']} verified in overlap")

    # Assertions ensuring all 10 architectural stages executed properly
    assert sp["total_matches"] > 0
    assert loftr["total_matches"] > 0
    assert sel["selected_algorithm"] in ["SuperPoint", "LoFTR"]
    assert ovl["overlap_area_pixels"] > 0
    assert cd["total_detected"] > 0
    assert len(reg["warped_target_b64"]) > 100
    assert len(cd["annotated_image_b64"]) > 100

    # Assertions verifying single-image processing and two-image graph matching
    assert "single_images" in body
    assert len(body["single_images"]) == 3
    for s_img in body["single_images"]:
        assert len(s_img["preview_b64"]) > 50
        assert len(s_img["annotated_features_b64"]) > 50
        assert s_img["total_craters"] > 0
        assert s_img["total_features"] > 0

    assert "graph_match" in body
    gm = body["graph_match"]
    assert gm["total_matched_features"] > 0
    assert len(gm["nodes"]) > 0
    assert len(gm["edges"]) > 0
    assert len(gm["side_by_side_graph_b64"]) > 50
    # Check node 1 has 1-based label, probability, and description
    node1 = gm["nodes"][0]
    assert node1["label"] == 1
    assert 0.0 <= node1["similarity_probability"] <= 1.0
    assert len(node1["similarity_description"]) > 10

    print("End-to-end pipeline test PASSED successfully!")

def test_match_specific_pair():
    print("Testing two-image graph matching endpoint for user-selected pair...")
    from app.main import match_specific_pair, MatchPairRequest
    res = match_specific_pair(MatchPairRequest(index_a=0, index_b=2))
    assert res.status_code == 200
    data = json.loads(res.body.decode('utf-8'))
    assert data["total_matched_features"] > 0
    assert len(data["nodes"]) == data["total_matched_features"]
    assert len(data["edges"]) > 0
    assert len(data["side_by_side_graph_b64"]) > 100
    # Check node labels are strictly 1 to n
    labels = [n["label"] for n in data["nodes"]]
    assert labels == list(range(1, len(data["nodes"]) + 1))
    print(f"Two-image graph match PASSED: {len(data['nodes'])} nodes labeled 1..{len(data['nodes'])}, {len(data['edges'])} edges.")

if __name__ == "__main__":
    test_health_check_endpoint()
    test_demo_pipeline_e2e()
    test_match_specific_pair()

