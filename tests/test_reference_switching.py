import urllib.request
import json
import numpy as np
import cv2
import io

print("=== 1. Health Check ===")
with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print("Health status:", data["status"])

print("\n=== 2. Home Dashboard HTML Check ===")
with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
    html = resp.read().decode("utf-8")
    assert "3 Images Processed" not in html, "Demo text should not be hardcoded in HTML"
    assert "— Images Processed" in html, "Placeholder found"
    assert "anchorRow" in html, "Reference anchor row found in HTML"
    assert "swapImagesBtn" in html, "Swap button found in HTML"
    print("Clean initial state verified: no demo data leak on initial load.")

print("\n=== 3. Explicit Demo Request Check ===")
req = urllib.request.Request("http://127.0.0.1:8000/api/pipeline/demo", data=b"", method="POST")
with urllib.request.urlopen(req) as resp:
    demo_res = json.loads(resp.read().decode("utf-8"))
    print("Demo processed count:", demo_res["processed_count"])
    gm = demo_res["graph_match"]
    print(f"Demo match: {gm['image_a_name']} <-> {gm['image_b_name']}, Nodes: {gm['total_matched_features']}, Craters: {gm['matched_craters_count']}")

print("\n=== 4. Testing Pair Matching when Switching Reference & Comparison Images ===")
pairs_to_test = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
for idx_a, idx_b in pairs_to_test:
    payload = json.dumps({"index_a": idx_a, "index_b": idx_b}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/pipeline/match_pair",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        m_res = json.loads(resp.read().decode("utf-8"))
        name_a = m_res["image_a_name"]
        name_b = m_res["image_b_name"]
        nodes_cnt = m_res["total_matched_features"]
        edges_cnt = len(m_res["edges"])
        craters_cnt = m_res["matched_craters_count"]
        print(f"  [Ref {idx_a} vs Comp {idx_b}] {name_a} ⇄ {name_b} -> {nodes_cnt} Nodes (1..{nodes_cnt}), {craters_cnt} Craters, {edges_cnt} Delaunay Edges (NO OVERLAY)")

print("\n=== 5. Testing Process Endpoint with Custom Images and Reference Selection ===")
# Generate two dummy lunar images as PNG bytes
def make_png(pattern_seed):
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (80 + pattern_seed * 20, 80), 30, 180, -1)
    cv2.circle(img, (160, 160 + pattern_seed * 10), 25, 220, -1)
    # Add noise
    noise = np.random.randint(0, 40, (256, 256), dtype=np.uint8)
    img = cv2.add(img, noise)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()

png1 = make_png(1)
png2 = make_png(2)

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()

# Part 1: file 1
body.write(f"--{boundary}\r\n".encode("utf-8"))
body.write(b'Content-Disposition: form-data; name="files"; filename="custom_lunar_A.png"\r\n')
body.write(b"Content-Type: image/png\r\n\r\n")
body.write(png1)
body.write(b"\r\n")

# Part 2: file 2
body.write(f"--{boundary}\r\n".encode("utf-8"))
body.write(b'Content-Disposition: form-data; name="files"; filename="custom_lunar_B.png"\r\n')
body.write(b"Content-Type: image/png\r\n\r\n")
body.write(png2)
body.write(b"\r\n")

# Part 3: anchor_index = 1 (custom_lunar_B as Reference!)
body.write(f"--{boundary}\r\n".encode("utf-8"))
body.write(b'Content-Disposition: form-data; name="anchor_index"\r\n\r\n')
body.write(b"1\r\n")

body.write(f"--{boundary}--\r\n".encode("utf-8"))
body_bytes = body.getvalue()

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/pipeline/process",
    data=body_bytes,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    proc_res = json.loads(resp.read().decode("utf-8"))
    print("Process with custom images SUCCESS!")
    print("Designated Reference Anchor:", proc_res["anchor"]["name"])
    print("Processed count:", proc_res["processed_count"])
    print("Initial match:", proc_res["graph_match"]["image_a_name"], "⇄", proc_res["graph_match"]["image_b_name"])

print("\n=== ALL INTEGRATION VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
