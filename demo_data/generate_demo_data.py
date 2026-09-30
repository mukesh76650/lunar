"""Generate Realistic Lunar PDS4 and Raster Test Dataset."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2

def generate_lunar_scene(seed=42, h=384, w=384, offset_x=0, offset_y=0, rot_deg=0.0):
    np.random.seed(seed)
    y, x = np.mgrid[0:h, 0:w]
    # Lunar mare base terrain
    base = 0.40 + 0.05 * np.sin(x / 30.0) + 0.04 * np.cos(y / 25.0)

    # Lunar craters (Tycho, Copernicus style craters)
    craters = [
        (90, 110, 38, 0.45),
        (260, 95, 52, 0.50),
        (160, 230, 44, 0.48),
        (300, 280, 30, 0.35),
        (75, 290, 28, 0.38),
        (190, 130, 20, 0.30),
        (220, 330, 16, 0.25),
        (330, 170, 22, 0.32)
    ]

    for cx, cy, r, depth in craters:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # Depression
        mask_inner = dist < r
        base[mask_inner] -= depth * (1.0 - (dist[mask_inner] / r)**1.5)
        # Rim
        mask_rim = (dist >= r * 0.75) & (dist <= r * 1.25)
        angle = np.arctan2(y - cy, x - cx)
        sun_align = np.cos(angle - np.radians(210))  # Sun illumination direction
        base[mask_rim] += 0.25 * sun_align[mask_rim] * (1.0 - np.abs(dist[mask_rim] - r) / (0.25 * r))

    # Add micrometeorite regolith texture
    regolith = np.random.normal(0, 0.018, (h, w))
    img = np.clip(base + regolith, 0.0, 1.0).astype(np.float32)

    # Apply rigid transform
    if rot_deg != 0.0 or offset_x != 0 or offset_y != 0:
        M = cv2.getRotationMatrix2D((w/2, h/2), rot_deg, 1.0)
        M[0, 2] += offset_x
        M[1, 2] += offset_y
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return img

def create_pds4_dataset(output_dir="demo_data"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Anchor Lunar Frame (Reference)
    anchor_img = generate_lunar_scene(seed=101, h=384, w=384, offset_x=0, offset_y=0, rot_deg=0.0)
    
    # 2. Target Frame 1 (Moderate offset + slight rotation)
    target1_img = generate_lunar_scene(seed=101, h=384, w=384, offset_x=18, offset_y=-12, rot_deg=2.5)

    # 3. Target Frame 2 (Larger shift)
    target2_img = generate_lunar_scene(seed=101, h=384, w=384, offset_x=-25, offset_y=20, rot_deg=-3.0)

    images = [
        ("LROC_NAC_M10001_ANCHOR", anchor_img, 35.0, 5.0, 30.0),
        ("LROC_NAC_M10002_TARGET1", target1_img, 38.0, 8.0, 32.0),
        ("LROC_NAC_M10003_TARGET2", target2_img, 42.0, 12.0, 35.0)
    ]

    for basename, img, inc, emi, phs in images:
        h, w = img.shape
        # Save PNG for quick visualization
        png_path = os.path.join(output_dir, f"{basename}.png")
        cv2.imwrite(png_path, (img * 255).astype(np.uint8))

        # Save raw float32 binary
        raw_path = os.path.join(output_dir, f"{basename}.raw")
        img.astype(">f4").tofile(raw_path)

        # Save standard PDS4 XML label
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <logical_identifier>urn:nasa:pds:lunar_crater_recon:{basename}</logical_identifier>
        <version_id>1.0</version_id>
        <title>{basename} Lunar Narrow Angle Camera Science Image</title>
        <information_model_version>1.14.0.0</information_model_version>
        <product_class>Product_Observational</product_class>
    </Identification_Area>
    <Observation_Area>
        <comment>Lunar surface observation for crater detection and geometric registration</comment>
        <Discipline_Area>
            <Geometry>
                <solar_incidence_angle unit="deg">{inc}</solar_incidence_angle>
                <emission_angle unit="deg">{emi}</emission_angle>
                <phase_angle unit="deg">{phs}</phase_angle>
            </Geometry>
        </Discipline_Area>
    </Observation_Area>
    <File_Area_Observational>
        <File>
            <file_name>{basename}.raw</file_name>
            <file_size>{h * w * 4}</file_size>
        </File>
        <Array_2D_Image>
            <offset unit="byte">0</offset>
            <axes>2</axes>
            <axis_index_order>Last_Index_Fastest</axis_index_order>
            <element_data_type>IEEE754MSBSingle</element_data_type>
            <scaling_factor>1.0</scaling_factor>
            <value_offset>0.0</value_offset>
            <Axis_Array>
                <axis_name>Line</axis_name>
                <elements>{h}</elements>
                <sequence_number>1</sequence_number>
            </Axis_Array>
            <Axis_Array>
                <axis_name>Sample</axis_name>
                <elements>{w}</elements>
                <sequence_number>2</sequence_number>
            </Axis_Array>
        </Array_2D_Image>
    </File_Area_Observational>
</Product_Observational>
"""
        xml_path = os.path.join(output_dir, f"{basename}.xml")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

    print(f"Generated {len(images)} PDS4 & raster lunar images in {output_dir}")

if __name__ == "__main__":
    create_pds4_dataset()
