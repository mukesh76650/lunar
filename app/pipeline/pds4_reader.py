"""PDS4 (Planetary Data System v4) and Lunar Image Reader.

Parses PDS4 XML labels, loads raster/binary arrays, and extracts
viewing geometry (incidence, emission, and phase angles).
Also provides transparent fallback for standard lunar imagery (GeoTIFF, PNG, JPEG).
"""
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import cv2

@dataclass
class PDS4Metadata:
    title: str = "Lunar Observation"
    product_id: str = "unknown"
    instrument: str = "LROC/Clementine/Chandrayaan"
    target: str = "Moon"
    lines: int = 0
    samples: int = 0
    data_type: str = "float32"
    byte_offset: int = 0
    scaling_factor: float = 1.0
    scaling_offset: float = 0.0
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    raw_tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class LunarImageProduct:
    name: str
    image: np.ndarray          # 2D float32 array normalized to [0, 1] or raw radiances
    raw_image: np.ndarray      # Original raw values
    metadata: PDS4Metadata
    file_path: Optional[str] = None

class PDS4Reader:
    """Robust parser for PDS4 XML metadata and associated image data."""
    
    PDS4_DTYPE_MAP = {
        "IEEE754MSBSingle": ">f4",
        "IEEE754LSBSingle": "<f4",
        "IEEE754MSBDouble": ">f8",
        "IEEE754LSBDouble": "<f8",
        "SignedMSB2": ">i2",
        "SignedLSB2": "<i2",
        "UnsignedMSB2": ">u2",
        "UnsignedLSB2": "<u2",
        "SignedByte": "i1",
        "UnsignedByte": "u1",
        "SignedMSB4": ">i4",
        "SignedLSB4": "<i4",
    }
    
    @classmethod
    def parse_xml_label(cls, xml_content: Union[str, bytes]) -> PDS4Metadata:
        """Parse PDS4 XML label string or bytes."""
        if isinstance(xml_content, str):
            root = ET.fromstring(xml_content)
        else:
            root = ET.fromstring(xml_content.decode('utf-8', errors='ignore'))
            
        # Strip XML namespaces for simplified element access
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
                
        meta = PDS4Metadata()
        
        # Product Identification
        prod_id = root.find(".//logical_identifier")
        if prod_id is not None and prod_id.text:
            meta.product_id = prod_id.text.strip()
            
        title = root.find(".//title")
        if title is not None and title.text:
            meta.title = title.text.strip()
            
        # Array 2D Image parameters
        array = root.find(".//Array_2D_Image")
        if array is not None:
            # Data type
            elem_dtype = array.find(".//element_data_type")
            if elem_dtype is not None and elem_dtype.text:
                meta.data_type = elem_dtype.text.strip()
                
            # Offset
            offset = array.find(".//offset")
            if offset is not None and offset.text:
                try:
                    meta.byte_offset = int(offset.text.strip())
                except ValueError:
                    pass
                    
            # Axes
            for axis in array.findall(".//Axis_Array"):
                axis_name = axis.find("axis_name")
                elements = axis.find("elements")
                if axis_name is not None and elements is not None and axis_name.text and elements.text:
                    name = axis_name.text.lower().strip()
                    val = int(elements.text.strip())
                    if "line" in name:
                        meta.lines = val
                    elif "sample" in name:
                        meta.samples = val

        # Illumination / Viewing geometry
        for tag in ["solar_incidence_angle", "incidence_angle", "Incidence_Angle"]:
            elem = root.find(f".//{tag}")
            if elem is not None and elem.text:
                try:
                    meta.incidence_angle_deg = float(elem.text.strip())
                    break
                except ValueError:
                    pass

        for tag in ["emission_angle", "Emission_Angle"]:
            elem = root.find(f".//{tag}")
            if elem is not None and elem.text:
                try:
                    meta.emission_angle_deg = float(elem.text.strip())
                    break
                except ValueError:
                    pass

        for tag in ["phase_angle", "Phase_Angle"]:
            elem = root.find(f".//{tag}")
            if elem is not None and elem.text:
                try:
                    meta.phase_angle_deg = float(elem.text.strip())
                    break
                except ValueError:
                    pass

        # Calibration scaling factors
        scale = root.find(".//scaling_factor")
        if scale is not None and scale.text:
            try:
                meta.scaling_factor = float(scale.text.strip())
            except ValueError:
                pass
                
        offset_elem = root.find(".//value_offset")
        if offset_elem is not None and offset_elem.text:
            try:
                meta.scaling_offset = float(offset_elem.text.strip())
            except ValueError:
                pass

        return meta

    @classmethod
    def load_pds4_pair(cls, xml_path: str, data_path: Optional[str] = None) -> LunarImageProduct:
        """Load PDS4 dataset from an XML label and data file."""
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            xml_text = f.read()
        meta = cls.parse_xml_label(xml_text)
        
        # If data_path not provided, try inferring from filename
        if data_path is None:
            base, _ = os.path.splitext(xml_path)
            for ext in [".raw", ".dat", ".img", ".bin", ".tif", ".tiff", ".fit"]:
                cand = base + ext
                if os.path.exists(cand):
                    data_path = cand
                    break
                    
        if data_path is None or not os.path.exists(data_path):
            raise FileNotFoundError(f"Associated PDS4 data file not found for label {xml_path}")

        # Check if data_path is a regular image
        ext = os.path.splitext(data_path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            raw_img = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
            if raw_img is None:
                raise ValueError(f"Could not read image file {data_path}")
            if raw_img.ndim == 3:
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            raw_img = raw_img.astype(np.float32)
        else:
            # Read binary raw array
            np_dtype = cls.PDS4_DTYPE_MAP.get(meta.data_type, "<f4")
            with open(data_path, "rb") as f:
                f.seek(meta.byte_offset)
                raw_bytes = f.read()
            arr = np.frombuffer(raw_bytes, dtype=np_dtype)
            if meta.lines > 0 and meta.samples > 0:
                raw_img = arr[:meta.lines * meta.samples].reshape((meta.lines, meta.samples)).astype(np.float32)
            else:
                # Square assumption fallback
                dim = int(np.sqrt(len(arr)))
                raw_img = arr[:dim*dim].reshape((dim, dim)).astype(np.float32)

        # Apply scaling
        img = raw_img * meta.scaling_factor + meta.scaling_offset
        # Ensure finite
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        
        return LunarImageProduct(
            name=os.path.basename(xml_path),
            image=img,
            raw_image=raw_img,
            metadata=meta,
            file_path=data_path
        )

    @classmethod
    def load_from_image_file(cls, file_path: str, name: Optional[str] = None) -> LunarImageProduct:
        """Load standard image file (PNG, JPG, TIFF) as LunarImageProduct."""
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image from {file_path}")
            
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                img = img.squeeze(2)
            
        img_f32 = img.astype(np.float32)
        # Normalize to [0, 1] if uint8/uint16
        if img.dtype == np.uint8:
            norm_img = img_f32 / 255.0
        elif img.dtype == np.uint16:
            norm_img = img_f32 / 65535.0
        else:
            # Min-max scaling
            vmin, vmax = img_f32.min(), img_f32.max()
            norm_img = (img_f32 - vmin) / (vmax - vmin + 1e-7)

        meta = PDS4Metadata(
            title=name or os.path.basename(file_path),
            lines=img.shape[0],
            samples=img.shape[1],
            data_type=str(img.dtype)
        )
        
        return LunarImageProduct(
            name=name or os.path.basename(file_path),
            image=norm_img,
            raw_image=img_f32,
            metadata=meta,
            file_path=file_path
        )

    @classmethod
    def load_from_bytes(cls, data: bytes, filename: str) -> LunarImageProduct:
        """Load from in-memory bytes (uploaded via Web API)."""
        # If XML, caller should handle PDS4 pairing, or we check if binary/image
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image bytes for {filename}")
            
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                img = img.squeeze(2)
            
        img_f32 = img.astype(np.float32)
        if img.dtype == np.uint8:
            norm_img = img_f32 / 255.0
        else:
            vmin, vmax = img_f32.min(), img_f32.max()
            norm_img = (img_f32 - vmin) / (vmax - vmin + 1e-7)

        meta = PDS4Metadata(
            title=filename,
            lines=img.shape[0],
            samples=img.shape[1],
            data_type=str(img.dtype)
        )
        return LunarImageProduct(
            name=filename,
            image=norm_img,
            raw_image=img_f32,
            metadata=meta
        )
