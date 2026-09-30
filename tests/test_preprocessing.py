import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from app.pipeline.pds4_reader import LunarImageProduct, PDS4Metadata
from app.pipeline.preprocessor import ImagePreprocessor
from app.config import PreprocessingConfig

def test_preprocessing_flow():
    # Create synthetic lunar surface patch with gradient and noise
    np.random.seed(42)
    h, w = 128, 128
    y, x = np.mgrid[0:h, 0:w]
    synthetic_lunar = np.sin(x / 10.0) * np.cos(y / 10.0) * 0.5 + 0.5
    # Add speckle noise
    noise = np.random.normal(0, 0.05, (h, w))
    raw_img = np.clip(synthetic_lunar + noise, 0.0, 1.0).astype(np.float32)

    meta = PDS4Metadata(
        title="Synthetic_Lunar_Test",
        lines=h,
        samples=w,
        incidence_angle_deg=45.0,
        emission_angle_deg=10.0,
        phase_angle_deg=35.0
    )
    product = LunarImageProduct(
        name="test_prod.xml",
        image=raw_img,
        raw_image=raw_img,
        metadata=meta
    )

    preprocessor = ImagePreprocessor()
    processed, log = preprocessor.process(product)

    assert processed.shape == (h, w)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
    assert log["hapke"]["applied"] is False
    assert log["nlm_denoise"]["applied"] is True
    assert log["nlm_denoise"]["native_resolution"] is True
    assert log["nlm_denoise"]["opencv_parallel"] is True
    assert log["percentile_norm"]["applied"] is True
    print("Preprocessing unit test passed successfully (Hapke removed, parallel native NLM verified)!")

if __name__ == "__main__":
    test_preprocessing_flow()
