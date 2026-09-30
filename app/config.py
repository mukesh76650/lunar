"""Pipeline Configuration and Hyperparameters."""
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

@dataclass
class PreprocessingConfig:
    """Parameters for radiometric calibration, Hapke correction, and denoising."""
    apply_calibration: bool = True
    calibration_scale: float = 1.0
    calibration_offset: float = 0.0
    
    # Hapke photometric correction (removed from active preprocessing pipeline)
    apply_hapke: bool = False
    default_incidence_deg: float = 30.0   # Sun incidence angle (i)
    default_emission_deg: float = 0.0     # Viewing emission angle (e)
    default_phase_deg: float = 30.0       # Phase angle (g)
    hapke_b: float = 0.25                 # Particle phase function asymmetry factor
    hapke_c: float = 0.70                 # Forward/backward scattering ratio
    
    # Non-Local Means (NLM) denoising with OpenCV parallel optimization
    apply_nlm: bool = True
    nlm_h: float = 8.0                    # Filter strength (larger removes more noise, preserves fewer edges)
    nlm_template_window_size: int = 7     # Size in pixels of template patch
    nlm_search_window_size: int = 21      # Size in pixels of search window area
    use_opencv_parallel: bool = True      # Enable cv2.setUseOptimized and cv2.setNumThreads
    opencv_threads: int = 0               # 0 = auto-detect all available CPU cores
    
    # Percentile Normalization
    apply_percentile_norm: bool = True
    percentile_low: float = 1.0           # Lower clip percentile
    percentile_high: float = 99.0         # Upper clip percentile

@dataclass
class SuperPointConfig:
    """SuperPoint feature detector and descriptor hyperparameters."""
    keypoint_threshold: float = 0.015
    nms_radius: int = 4
    max_keypoints: int = 350
    descriptor_dim: int = 256
    match_ratio_thresh: float = 0.78      # Mutual nearest neighbor / ratio test threshold
    mutual_check: bool = True

@dataclass
class LoFTRConfig:
    """LoFTR detector-free learned correspondence hyperparameters."""
    match_threshold: float = 0.35
    coarse_scale: int = 8
    temperature: float = 0.1
    top_k_matches: int = 400

@dataclass
class RANSACConfig:
    """RANSAC geometric verification settings."""
    reprojection_threshold: float = 3.5   # Maximum reprojection error in pixels to consider as inlier
    confidence: float = 0.999             # Desired probability of finding a good homography
    max_iters: int = 5000                 # Maximum RANSAC iterations
    min_inliers_required: int = 10        # Minimum inliers required to declare valid transformation

@dataclass
class EvaluationWeights:
    """Weights for the composite reliability score between SuperPoint and LoFTR."""
    w_inlier_ratio: float = 0.35
    w_inlier_count: float = 0.20
    w_spatial_coverage: float = 0.30
    w_rmse_penalty: float = 0.15
    min_acceptable_inliers: int = 12
    max_acceptable_rmse: float = 4.5
    min_acceptable_hull_ratio: float = 0.05

@dataclass
class CraterDetectorConfig:
    """Morphological and circular/elliptical crater detection settings."""
    min_radius: int = 8
    max_radius: int = 120
    dp: float = 1.2
    param1: float = 50.0                  # Higher threshold for Canny edge detector
    param2: float = 38.0                  # Accumulator threshold for circle centers (increased to eliminate noise)
    rim_gradient_threshold: float = 20.0
    min_confidence: float = 0.60          # Require prominent morphological structure
    max_craters: int = 30                 # Cap to most prominent craters per image

@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    superpoint: SuperPointConfig = field(default_factory=SuperPointConfig)
    loftr: LoFTRConfig = field(default_factory=LoFTRConfig)
    ransac: RANSACConfig = field(default_factory=RANSACConfig)
    evaluation: EvaluationWeights = field(default_factory=EvaluationWeights)
    crater_detector: CraterDetectorConfig = field(default_factory=CraterDetectorConfig)
