"""Anchor-Based Image Pairing Module.

Designates one lunar image as the anchor/reference frame, and builds pairs:
(Anchor, Image 1), (Anchor, Image 2), ..., (Anchor, Image N).
All target images are registered into the anchor's coordinate system.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

@dataclass
class LunarImageItem:
    id: str
    name: str
    image: np.ndarray      # Preprocessed 2D float32 image [0, 1]
    raw_image: np.ndarray  # Original image
    metadata: Dict[str, Any]

@dataclass
class ImagePair:
    pair_id: str
    anchor: LunarImageItem
    target: LunarImageItem
    index: int             # Target index relative to inputs

class AnchorPairingManager:
    """Manages anchor selection and constructs pairwise batches against the anchor."""

    @staticmethod
    def create_pairs(
        items: List[LunarImageItem],
        anchor_id_or_index: Optional[Any] = None
    ) -> Tuple[LunarImageItem, List[ImagePair]]:
        """Create pairs of (Anchor, Target_k).
        
        Args:
            items: List of LunarImageItems
            anchor_id_or_index: Index (int) or ID/name (str) of designated anchor.
                                If None, defaults to the first image (index 0).
        Returns:
            anchor: The selected anchor item
            pairs: List of ImagePair objects
        """
        if not items:
            raise ValueError("At least one image item is required for pairing.")

        # Resolve anchor index
        anchor_idx = 0
        if anchor_id_or_index is not None:
            if isinstance(anchor_id_or_index, int):
                if 0 <= anchor_id_or_index < len(items):
                    anchor_idx = anchor_id_or_index
            elif isinstance(anchor_id_or_index, str):
                for i, itm in enumerate(items):
                    if itm.id == anchor_id_or_index or itm.name == anchor_id_or_index:
                        anchor_idx = i
                        break

        anchor_item = items[anchor_idx]
        pairs: List[ImagePair] = []

        for i, target_item in enumerate(items):
            if i == anchor_idx:
                continue
            pair = ImagePair(
                pair_id=f"pair_{anchor_item.id}_to_{target_item.id}",
                anchor=anchor_item,
                target=target_item,
                index=i
            )
            pairs.append(pair)

        return anchor_item, pairs
