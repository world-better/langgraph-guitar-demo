# src/presets.py
from typing import Dict, Any, Tuple

# --- Preset Placement Definition for the Demo ---
# This file now contains a single, hardcoded placement area using
# PROPORTIONAL coordinates (values between 0.0 and 1.0).
# This makes the placement independent of the actual image resolution.
#
# The coordinates are ordered: top-left, top-right, bottom-right, bottom-left.
# Each point is a tuple of (proportion_of_width, proportion_of_height).

DEMO_PLACEMENT: Dict[str, Any] = {
    "name": "Demo Placement on Lower Bout",
    # Updated coordinates estimated for the new 500x500 square image format.
    # Adjusted to be smaller and higher, per user feedback.
    "points_proportional": ((0.51, 0.18), (0.79, 0.18), (0.83, 0.53), (0.45, 0.54))
} 