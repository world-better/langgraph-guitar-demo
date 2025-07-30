from typing import Optional, Dict, Any, Tuple
from PIL import Image # We'll still use PIL Image objects in the state for compatibility/easier handling
from langgraph.graph.message import AnyMessage
from typing_extensions import TypedDict, NotRequired # Needed for Python < 3.11

class GraphState(TypedDict):
    """
    Represents the state of our graph for the guitar sticker application.
    This state supports two conditional paths: placing a pre-existing sticker
    or generating a new one from a text prompt.
    """
    # --- User Input ---
    # The user's text prompt. If provided, triggers the generative path.
    message: NotRequired[Optional[str]]

    # --- Input paths (using defaults) ---
    guitar_image_path: NotRequired[Optional[str]]
    sticker_image_path: NotRequired[Optional[str]]
    output_path: NotRequired[Optional[str]]

    # --- Intermediate paths generated during the workflow ---
    processed_sticker_path: NotRequired[Optional[str]]
    mask_path: NotRequired[Optional[str]]
    
    # --- Final result from the API ---
    final_image_url: NotRequired[Optional[str]]
    
    # --- In-memory objects for processing ---
    # These are loaded/generated and used by nodes, but not primary inputs.
    final_pil_image: NotRequired[Optional[Image.Image]]
    
    # --- General fields ---
    error: NotRequired[Optional[str]]
    messages: NotRequired[list[AnyMessage]]

    # --- Deprecated fields from LLM version ---
    # sticker_transform_points is now hardcoded in presets.
    # Other fields are no longer needed.