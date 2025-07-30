from .state import GraphState
from .utils import (
    pil_to_cv2,
    cv2_to_pil,
    save_image_cv2,
    apply_perspective_transform_sticker,
    blend_images_alpha,
)
from .presets import DEMO_PLACEMENT
from PIL import Image
import numpy as np
import os
import replicate
import requests
from typing import Dict, Any, Literal

# --- Conditional Router ---

def route_by_message(state: GraphState) -> Literal["generative_api", "placement"]:
    """
    This function determines which path to take based on user input.
    - If a 'message' (text prompt) is provided, it routes to the 'generative_api' path.
    - Otherwise, it routes to the 'placement' path for existing stickers.
    """
    print("---Routing by user input---")
    if state.get("message"):
        print("Message found, routing to: Generative API")
        return "generative_api"
    print("No message, routing to: Simple Placement")
    return "placement"

# --- Nodes for the Graph ---

def load_assets_node(state: GraphState) -> GraphState:
    """
    Loads all preset assets for the demo: guitar image, sticker image,
    and the hardcoded coordinates for placement.
    """
    print("---Loading Demo Assets---")
    
    guitar_path = "images/guitar.png"
    sticker_path = "images/sticker.png"
    output_path = "output/final_demo.png"
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return {
            "guitar_image_path": guitar_path,
            "sticker_image_path": sticker_path,
            "output_path": output_path,
        }
    except Exception as e:
        return {"error": f"Failed to load assets: {e}"}

def prepare_assets_node(state: GraphState) -> GraphState:
    """
    Prepares the assets needed for BOTH branches:
    1. Creates the warped sticker (`processed_sticker.png`).
    2. Creates the mask for the generative path (`mask.png`).
    """
    print("---Preparing Warped Sticker and Mask---")
    sticker_path = state["sticker_image_path"]
    guitar_path = state["guitar_image_path"]
    output_path = state["output_path"]
    proportional_points = DEMO_PLACEMENT["points_proportional"]

    try:
        sticker_pil = Image.open(sticker_path).convert("RGBA")
        guitar_pil = Image.open(guitar_path).convert("RGBA")
        
        output_dir = os.path.dirname(output_path)
        processed_sticker_path = os.path.join(output_dir, "processed_sticker.png")
        mask_path = os.path.join(output_dir, "mask.png")

        guitar_width, guitar_height = guitar_pil.size

        transform_points_pixels = [
            (int(p[0] * guitar_width), int(p[1] * guitar_height))
            for p in proportional_points
        ]
        
        sticker_cv2 = pil_to_cv2(sticker_pil)

        processed_sticker_cv2, processed_mask_cv2 = apply_perspective_transform_sticker(
            sticker_cv2, tuple(transform_points_pixels), (guitar_width, guitar_height)
        )

        save_image_cv2(processed_sticker_cv2, processed_sticker_path)
        save_image_cv2(processed_mask_cv2, mask_path)

        print(f"Saved processed sticker: {processed_sticker_path}")
        print(f"Saved mask: {mask_path}")
        
        return {
            "processed_sticker_path": processed_sticker_path,
            "mask_path": mask_path,
            "final_pil_image": None # ensure this is cleared
        }
    except Exception as e:
        return {"error": f"Failed to prepare assets: {e}"}

def generative_api_node(state: GraphState) -> GraphState:
    """
    Calls the Replicate API with a DYNAMIC prompt based on user's message.
    """
    print("---Calling Generative API---")
    guitar_path = state["guitar_image_path"]
    mask_path = state["mask_path"]
    user_message = state.get("message", "a cool sticker") # Default prompt

    if not os.environ.get("REPLICATE_API_TOKEN"):
        return {"error": "REPLICATE_API_TOKEN environment variable not set."}

    try:
        # Dynamic prompt combining user message with style instructions
        prompt = f"A high-resolution, photorealistic image of '{user_message}', seamlessly blended onto the glossy surface of a white electric guitar, matching the ambient studio lighting and reflections."
        print(f"Using dynamic prompt: {prompt}")

        output = replicate.run(
            "anotherjesse/controlnet-inpaint-test:3a294336beb5dcaba6c5baf1418058171298d559e4eef8dafed1e0a8c3594984",
            input={
                "image": open(guitar_path, "rb"),
                "mask": open(mask_path, "rb"),
                "prompt": prompt,
                "control_image": open(guitar_path, "rb"),
            }
        )
        
        final_image_url = next(output)
        print(f"Image generated successfully. URL: {final_image_url}")
        return {"final_image_url": final_image_url}
    except Exception as e:
        return {"error": f"Failed to call Replicate API: {e}"}

def placement_node(state: GraphState) -> GraphState:
    """
    Performs a simple OpenCV alpha blend to place the existing sticker.
    This does not produce photorealistic results but demonstrates the path.
    """
    print("---Performing Simple Sticker Placement---")
    try:
        guitar_path = state["guitar_image_path"]
        processed_sticker_path = state["processed_sticker_path"]

        # Load images with PIL for consistency, then convert
        guitar_pil = Image.open(guitar_path).convert("RGBA")
        processed_sticker_pil = Image.open(processed_sticker_path).convert("RGBA")

        # Convert to OpenCV format for blending
        guitar_cv2 = pil_to_cv2(guitar_pil)
        processed_sticker_cv2 = pil_to_cv2(processed_sticker_pil)

        # Blend the images
        final_image_cv2 = blend_images_alpha(guitar_cv2, processed_sticker_cv2)

        # Convert back to PIL for saving
        final_pil_image = cv2_to_pil(final_image_cv2)

        return {"final_pil_image": final_pil_image}
    except Exception as e:
        return {"error": f"Failed during placement blending: {e}"}


def save_result_node(state: GraphState) -> GraphState:
    """
    Saves the final image. It can handle two cases:
    1. Downloading from a URL (from the generative path).
    2. Saving a local PIL image (from the placement path).
    """
    print("---Saving Final Result---")
    final_image_url = state.get("final_image_url")
    final_pil_image = state.get("final_pil_image")
    output_path = state.get("output_path")

    if not output_path:
         return {"error": "Missing output path for saving."}

    if not final_image_url and not final_pil_image:
        return {"error": "No final image found to save."}

    try:
        if final_image_url:
            print(f"Downloading final image from {final_image_url}...")
            response = requests.get(final_image_url, stream=True)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        elif final_pil_image:
            print(f"Saving local image to {output_path}...")
            final_pil_image.save(output_path)

        print(f"Final image saved to: {output_path}")
        return {}
    except Exception as e:
        return {"error": f"Failed to save final image: {e}"}