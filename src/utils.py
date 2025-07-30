import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

# --- Conversion Functions ---
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Converts a PIL Image to an OpenCV (NumPy array) image."""
    # Convert PIL Image to NumPy array
    np_img = np.array(pil_img)

    # PIL uses RGB, OpenCV uses BGR. Convert if needed.
    # Also handle potential alpha channel
    if np_img.ndim == 3: # Color image
        if np_img.shape[2] == 4: # RGBA to BGRA
            return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
        elif np_img.shape[2] == 3: # RGB to BGR
            return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img # Grayscale or other formats

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Converts an OpenCV (NumPy array) image to a PIL Image."""
    # OpenCV uses BGR, PIL uses RGB. Convert if needed.
    # Also handle potential alpha channel
    if cv2_img.ndim == 3: # Color image
        if cv2_img.shape[2] == 4: # BGRA to RGBA
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))
        elif cv2_img.shape[2] == 3: # BGR to RGB
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2_img) # Grayscale or other formats

# --- Image Loading/Saving ---
def load_image_cv2(image_path: str) -> np.ndarray:
    """
    Loads an image using OpenCV and ensures it has an alpha channel if it's a PNG.
    Returns a NumPy array in BGRA format if transparency is present.
    """
    try:
        # IMREAD_UNCHANGED ensures alpha channel is read if present (e.g., for PNGs)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {image_path}")

        # If image has 3 channels (BGR), convert to BGRA for consistent alpha handling
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        return img
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")

def save_image_cv2(image_np: np.ndarray, output_path: str):
    """
    Saves an OpenCV (NumPy array) image to the specified path.
    Handles transparency automatically if the image has an alpha channel (4 channels).
    """
    try:
        cv2.imwrite(output_path, image_np)
    except Exception as e:
        raise ValueError(f"Error saving image to {output_path}: {e}")

# --- Advanced Image Manipulation ---

def apply_perspective_transform_sticker(
    sticker_img_cv2: np.ndarray,
    target_points: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    output_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a perspective transformation to a sticker image (BGRA) and its alpha channel separately.
    This method is more robust than warping the BGRA image directly.

    Args:
        sticker_img_cv2: The sticker image as an OpenCV (BGRA) NumPy array.
        target_points: A tuple of 4 (x, y) coordinates for the destination.
        output_size: The (width, height) of the output canvas.

    Returns:
        A tuple containing:
        - warped_sticker (np.ndarray): The transformed sticker with a transparent background (BGRA).
        - warped_mask (np.ndarray): The transformed alpha channel as a grayscale mask (0-255).
    """
    if sticker_img_cv2.shape[2] != 4:
        raise ValueError("Sticker image must be in BGRA format (4 channels).")

    # 1. Split the sticker into its color (BGR) and alpha channels
    bgr_sticker = sticker_img_cv2[:, :, :3]
    alpha_mask = sticker_img_cv2[:, :, 3]

    # 2. Define source points from the sticker's dimensions
    h_sticker, w_sticker = sticker_img_cv2.shape[:2]
    src_points = np.float32([
        [0, 0],
        [w_sticker - 1, 0],
        [w_sticker - 1, h_sticker - 1],
        [0, h_sticker - 1]
    ])
    dst_points = np.float32(target_points)

    # 3. Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 4. Warp the BGR color part of the sticker
    warped_bgr = cv2.warpPerspective(
        bgr_sticker,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0] # Fill with black outside
    )

    # 5. Warp the alpha mask
    # This will become our final mask for inpainting
    warped_mask = cv2.warpPerspective(
        alpha_mask,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0 # Fill with black (transparent)
    )

    # 6. Create the final warped sticker (BGRA)
    # Merge the warped BGR with the warped alpha mask
    warped_sticker = cv2.merge([warped_bgr[:, :, 0], warped_bgr[:, :, 1], warped_bgr[:, :, 2], warped_mask])
    
    # Where the mask is 0, make the color black to avoid strange fringes.
    warped_sticker[warped_mask == 0] = [0, 0, 0, 0]

    return warped_sticker, warped_mask

def blend_images_alpha(background_cv2: np.ndarray, foreground_cv2: np.ndarray) -> np.ndarray:
    """
    Blends a foreground image (with alpha) onto a background image.
    """
    # Ensure background is BGRA
    if background_cv2.shape[2] == 3:
        background_cv2 = cv2.cvtColor(background_cv2, cv2.COLOR_BGR2BGRA)

    # Normalize alpha channel to 0-1
    alpha = foreground_cv2[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha

    # Perform blending for each channel
    blended_bgr = np.zeros_like(background_cv2[:,:,:3])
    for i in range(3): # B, G, R channels
        blended_bgr[:,:,i] = (alpha * foreground_cv2[:,:,i] + alpha_inv * background_cv2[:,:,i])

    # Merge blended BGR with background alpha
    blended_bgra = cv2.merge((blended_bgr.astype(np.uint8), background_cv2[:,:,3]))

    return blended_bgra