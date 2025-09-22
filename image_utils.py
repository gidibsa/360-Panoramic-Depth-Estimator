"""
Image processing utilities for 360° panoramic depth estimation.
Provides image conversion, validation, and processing functions used throughout the pipeline.
Maintains native resolution processing without downscaling.
"""

import numpy as np
import cv2
from PIL import Image
import logging
from typing import List, Tuple, Optional, Union
import os
from pathlib import Path
import warnings

# Embedded constants for image processing utilities
IMAGE_CONFIG = {
    'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'],
    'max_channels': 3,
    'min_resolution': (512, 256),  # (width, height)
    'depth_scale_factor': 65535,  # For 16-bit conversion (2^16 - 1)
    'interpolation_method': cv2.INTER_LINEAR,
    'panoramic_aspect_ratio_tolerance': 0.3,  # Tolerance for 2:1 aspect ratio
    'default_bit_depth': 8,
    'depth_bit_depth': 16,
    'gamma_correction_default': 1.0,
    'normalize_range': [0.0, 1.0]
}

def load_and_validate_image(image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Load and validate image from various input types with comprehensive validation.
    
    Args:
        image_input: Input image (file path, PIL Image, or numpy array)
        
    Returns:
        Validated PIL Image in RGB format
        
    Raises:
        ValueError: If image is invalid or unsupported format
        FileNotFoundError: If file path doesn't exist
        IOError: If image cannot be loaded
    """
    logging.info("Loading and validating image")
    
    try:
        if isinstance(image_input, str):
            # Handle file path input
            image_path = Path(image_input)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_input}")
            
            # Check file extension
            file_ext = image_path.suffix.lower()
            if file_ext not in IMAGE_CONFIG['supported_formats']:
                raise ValueError(f"Unsupported image format: {file_ext}. Supported formats: {IMAGE_CONFIG['supported_formats']}")
            
            # Load image
            pil_image = Image.open(image_path)
            logging.info(f"Successfully loaded image from file: {image_input}")
            
        elif isinstance(image_input, Image.Image):
            # Handle PIL Image input
            pil_image = image_input.copy()  # Create copy to avoid modifying original
            logging.info("Using provided PIL Image")
            
        elif isinstance(image_input, np.ndarray):
            # Handle numpy array input
            pil_image = convert_numpy_to_pil(image_input)
            logging.info("Converted numpy array to PIL Image")
            
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}. Expected str, PIL.Image, or numpy.ndarray")
        
        # Validate and convert to RGB
        if pil_image.mode not in ['RGB', 'RGBA', 'L']:
            logging.warning(f"Converting image from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'RGBA':
            logging.info("Converting RGBA to RGB (removing alpha channel)")
            # Create white background and paste RGBA image
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            rgb_image.paste(pil_image, mask=pil_image.split()[-1])  # Use alpha as mask
            pil_image = rgb_image
        elif pil_image.mode == 'L':
            logging.info("Converting grayscale to RGB")
            pil_image = pil_image.convert('RGB')
        
        # Validate image dimensions
        width, height = pil_image.size
        min_width, min_height = IMAGE_CONFIG['min_resolution']
        
        if width < min_width or height < min_height:
            raise ValueError(f"Image resolution too small: {width}x{height}. Minimum required: {min_width}x{min_height}")
        
        logging.info(f"Image validation successful: {width}x{height}, mode: {pil_image.mode}")
        return pil_image
        
    except Exception as e:
        logging.error(f"Failed to load and validate image: {e}")
        raise

def validate_panoramic_format(image: Image.Image) -> bool:
    """
    Validate if image follows panoramic/equirectangular format (roughly 2:1 aspect ratio).
    
    Args:
        image: PIL Image to validate
        
    Returns:
        True if image appears to be in panoramic format, False otherwise
    """
    width, height = image.size
    aspect_ratio = width / height
    expected_ratio = 2.0
    tolerance = IMAGE_CONFIG['panoramic_aspect_ratio_tolerance']
    
    is_panoramic = abs(aspect_ratio - expected_ratio) <= tolerance
    
    if is_panoramic:
        logging.info(f"Image aspect ratio {aspect_ratio:.2f} is valid for panoramic format")
    else:
        logging.warning(f"Image aspect ratio {aspect_ratio:.2f} deviates from expected panoramic ratio {expected_ratio} (tolerance: ±{tolerance})")
    
    return is_panoramic

def resize_maintain_aspect(image: Union[Image.Image, np.ndarray], 
                          target_size: Optional[Tuple[int, int]] = None,
                          max_dimension: Optional[int] = None) -> Union[Image.Image, np.ndarray]:
    """
    Resize image while maintaining aspect ratio (NOT USED - for completeness only).
    This function maintains the aspect ratio but our pipeline preserves native resolution.
    
    Args:
        image: Input image (PIL or numpy array)
        target_size: Target (width, height) - if None, uses max_dimension
        max_dimension: Maximum dimension for largest side
        
    Returns:
        Resized image of same type as input
        
    Note:
        This function is provided for completeness but should NOT be used in the main pipeline
        as we preserve native resolution throughout processing.
    """
    logging.warning("resize_maintain_aspect called - This should NOT be used in native resolution pipeline")
    
    if isinstance(image, Image.Image):
        original_size = image.size
        if target_size:
            # Calculate size maintaining aspect ratio
            original_ratio = original_size[0] / original_size[1]
            target_ratio = target_size[0] / target_size[1]
            
            if original_ratio > target_ratio:
                # Image is wider, fit to width
                new_width = target_size[0]
                new_height = int(target_size[0] / original_ratio)
            else:
                # Image is taller, fit to height
                new_height = target_size[1]
                new_width = int(target_size[1] * original_ratio)
            
            return image.resize((new_width, new_height), Image.LANCZOS)
        
        elif max_dimension:
            # Resize based on maximum dimension
            if max(original_size) > max_dimension:
                if original_size[0] > original_size[1]:
                    new_width = max_dimension
                    new_height = int(max_dimension * original_size[1] / original_size[0])
                else:
                    new_height = max_dimension
                    new_width = int(max_dimension * original_size[0] / original_size[1])
                
                return image.resize((new_width, new_height), Image.LANCZOS)
    
    return image  # Return original if no resizing needed

def convert_pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array with proper dtype handling.
    
    Args:
        pil_image: Input PIL Image
        
    Returns:
        Numpy array representation of the image
    """
    logging.debug(f"Converting PIL Image to numpy array: {pil_image.size}, mode: {pil_image.mode}")
    
    try:
        # Convert to numpy array
        np_array = np.array(pil_image)
        
        # Ensure 3 channels for RGB
        if len(np_array.shape) == 2:
            # Grayscale - expand to 3 channels
            np_array = np.stack([np_array] * 3, axis=-1)
        elif len(np_array.shape) == 3 and np_array.shape[2] == 4:
            # RGBA - remove alpha channel
            np_array = np_array[:, :, :3]
        
        # Ensure uint8 dtype
        if np_array.dtype != np.uint8:
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                # Assume float values are in [0, 1] range
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)
        
        logging.debug(f"Converted to numpy array: shape {np_array.shape}, dtype {np_array.dtype}")
        return np_array
        
    except Exception as e:
        logging.error(f"Failed to convert PIL to numpy: {e}")
        raise RuntimeError(f"PIL to numpy conversion failed: {e}")

def convert_numpy_to_pil(numpy_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image with proper dtype and format handling.
    
    Args:
        numpy_array: Input numpy array
        
    Returns:
        PIL Image representation
    """
    logging.debug(f"Converting numpy array to PIL Image: shape {numpy_array.shape}, dtype {numpy_array.dtype}")
    
    try:
        # Handle different input shapes and dtypes
        if len(numpy_array.shape) == 2:
            # Grayscale image
            if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
                # Normalize float values to [0, 255]
                numpy_array = (numpy_array * 255).astype(np.uint8)
            elif numpy_array.dtype != np.uint8:
                numpy_array = numpy_array.astype(np.uint8)
            
            pil_image = Image.fromarray(numpy_array, mode='L')
            
        elif len(numpy_array.shape) == 3:
            if numpy_array.shape[2] == 1:
                # Single channel - convert to grayscale
                numpy_array = numpy_array.squeeze(axis=2)
                if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
                    numpy_array = (numpy_array * 255).astype(np.uint8)
                elif numpy_array.dtype != np.uint8:
                    numpy_array = numpy_array.astype(np.uint8)
                pil_image = Image.fromarray(numpy_array, mode='L')
                
            elif numpy_array.shape[2] == 3:
                # RGB image
                if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
                    numpy_array = (numpy_array * 255).astype(np.uint8)
                elif numpy_array.dtype != np.uint8:
                    numpy_array = numpy_array.astype(np.uint8)
                pil_image = Image.fromarray(numpy_array, mode='RGB')
                
            elif numpy_array.shape[2] == 4:
                # RGBA image
                if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
                    numpy_array = (numpy_array * 255).astype(np.uint8)
                elif numpy_array.dtype != np.uint8:
                    numpy_array = numpy_array.astype(np.uint8)
                pil_image = Image.fromarray(numpy_array, mode='RGBA')
                
            else:
                raise ValueError(f"Unsupported number of channels: {numpy_array.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {numpy_array.shape}")
        
        logging.debug(f"Converted to PIL Image: {pil_image.size}, mode: {pil_image.mode}")
        return pil_image
        
    except Exception as e:
        logging.error(f"Failed to convert numpy to PIL: {e}")
        raise RuntimeError(f"Numpy to PIL conversion failed: {e}")

def normalize_depth_map(depth_array: np.ndarray, 
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normalize depth map to specified range while preserving relative depth relationships.
    
    Args:
        depth_array: Input depth array
        target_range: Target range tuple (min, max) for normalization
        
    Returns:
        Normalized depth array as float32
    """
    logging.debug(f"Normalizing depth map: shape {depth_array.shape}, dtype {depth_array.dtype}")
    
    try:
        # Convert to float32 for processing
        depth_float = depth_array.astype(np.float32)
        
        # Get valid (non-zero, non-inf, non-nan) values for normalization
        valid_mask = np.isfinite(depth_float) & (depth_float > 0)
        
        if not np.any(valid_mask):
            logging.warning("No valid depth values found, returning zeros")
            return np.zeros_like(depth_float)
        
        valid_depths = depth_float[valid_mask]
        depth_min = np.min(valid_depths)
        depth_max = np.max(valid_depths)
        
        if depth_max <= depth_min:
            logging.warning(f"Invalid depth range: min={depth_min}, max={depth_max}, returning zeros")
            return np.zeros_like(depth_float)
        
        # Normalize to [0, 1]
        depth_normalized = np.zeros_like(depth_float)
        depth_normalized[valid_mask] = (valid_depths - depth_min) / (depth_max - depth_min)
        
        # Scale to target range
        target_min, target_max = target_range
        if target_min != 0.0 or target_max != 1.0:
            depth_normalized = depth_normalized * (target_max - target_min) + target_min
        
        logging.debug(f"Depth normalization complete: range [{np.min(depth_normalized[valid_mask]):.4f}, {np.max(depth_normalized[valid_mask]):.4f}]")
        return depth_normalized
        
    except Exception as e:
        logging.error(f"Failed to normalize depth map: {e}")
        raise RuntimeError(f"Depth normalization failed: {e}")

def ensure_16bit_output(depth_array: np.ndarray) -> np.ndarray:
    """
    Ensure depth array is in 16-bit format for output.
    
    Args:
        depth_array: Input depth array (typically float32 in [0,1] range)
        
    Returns:
        16-bit depth array (uint16)
    """
    logging.debug(f"Converting depth array to 16-bit: shape {depth_array.shape}, dtype {depth_array.dtype}")
    
    try:
        # Ensure input is float and normalized
        if depth_array.dtype != np.float32:
            depth_array = depth_array.astype(np.float32)
        
        # Clamp values to [0, 1] range
        depth_clamped = np.clip(depth_array, 0.0, 1.0)
        
        # Convert to 16-bit
        depth_16bit = (depth_clamped * IMAGE_CONFIG['depth_scale_factor']).astype(np.uint16)
        
        logging.debug(f"16-bit conversion complete: range [{depth_16bit.min()}, {depth_16bit.max()}]")
        return depth_16bit
        
    except Exception as e:
        logging.error(f"Failed to convert to 16-bit: {e}")
        raise RuntimeError(f"16-bit conversion failed: {e}")

def apply_gamma_correction(image: Union[Image.Image, np.ndarray], 
                         gamma: float = IMAGE_CONFIG['gamma_correction_default']) -> Union[Image.Image, np.ndarray]:
    """
    Apply gamma correction to image (optional enhancement, not used in main pipeline).
    
    Args:
        image: Input image (PIL or numpy array)
        gamma: Gamma correction factor (1.0 = no correction)
        
    Returns:
        Gamma corrected image of same type as input
    """
    if gamma == 1.0:
        return image  # No correction needed
    
    logging.debug(f"Applying gamma correction: gamma={gamma}")
    
    try:
        if isinstance(image, Image.Image):
            # Convert to numpy for processing
            np_array = convert_pil_to_numpy(image)
            
            # Apply gamma correction
            gamma_corrected = apply_gamma_correction(np_array, gamma)
            
            # Convert back to PIL
            return convert_numpy_to_pil(gamma_corrected)
        
        elif isinstance(image, np.ndarray):
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                normalized = image.astype(np.float32) / 255.0
            else:
                normalized = image.astype(np.float32)
            
            # Apply gamma correction
            gamma_corrected = np.power(normalized, gamma)
            
            # Convert back to original dtype
            if image.dtype == np.uint8:
                return (gamma_corrected * 255).astype(np.uint8)
            else:
                return gamma_corrected.astype(image.dtype)
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
    except Exception as e:
        logging.error(f"Gamma correction failed: {e}")
        return image  # Return original on failure

def get_image_statistics(image: Union[Image.Image, np.ndarray]) -> dict:
    """
    Get comprehensive statistics about an image.
    
    Args:
        image: Input image (PIL or numpy array)
        
    Returns:
        Dictionary containing image statistics
    """
    logging.debug("Computing image statistics")
    
    try:
        if isinstance(image, Image.Image):
            np_array = convert_pil_to_numpy(image)
            size = image.size
            mode = image.mode
        else:
            np_array = image
            size = (image.shape[1], image.shape[0]) if len(image.shape) >= 2 else (image.shape[0], 1)
            mode = f"Array_{image.dtype}"
        
        stats = {
            'size': size,
            'shape': np_array.shape,
            'dtype': str(np_array.dtype),
            'mode': mode,
            'channels': np_array.shape[2] if len(np_array.shape) == 3 else 1,
            'min_value': float(np.min(np_array)),
            'max_value': float(np.max(np_array)),
            'mean_value': float(np.mean(np_array)),
            'std_value': float(np.std(np_array)),
            'memory_usage_mb': np_array.nbytes / (1024 * 1024)
        }
        
        # Add panoramic format validation
        if len(np_array.shape) >= 2:
            aspect_ratio = size[0] / size[1]
            stats['aspect_ratio'] = aspect_ratio
            stats['is_panoramic'] = validate_panoramic_format(Image.fromarray(np_array) if isinstance(image, np.ndarray) else image)
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to compute image statistics: {e}")
        return {'error': str(e)}

def save_image_with_metadata(image: Union[Image.Image, np.ndarray], 
                           output_path: str,
                           metadata: Optional[dict] = None) -> str:
    """
    Save image with optional metadata preservation.
    
    Args:
        image: Image to save (PIL or numpy array)
        output_path: Output file path
        metadata: Optional metadata dictionary to embed
        
    Returns:
        Path to saved image file
    """
    logging.info(f"Saving image with metadata to: {output_path}")
    
    try:
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = convert_numpy_to_pil(image)
        else:
            pil_image = image.copy()
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata if provided
        if metadata:
            # Convert metadata to PngInfo for PNG files
            if output_path.suffix.lower() == '.png':
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                for key, value in metadata.items():
                    pnginfo.add_text(str(key), str(value))
                pil_image.save(str(output_path), pnginfo=pnginfo)
            else:
                # For other formats, save without metadata
                pil_image.save(str(output_path))
                logging.warning(f"Metadata not supported for {output_path.suffix} format")
        else:
            pil_image.save(str(output_path))
        
        logging.info(f"Image saved successfully: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logging.error(f"Failed to save image: {e}")
        raise IOError(f"Image save failed: {e}")

def validate_depth_map(depth_array: np.ndarray) -> bool:
    """
    Validate depth map array for common issues.
    
    Args:
        depth_array: Depth map array to validate
        
    Returns:
        True if depth map is valid, False otherwise
    """
    logging.debug("Validating depth map")
    
    try:
        # Check basic properties
        if not isinstance(depth_array, np.ndarray):
            logging.error("Depth map is not a numpy array")
            return False
        
        if len(depth_array.shape) != 2:
            logging.error(f"Depth map has wrong shape: {depth_array.shape}, expected 2D")
            return False
        
        if depth_array.size == 0:
            logging.error("Depth map is empty")
            return False
        
        # Check for invalid values
        finite_mask = np.isfinite(depth_array)
        finite_ratio = np.sum(finite_mask) / depth_array.size
        
        if finite_ratio < 0.5:
            logging.warning(f"Depth map has high ratio of invalid values: {1-finite_ratio:.2%}")
            return False
        
        # Check value range
        valid_values = depth_array[finite_mask]
        if len(valid_values) > 0:
            depth_min, depth_max = np.min(valid_values), np.max(valid_values)
            if depth_min == depth_max:
                logging.warning("Depth map has constant values")
                return False
            
            # Check for reasonable depth range (assuming normalized depths)
            if depth_min < 0 or depth_max > 1000:  # Allow for non-normalized depths
                logging.warning(f"Depth map has unusual value range: [{depth_min}, {depth_max}]")
        
        logging.debug("Depth map validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Depth map validation failed: {e}")
        return False

# Utility function for cleaning up temporary files
def cleanup_temp_images(temp_dir: str):
    """
    Clean up temporary image files in specified directory.
    
    Args:
        temp_dir: Directory containing temporary files
    """
    if not os.path.exists(temp_dir):
        return
    
    try:
        import shutil
        for file_path in Path(temp_dir).glob("*"):
            if file_path.suffix.lower() in IMAGE_CONFIG['supported_formats']:
                file_path.unlink()
                logging.debug(f"Removed temporary image: {file_path}")
        
        logging.info(f"Cleaned up temporary images in: {temp_dir}")
        
    except Exception as e:
        logging.warning(f"Failed to cleanup temporary images: {e}")

# Setup module logging
logging.getLogger(__name__).setLevel(logging.INFO)