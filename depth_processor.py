"""
Main depth processing pipeline for 360° panoramic depth estimation.
Coordinates the entire pipeline from equirectangular input to final depth map output.
Uses 360MonoDepth icosahedral projection technique with Marigold depth estimation.
"""

import numpy as np
import logging
from PIL import Image
from typing import List, Tuple, Optional, Union
import os
import tempfile
import gc
import torch
from pathlib import Path

# Import project modules
from icosahedron_projections import generate_icosahedron_projections, stitch_projections_to_equirect
from marigold_wrapper import initialize_marigold_model, batch_process_projections, clear_model_cache
from image_utils import (
    validate_panoramic_format, 
    convert_pil_to_numpy, 
    convert_numpy_to_pil,
    normalize_depth_map,
    ensure_16bit_output
)

# Embedded constants for depth processing pipeline
PROCESSING_CONFIG = {
    'output_bit_depth': 16,
    'max_image_size': 8192,  # Maximum dimension for safety
    'min_image_size': 512,   # Minimum dimension for processing
    'temp_dir': './temp_processing',
    'supported_formats': ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
    'depth_output_format': 'PNG',
    'memory_cleanup_interval': 1,  # Clean memory every N projections
    'enable_progress_logging': True,
    'cpu_optimization': True
}

class DepthProcessor:
    """
    Main depth processing pipeline coordinator.
    Handles the complete workflow from input validation to final depth map generation.
    """
    
    def __init__(self):
        """Initialize depth processor with CPU optimization."""
        logging.info("Initializing depth processor for CPU-based processing")
        
        # Initialize Marigold model
        self.marigold_pipeline = None
        self.temp_dir = None
        self._setup_temp_directory()
        
        # Performance tracking
        self.processing_stats = {
            'total_projections': 0,
            'successful_projections': 0,
            'failed_projections': 0,
            'total_processing_time': 0.0
        }
        
    def _setup_temp_directory(self):
        """Setup temporary directory for intermediate processing files."""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="360depth_")
            logging.info(f"Created temporary directory: {self.temp_dir}")
        except Exception as e:
            logging.error(f"Failed to create temporary directory: {e}")
            self.temp_dir = "./temp"
            os.makedirs(self.temp_dir, exist_ok=True)
    
    def validate_input_image(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """
        Validate and preprocess input panoramic image.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            Validated PIL Image
            
        Raises:
            ValueError: If image is invalid or not panoramic format
        """
        logging.info("Validating input panoramic image")
        
        # Handle different input types
        if isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            
            # Check file extension
            file_ext = Path(image).suffix.lower()
            if file_ext not in PROCESSING_CONFIG['supported_formats']:
                raise ValueError(f"Unsupported image format: {file_ext}")
            
            try:
                pil_image = Image.open(image)
                logging.info(f"Loaded image from file: {image}")
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")
                
        elif isinstance(image, np.ndarray):
            try:
                pil_image = convert_numpy_to_pil(image)
                logging.info("Converted numpy array to PIL Image")
            except Exception as e:
                raise ValueError(f"Failed to convert numpy array to PIL: {e}")
                
        elif isinstance(image, Image.Image):
            pil_image = image
            logging.info("Using provided PIL Image")
            
        else:
            raise ValueError(f"Unsupported input image type: {type(image)}")
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            logging.info(f"Converting image from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        
        # Validate image dimensions
        width, height = pil_image.size
        logging.info(f"Image dimensions: {width}x{height}")
        
        # Check minimum size
        if width < PROCESSING_CONFIG['min_image_size'] or height < PROCESSING_CONFIG['min_image_size']:
            raise ValueError(f"Image too small: {width}x{height}. Minimum size: {PROCESSING_CONFIG['min_image_size']}")
        
        # Check maximum size for memory safety
        if width > PROCESSING_CONFIG['max_image_size'] or height > PROCESSING_CONFIG['max_image_size']:
            logging.warning(f"Large image detected: {width}x{height}. Processing may be slow on CPU.")
        
        # Validate panoramic format (should be roughly 2:1 aspect ratio)
        aspect_ratio = width / height
        if not validate_panoramic_format(pil_image):
            logging.warning(f"Image aspect ratio {aspect_ratio:.2f} may not be standard equirectangular (2:1)")
        
        logging.info(f"Input image validation successful: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
        return pil_image
    
    def _initialize_marigold_if_needed(self):
        """Initialize Marigold model if not already loaded."""
        if self.marigold_pipeline is None:
            logging.info("Initializing Marigold depth estimation model")
            try:
                self.marigold_pipeline = initialize_marigold_model()
                logging.info("Marigold model initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Marigold model: {e}")
                raise RuntimeError(f"Model initialization failed: {e}")
    
    def _cleanup_memory(self):
        """Perform memory cleanup to prevent OOM issues."""
        logging.debug("Performing memory cleanup")
        
        # Clear model cache
        if self.marigold_pipeline:
            clear_model_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def convert_to_16bit_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """
        Convert depth array to 16-bit format for output.
        
        Args:
            depth_array: Input depth array (typically float32 in [0,1] range)
            
        Returns:
            16-bit depth array (uint16)
        """
        logging.info("Converting depth map to 16-bit format")
        
        try:
            # Ensure input is float32
            if depth_array.dtype != np.float32:
                depth_array = depth_array.astype(np.float32)
            
            # Normalize to [0, 1] range if needed
            depth_normalized = normalize_depth_map(depth_array)
            
            # Convert to 16-bit
            depth_16bit = ensure_16bit_output(depth_normalized)
            
            logging.info(f"Converted depth map to 16-bit: {depth_16bit.shape}, dtype: {depth_16bit.dtype}")
            logging.info(f"Depth value range: {depth_16bit.min()} - {depth_16bit.max()}")
            
            return depth_16bit
            
        except Exception as e:
            logging.error(f"Failed to convert depth to 16-bit: {e}")
            raise RuntimeError(f"16-bit conversion failed: {e}")
    
    def save_depth_image(self, depth_array: np.ndarray, output_path: Optional[str] = None) -> str:
        """
        Save depth map as 16-bit PNG image.
        
        Args:
            depth_array: 16-bit depth array
            output_path: Optional output path (if None, uses temp directory)
            
        Returns:
            Path to saved depth image
        """
        logging.info("Saving 16-bit depth image")
        
        try:
            if output_path is None:
                output_path = os.path.join(self.temp_dir, "depth_output.png")
            
            # Ensure 16-bit format
            if depth_array.dtype != np.uint16:
                depth_array = self.convert_to_16bit_depth(depth_array)
            
            # Convert to PIL Image and save
            depth_pil = Image.fromarray(depth_array, mode='I;16')
            depth_pil.save(output_path, format='PNG')
            
            logging.info(f"Depth image saved: {output_path}")
            logging.info(f"File size: {os.path.getsize(output_path)} bytes")
            
            return output_path
            
        except Exception as e:
            logging.error(f"Failed to save depth image: {e}")
            raise RuntimeError(f"Depth image save failed: {e}")
    
    def _log_processing_progress(self, current_step: int, total_steps: int, step_name: str):
        """Log processing progress for user feedback."""
        if PROCESSING_CONFIG['enable_progress_logging']:
            progress_pct = (current_step / total_steps) * 100
            logging.info(f"Progress: {progress_pct:.1f}% - {step_name} ({current_step}/{total_steps})")
    
    def process_depth(self, input_image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """
        Main depth processing pipeline.
        Coordinates the complete workflow from input to 16-bit depth map output.
        
        Args:
            input_image: Input panoramic image
            
        Returns:
            16-bit depth map as PIL Image
        """
        logging.info("=== Starting 360° panoramic depth estimation pipeline ===")
        
        try:
            # Step 1: Validate input image
            logging.info("Step 1/6: Validating input image")
            validated_image = self.validate_input_image(input_image)
            original_size = validated_image.size
            self._log_processing_progress(1, 6, "Input validation completed")
            
            # Step 2: Initialize Marigold model
            logging.info("Step 2/6: Initializing Marigold depth estimation model")
            self._initialize_marigold_if_needed()
            self._log_processing_progress(2, 6, "Model initialization completed")
            
            # Step 3: Generate icosahedral projections
            logging.info("Step 3/6: Generating 20 icosahedral projections")
            projections = generate_icosahedron_projections(validated_image)
            
            if len(projections) != 20:
                raise RuntimeError(f"Expected 20 projections, got {len(projections)}")
            
            logging.info(f"Generated {len(projections)} projections at native resolution")
            self._log_processing_progress(3, 6, f"Generated {len(projections)} projections")
            
            # Step 4: Process depth estimation for each projection
            logging.info("Step 4/6: Processing depth estimation for all projections")
            depth_projections = batch_process_projections(
                projections, 
                self.marigold_pipeline,
                progress_callback=lambda i, total: self._log_processing_progress(
                    i, total, f"Processed projection {i}/{total}"
                )
            )
            
            if len(depth_projections) != 20:
                raise RuntimeError(f"Expected 20 depth projections, got {len(depth_projections)}")
            
            logging.info(f"Successfully processed {len(depth_projections)} depth projections")
            self._log_processing_progress(4, 6, "Depth estimation completed")
            
            # Step 5: Stitch projections back to equirectangular format
            logging.info("Step 5/6: Stitching projections back to equirectangular format")
            stitched_depth = stitch_projections_to_equirect(
                depth_projections, 
                (original_size[1], original_size[0])  # (height, width)
            )
            
            logging.info(f"Stitched depth map shape: {stitched_depth.shape}")
            self._log_processing_progress(5, 6, "Projection stitching completed")
            
            # Step 6: Convert to 16-bit and create output image
            logging.info("Step 6/6: Converting to 16-bit depth image")
            depth_16bit = self.convert_to_16bit_depth(stitched_depth)
            
            # Convert to PIL Image for output
            output_image = Image.fromarray(depth_16bit, mode='I;16')
            
            # Update processing stats
            self.processing_stats['total_projections'] = 20
            self.processing_stats['successful_projections'] = len(depth_projections)
            
            logging.info(f"Final output image size: {output_image.size}")
            logging.info(f"Final output image mode: {output_image.mode}")
            self._log_processing_progress(6, 6, "16-bit conversion completed")
            
            # Cleanup memory
            self._cleanup_memory()
            
            logging.info("=== 360° panoramic depth estimation pipeline completed successfully ===")
            return output_image
            
        except Exception as e:
            logging.error(f"Depth processing pipeline failed: {e}")
            
            # Cleanup on error
            try:
                self._cleanup_memory()
            except:
                pass
            
            # Re-raise the exception for proper error handling
            raise RuntimeError(f"Depth processing failed: {e}")
    
    def cleanup_temp_files(self):
        """Clean up temporary processing files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logging.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp directory: {e}")
    
    def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_temp_files()
        except:
            pass

# Global processor instance for stateful processing
_depth_processor = None

def get_depth_processor() -> DepthProcessor:
    """Get singleton depth processor instance."""
    global _depth_processor
    if _depth_processor is None:
        _depth_processor = DepthProcessor()
    return _depth_processor

def process_depth(input_image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    """
    Main entry point for depth processing pipeline.
    This is the function called by app.py.
    
    Args:
        input_image: Input panoramic image
        
    Returns:
        16-bit depth map as PIL Image
    """
    processor = get_depth_processor()
    return processor.process_depth(input_image)

def validate_input_image(image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    """
    Validate input panoramic image.
    
    Args:
        image: Input image
        
    Returns:
        Validated PIL Image
    """
    processor = get_depth_processor()
    return processor.validate_input_image(image)

def convert_to_16bit_depth(depth_array: np.ndarray) -> np.ndarray:
    """
    Convert depth array to 16-bit format.
    
    Args:
        depth_array: Input depth array
        
    Returns:
        16-bit depth array
    """
    processor = get_depth_processor()
    return processor.convert_to_16bit_depth(depth_array)

def save_depth_image(depth_array: np.ndarray, output_path: Optional[str] = None) -> str:
    """
    Save depth image to file.
    
    Args:
        depth_array: 16-bit depth array
        output_path: Optional output path
        
    Returns:
        Path to saved file
    """
    processor = get_depth_processor()
    return processor.save_depth_image(depth_array, output_path)

def cleanup_temp_files():
    """Clean up temporary processing files."""
    global _depth_processor
    if _depth_processor is not None:
        _depth_processor.cleanup_temp_files()

def get_processing_stats() -> dict:
    """Get current processing statistics."""
    processor = get_depth_processor()
    return processor.get_processing_stats()

# Setup logging configuration
def setup_logging():
    """Setup logging configuration for the depth processor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

# Initialize logging when module is imported
setup_logging()