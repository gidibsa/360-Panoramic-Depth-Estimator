"""
Marigold depth estimation model wrapper for 360° panoramic depth estimation.
Handles initialization, configuration, and batch processing of icosahedral projections
using the latest Marigold depth estimation model optimized for CPU processing.
"""

import numpy as np
import torch
import logging
from PIL import Image
from typing import List, Optional, Callable, Union, Tuple
import gc
import time
from pathlib import Path
import warnings

# Import Marigold pipeline from diffusers
try:
    from diffusers import MarigoldDepthPipeline
    from diffusers.utils import logging as diffusers_logging
except ImportError as e:
    logging.error(f"Failed to import Marigold dependencies: {e}")
    raise ImportError("Please install diffusers: pip install diffusers")

# Import image utilities
from image_utils import convert_pil_to_numpy, convert_numpy_to_pil, validate_depth_map

# Embedded constants for Marigold depth estimation
MARIGOLD_CONFIG = {
    'model_name': 'prs-eth/marigold-depth-v1-1',
    'variant': 'fp16',  # Use FP16 for memory efficiency
    'torch_dtype': torch.float16,
    'device': 'cpu',  # Force CPU processing for Hugging Face Spaces
    'ensemble_size': 1,  # Minimum for speed (as per requirements)
    'denoising_steps': 1,  # Minimum for speed (as per requirements)
    'processing_resolution': 'native',  # Native resolution processing
    'output_type': 'np',  # Return numpy arrays
    'seed': 42,  # For reproducibility
    'enable_memory_efficient_attention': True,
    'enable_cpu_offload': False,  # Keep on CPU
    'low_cpu_mem_usage': True,
    'safety_checker': None,  # Disable safety checker for speed
    'requires_safety_checker': False
}

# Performance optimization constants
PERFORMANCE_CONFIG = {
    'batch_size': 1,  # Process one projection at a time on CPU
    'max_memory_usage_mb': 4000,  # 4GB memory limit
    'cleanup_frequency': 5,  # Clean memory every N projections
    'torch_compile': False,  # Disable torch compile for CPU
    'attention_slicing': True,  # Enable attention slicing for memory
    'progress_logging': True,
    'warmup_runs': 1  # Number of warmup runs for consistent timing
}

class MarigoldDepthProcessor:
    """
    Wrapper class for Marigold depth estimation model with CPU optimization.
    Handles model initialization, memory management, and batch processing.
    """
    
    def __init__(self):
        """Initialize Marigold depth processor with CPU optimization."""
        logging.info("Initializing Marigold depth processor for CPU-based processing")
        
        # Initialize pipeline
        self.pipeline = None
        self.device = MARIGOLD_CONFIG['device']
        self.model_loaded = False
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'average_time_per_image': 0.0,
            'memory_peak_mb': 0.0,
            'successful_predictions': 0,
            'failed_predictions': 0
        }
        
        # Setup CPU optimization
        self._configure_cpu_processing()
        
    def _configure_cpu_processing(self):
        """Configure PyTorch and system settings for optimal CPU processing."""
        logging.info("Configuring CPU optimization settings")
        
        try:
            # Set PyTorch to use all available CPU cores
            torch.set_num_threads(torch.get_num_threads())
            
            # Disable CUDA if available to force CPU usage
            if torch.cuda.is_available():
                logging.warning("CUDA detected but forcing CPU usage as per configuration")
                torch.cuda.set_device(-1)  # Disable CUDA
            
            # Set memory allocation strategy
            torch.backends.cudnn.enabled = False
            
            # Configure memory management
            if hasattr(torch.backends, 'opt_einsum'):
                torch.backends.opt_einsum.enabled = True
            
            # Suppress diffusers logging for cleaner output
            diffusers_logging.set_verbosity_error()
            
            logging.info(f"CPU optimization configured: {torch.get_num_threads()} threads")
            
        except Exception as e:
            logging.warning(f"Failed to configure some CPU optimizations: {e}")
    
    def _load_model(self) -> MarigoldDepthPipeline:
        """
        Load and initialize the Marigold depth estimation model.
        
        Returns:
            Initialized MarigoldDepthPipeline
            
        Raises:
            RuntimeError: If model loading fails
        """
        logging.info(f"Loading Marigold model: {MARIGOLD_CONFIG['model_name']}")
        
        try:
            # Suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Load the pipeline with CPU-optimized settings
                pipeline = MarigoldDepthPipeline.from_pretrained(
                    MARIGOLD_CONFIG['model_name'],
                    variant=MARIGOLD_CONFIG['variant'],
                    torch_dtype=MARIGOLD_CONFIG['torch_dtype'],
                    low_cpu_mem_usage=MARIGOLD_CONFIG['low_cpu_mem_usage'],
                    safety_checker=MARIGOLD_CONFIG['safety_checker'],
                    requires_safety_checker=MARIGOLD_CONFIG['requires_safety_checker']
                )
                
                # Move to CPU device
                pipeline = pipeline.to(self.device)
                
                # Enable memory efficient attention if supported
                if MARIGOLD_CONFIG['enable_memory_efficient_attention']:
                    try:
                        pipeline.enable_attention_slicing()
                        logging.info("Enabled attention slicing for memory efficiency")
                    except Exception as e:
                        logging.warning(f"Could not enable attention slicing: {e}")
                
                # Disable safety checker for speed
                if hasattr(pipeline, 'safety_checker'):
                    pipeline.safety_checker = None
                
                logging.info(f"Successfully loaded Marigold model on {self.device}")
                return pipeline
                
        except Exception as e:
            logging.error(f"Failed to load Marigold model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def initialize_model(self) -> bool:
        """
        Initialize the Marigold model if not already loaded.
        
        Returns:
            True if model is successfully initialized, False otherwise
        """
        if self.model_loaded and self.pipeline is not None:
            logging.debug("Marigold model already initialized")
            return True
        
        try:
            self.pipeline = self._load_model()
            self.model_loaded = True
            
            # Perform warmup run for consistent performance
            self._warmup_model()
            
            logging.info("Marigold model initialization completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            self.model_loaded = False
            return False
    
    def _warmup_model(self):
        """Perform warmup runs to ensure consistent performance."""
        if PERFORMANCE_CONFIG['warmup_runs'] <= 0:
            return
        
        logging.info(f"Performing {PERFORMANCE_CONFIG['warmup_runs']} warmup run(s)")
        
        try:
            # Create a small test image for warmup
            warmup_size = 256
            test_image = Image.new('RGB', (warmup_size, warmup_size), color='gray')
            
            for i in range(PERFORMANCE_CONFIG['warmup_runs']):
                logging.debug(f"Warmup run {i + 1}/{PERFORMANCE_CONFIG['warmup_runs']}")
                
                with torch.no_grad():
                    _ = self.pipeline(
                        test_image,
                        ensemble_size=MARIGOLD_CONFIG['ensemble_size'],
                        denoising_steps=MARIGOLD_CONFIG['denoising_steps'],
                        processing_res=warmup_size,
                        match_input_res=True,
                        batch_size=PERFORMANCE_CONFIG['batch_size'],
                        seed=MARIGOLD_CONFIG['seed'],
                        output_type=MARIGOLD_CONFIG['output_type'],
                        show_progress_bar=False
                    )
                
                # Clear cache after warmup
                self._clear_memory_cache()
            
            logging.info("Model warmup completed")
            
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}")
    
    def _clear_memory_cache(self):
        """Clear memory caches to prevent OOM issues."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear pipeline cache if available
            if hasattr(self.pipeline, 'maybe_free_model_hooks'):
                self.pipeline.maybe_free_model_hooks()
                
        except Exception as e:
            logging.debug(f"Memory cache clearing encountered minor issues: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0
    
    def estimate_depth_single_image(self, image: Image.Image) -> np.ndarray:
        """
        Estimate depth for a single image using Marigold.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Depth map as numpy array (float32, values in [0,1] range)
            
        Raises:
            RuntimeError: If depth estimation fails
        """
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Marigold model not initialized. Call initialize_model() first.")
        
        logging.debug(f"Processing single image: {image.size}")
        
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image dimensions for native resolution processing
            width, height = image.size
            processing_res = max(width, height) if MARIGOLD_CONFIG['processing_resolution'] == 'native' else min(width, height)
            
            # Run depth estimation
            with torch.no_grad():
                depth_result = self.pipeline(
                    image,
                    ensemble_size=MARIGOLD_CONFIG['ensemble_size'],
                    denoising_steps=MARIGOLD_CONFIG['denoising_steps'],
                    processing_res=processing_res,
                    match_input_res=True,  # Maintain input resolution
                    batch_size=PERFORMANCE_CONFIG['batch_size'],
                    seed=MARIGOLD_CONFIG['seed'],
                    output_type=MARIGOLD_CONFIG['output_type'],
                    show_progress_bar=False
                )
            
            # Extract depth prediction
            if hasattr(depth_result, 'prediction'):
                depth_array = depth_result.prediction[0]  # First (and only) prediction
            else:
                depth_array = depth_result[0]  # Direct array access
            
            # Ensure depth array is float32 and 2D
            if depth_array.ndim == 3:
                depth_array = depth_array.squeeze()
            depth_array = depth_array.astype(np.float32)
            
            # Validate depth map
            if not validate_depth_map(depth_array):
                logging.warning("Generated depth map failed validation, but proceeding")
            
            # Update processing stats
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            self.processing_stats['average_time_per_image'] = (
                self.processing_stats['total_processing_time'] / 
                self.processing_stats['total_processed']
            )
            self.processing_stats['memory_peak_mb'] = max(
                self.processing_stats['memory_peak_mb'], 
                memory_after
            )
            self.processing_stats['successful_predictions'] += 1
            
            logging.debug(f"Depth estimation completed in {processing_time:.2f}s, "
                         f"output shape: {depth_array.shape}, "
                         f"memory usage: {memory_after - memory_before:.1f}MB increase")
            
            return depth_array
            
        except Exception as e:
            self.processing_stats['failed_predictions'] += 1
            logging.error(f"Depth estimation failed for image {image.size}: {e}")
            raise RuntimeError(f"Depth estimation failed: {e}")
    
    def batch_process_projections(self, 
                                projections: List[Image.Image],
                                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[np.ndarray]:
        """
        Process multiple projections in batch (serially for CPU efficiency).
        
        Args:
            projections: List of PIL Images to process
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of depth maps as numpy arrays
            
        Raises:
            RuntimeError: If batch processing fails
        """
        if not self.model_loaded or self.pipeline is None:
            raise RuntimeError("Marigold model not initialized. Call initialize_model() first.")
        
        logging.info(f"Starting batch processing of {len(projections)} projections")
        
        depth_maps = []
        failed_count = 0
        
        for i, projection in enumerate(projections):
            try:
                logging.info(f"Processing projection {i + 1}/{len(projections)}: {projection.size}")
                
                # Process single projection
                depth_map = self.estimate_depth_single_image(projection)
                depth_maps.append(depth_map)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, len(projections))
                
                # Periodic memory cleanup
                if (i + 1) % PERFORMANCE_CONFIG['cleanup_frequency'] == 0:
                    logging.debug(f"Performing memory cleanup after projection {i + 1}")
                    self._clear_memory_cache()
                
            except Exception as e:
                failed_count += 1
                logging.error(f"Failed to process projection {i + 1}: {e}")
                
                # Create a zero depth map as fallback
                if projections and len(projections) > 0:
                    # Use size from first successful projection or current projection
                    ref_size = projection.size if projection else projections[0].size
                    fallback_depth = np.zeros((ref_size[1], ref_size[0]), dtype=np.float32)
                    depth_maps.append(fallback_depth)
                    logging.warning(f"Using zero depth map as fallback for projection {i + 1}")
                
                # Stop processing if too many failures
                if failed_count > len(projections) // 4:  # More than 25% failures
                    raise RuntimeError(f"Too many projection processing failures: {failed_count}")
        
        # Final memory cleanup
        self._clear_memory_cache()
        
        logging.info(f"Batch processing completed: {len(depth_maps)} depth maps generated, {failed_count} failures")
        
        if len(depth_maps) != len(projections):
            raise RuntimeError(f"Mismatch in projection count: expected {len(projections)}, got {len(depth_maps)}")
        
        return depth_maps
    
    def get_processing_stats(self) -> dict:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def clear_model_cache(self):
        """Clear model cache and free memory."""
        logging.info("Clearing Marigold model cache")
        self._clear_memory_cache()
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.clear_model_cache()
        except:
            pass

# Global processor instance for stateful processing
_marigold_processor = None

def get_marigold_processor() -> MarigoldDepthProcessor:
    """Get singleton Marigold processor instance."""
    global _marigold_processor
    if _marigold_processor is None:
        _marigold_processor = MarigoldDepthProcessor()
    return _marigold_processor

def initialize_marigold_model() -> MarigoldDepthPipeline:
    """
    Initialize Marigold depth estimation model.
    
    Returns:
        Initialized MarigoldDepthPipeline
        
    Raises:
        RuntimeError: If initialization fails
    """
    processor = get_marigold_processor()
    
    if processor.initialize_model():
        return processor.pipeline
    else:
        raise RuntimeError("Failed to initialize Marigold model")

def estimate_depth_single_image(image: Image.Image) -> np.ndarray:
    """
    Estimate depth for a single image.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Depth map as numpy array
    """
    processor = get_marigold_processor()
    return processor.estimate_depth_single_image(image)

def batch_process_projections(projections: List[Image.Image], 
                            pipeline: Optional[MarigoldDepthPipeline] = None,
                            progress_callback: Optional[Callable[[int, int], None]] = None) -> List[np.ndarray]:
    """
    Process multiple projections in batch.
    
    Args:
        projections: List of PIL Images to process
        pipeline: Optional pre-initialized pipeline (ignored, uses global processor)
        progress_callback: Optional progress callback function
        
    Returns:
        List of depth maps as numpy arrays
    """
    processor = get_marigold_processor()
    
    # Ensure model is initialized
    if not processor.model_loaded:
        processor.initialize_model()
    
    return processor.batch_process_projections(projections, progress_callback)

def configure_cpu_processing():
    """Configure CPU processing optimization settings."""
    processor = get_marigold_processor()
    processor._configure_cpu_processing()

def clear_model_cache():
    """Clear model cache to free memory."""
    global _marigold_processor
    if _marigold_processor is not None:
        _marigold_processor.clear_model_cache()

def get_processing_stats() -> dict:
    """Get current processing statistics."""
    processor = get_marigold_processor()
    return processor.get_processing_stats()

def get_marigold_config() -> dict:
    """Get current Marigold configuration."""
    return {
        'model_config': MARIGOLD_CONFIG.copy(),
        'performance_config': PERFORMANCE_CONFIG.copy()
    }

# Setup logging when module is imported
logging.getLogger(__name__).setLevel(logging.INFO)

# Log configuration on import
logging.info(f"Marigold wrapper initialized with model: {MARIGOLD_CONFIG['model_name']}")
logging.info(f"CPU processing enabled with {torch.get_num_threads()} threads")