"""
360° Panoramic Depth Estimation using Marigold and 360MonoDepth Icosahedral Projection
Hugging Face Spaces application for generating high-quality depth maps from panoramic images.
"""

import gradio as gr
import logging
import time
import traceback
from PIL import Image
import numpy as np
from typing import Optional, Tuple
import gc
import os

# Import project modules
from depth_processor import process_depth, get_processing_stats, cleanup_temp_files
from image_utils import get_image_statistics, validate_panoramic_format
from marigold_wrapper import get_marigold_config, get_processing_stats as get_marigold_stats

# Configure logging for Hugging Face Spaces
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Application constants
APP_CONFIG = {
    'title': "🌍 360° Panoramic Depth Estimation",
    'description': """
    Generate high-quality depth maps from 360° panoramic images using:
    - **360MonoDepth icosahedral projection technique** for optimal panoramic processing
    - **Marigold depth estimation model** (latest v1.1) for accurate depth prediction
    - **Native resolution processing** - no downscaling for maximum detail preservation
    - **16-bit depth output** for professional applications
    
    Upload an equirectangular panoramic image and get a detailed depth map!
    """,
    'max_image_size_mb': 50,  # Maximum file size in MB
    'supported_formats': ['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
    'processing_timeout': 600,  # 10 minutes timeout
    'enable_examples': True
}

def validate_input_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate input image for processing.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if image is None:
            return False, "No image provided"
        
        # Check image size
        width, height = image.size
        if width < 512 or height < 256:
            return False, f"Image too small: {width}x{height}. Minimum size: 512x256"
        
        # Check file size (approximate)
        img_array = np.array(image)
        size_mb = img_array.nbytes / (1024 * 1024)
        if size_mb > APP_CONFIG['max_image_size_mb']:
            return False, f"Image too large: {size_mb:.1f}MB. Maximum: {APP_CONFIG['max_image_size_mb']}MB"
        
        # Check panoramic format
        aspect_ratio = width / height
        if not validate_panoramic_format(image):
            return False, f"Image aspect ratio {aspect_ratio:.2f} doesn't appear to be panoramic (expected ~2:1)"
        
        return True, f"Valid panoramic image: {width}x{height} ({aspect_ratio:.2f} aspect ratio)"
        
    except Exception as e:
        return False, f"Image validation failed: {str(e)}"

def create_processing_info(stats: dict) -> str:
    """Create formatted processing information string."""
    try:
        info_lines = [
            "### Processing Statistics",
            f"- **Total projections processed:** {stats.get('total_projections', 0)}/20",
            f"- **Successful projections:** {stats.get('successful_projections', 0)}",
            f"- **Processing time:** {stats.get('total_processing_time', 0):.1f}s",
        ]
        
        # Add Marigold stats if available
        marigold_stats = get_marigold_stats()
        if marigold_stats.get('total_processed', 0) > 0:
            info_lines.extend([
                "",
                "### Marigold Model Statistics",
                f"- **Images processed:** {marigold_stats.get('total_processed', 0)}",
                f"- **Average time per image:** {marigold_stats.get('average_time_per_image', 0):.2f}s",
                f"- **Success rate:** {marigold_stats.get('successful_predictions', 0)}/{marigold_stats.get('total_processed', 0)}",
                f"- **Peak memory usage:** {marigold_stats.get('memory_peak_mb', 0):.1f}MB"
            ])
        
        return "\n".join(info_lines)
        
    except Exception:
        return "Processing statistics unavailable"

def process_image_with_progress(input_img: Image.Image, progress=gr.Progress()) -> Tuple[Optional[Image.Image], str]:
    """
    Process panoramic image with progress updates.
    
    Args:
        input_img: Input panoramic image
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (depth_image, status_message)
    """
    if input_img is None:
        return None, "❌ No image provided"
    
    # Update progress
    progress(0.1, desc="Validating input image...")
    
    # Validate input image
    is_valid, validation_message = validate_input_image(input_img)
    if not is_valid:
        return None, f"❌ {validation_message}"
    
    logging.info(f"Processing image: {input_img.size}, mode: {input_img.mode}")
    
    try:
        # Update progress
        progress(0.2, desc="Initializing depth estimation pipeline...")
        
        start_time = time.time()
        
        # Process depth estimation
        progress(0.3, desc="Generating icosahedral projections...")
        
        # The actual processing with internal progress tracking
        depth_image = process_depth(input_img)
        
        # Update progress at key stages (these will be overridden by internal logging)
        progress(0.8, desc="Converting to 16-bit depth format...")
        
        processing_time = time.time() - start_time
        
        progress(1.0, desc="Processing completed!")
        
        # Get processing statistics
        stats = get_processing_stats()
        stats['total_processing_time'] = processing_time
        
        # Create status message
        status_lines = [
            "✅ **Processing completed successfully!**",
            f"⏱️ **Total time:** {processing_time:.1f} seconds",
            f"📐 **Output size:** {depth_image.size}",
            f"🎯 **Mode:** {depth_image.mode} (16-bit depth)",
            "",
            create_processing_info(stats)
        ]
        
        status_message = "\n".join(status_lines)
        
        # Cleanup temporary files
        cleanup_temp_files()
        
        # Force garbage collection
        gc.collect()
        
        logging.info(f"Successfully processed image in {processing_time:.1f}s")
        
        return depth_image, status_message
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Processing failed: {e}")
        logging.error(f"Full traceback: {error_trace}")
        
        # Cleanup on error
        try:
            cleanup_temp_files()
            gc.collect()
        except:
            pass
        
        error_message = f"""
        ❌ **Processing failed!**
        
        **Error:** {str(e)}
        
        **Possible solutions:**
        - Ensure your image is in equirectangular format (2:1 aspect ratio)
        - Try with a smaller image size
        - Check that the image is a valid panoramic photograph
        
        If the problem persists, please report this issue.
        """
        
        return None, error_message

def create_examples() -> list:
    """Create example images for the interface."""
    # Note: In a real deployment, you would have actual example images
    # For now, return empty list as examples need to be actual files
    return []

def create_interface():
    """Create the main Gradio interface."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .status-box {
        border-left: 4px solid #ff7c00;
        padding: 10px;
        background-color: #f9f9f9;
        margin: 10px 0;
    }
    .progress-text {
        font-weight: bold;
        color: #ff7c00;
    }
    """
    
    with gr.Blocks(
        title=APP_CONFIG['title'],
        css=custom_css,
        theme=gr.themes.Default(primary_hue="orange")
    ) as interface:
        
        # Header
        gr.Markdown(f"""
        # {APP_CONFIG['title']}
        {APP_CONFIG['description']}
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 📤 Input")
                
                input_image = gr.Image(
                    type="pil",
                    label="Upload Panoramic Image",
                    height=300,
                    sources=["upload", "clipboard"],
                    formats=APP_CONFIG['supported_formats']
                )
                
                # Processing button
                process_btn = gr.Button(
                    "🚀 Generate Depth Map",
                    variant="primary",
                    size="lg"
                )
                
                # Image info display
                gr.Markdown("### 📊 Image Information")
                image_info = gr.Markdown("Upload an image to see details...")
                
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## 📥 Output")
                
                output_image = gr.Image(
                    type="pil",
                    label="16-bit Depth Map",
                    height=300,
                    show_download_button=True
                )
                
                # Status and statistics
                gr.Markdown("### 📈 Processing Status")
                status_display = gr.Markdown("Ready to process...")
        
        # Examples section (if enabled)
        if APP_CONFIG['enable_examples']:
            examples = create_examples()
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    outputs=[output_image, status_display],
                    fn=process_image_with_progress,
                    label="📋 Example Images"
                )
        
        # Information and tips section
        with gr.Accordion("💡 Tips and Information", open=False):
            gr.Markdown("""
            ### 🎯 How to get the best results:
            
            1. **Use equirectangular panoramic images** with 2:1 aspect ratio
            2. **Ensure good image quality** - avoid heavily compressed or blurry images
            3. **Higher resolution images** will produce more detailed depth maps
            4. **Processing time** depends on image resolution (typically 2-10 minutes)
            
            ### 🔧 Technical Details:
            
            - **Model:** Marigold Depth v1.1 (prs-eth/marigold-depth-v1-1)
            - **Projection:** 360MonoDepth icosahedral technique (20 projections)
            - **Resolution:** Native resolution processing (no downscaling)
            - **Output:** 16-bit grayscale depth PNG
            - **Processing:** CPU-optimized for Hugging Face Spaces
            
            ### 📝 Supported Formats:
            - JPEG (.jpg, .jpeg)
            - PNG (.png)
            - TIFF (.tiff, .tif)
            - BMP (.bmp)
            
            ### ⚠️ Limitations:
            - Maximum file size: {max_size}MB
            - Processing timeout: {timeout} minutes
            - CPU processing only (may be slower than GPU)
            """.format(
                max_size=APP_CONFIG['max_image_size_mb'],
                timeout=APP_CONFIG['processing_timeout'] // 60
            ))
        
        # Footer
        gr.Markdown("""
        ---
        
        **Built with:** [360MonoDepth](https://github.com/HAL-lucination/360MonoDepth) • 
        [Marigold](https://github.com/prs-eth/Marigold) • 
        [Gradio](https://gradio.app/) • 
        [Hugging Face Spaces](https://huggingface.co/spaces)
        
        *For research and educational purposes. Please cite the original papers if you use this tool in your work.*
        """)
        
        # Event handlers
        def update_image_info(image):
            """Update image information display."""
            if image is None:
                return "Upload an image to see details..."
            
            try:
                stats = get_image_statistics(image)
                
                info_lines = [
                    f"**Size:** {stats['size'][0]} × {stats['size'][1]} pixels",
                    f"**Aspect Ratio:** {stats.get('aspect_ratio', 'N/A'):.2f}",
                    f"**Mode:** {stats['mode']}",
                    f"**Channels:** {stats['channels']}",
                    f"**Memory:** {stats['memory_usage_mb']:.1f} MB",
                    f"**Panoramic:** {'✅ Yes' if stats.get('is_panoramic', False) else '⚠️ No (may still work)'}",
                ]
                
                return "\n".join(info_lines)
                
            except Exception as e:
                return f"Error analyzing image: {str(e)}"
        
        # Connect event handlers
        input_image.change(
            fn=update_image_info,
            inputs=input_image,
            outputs=image_info
        )
        
        process_btn.click(
            fn=process_image_with_progress,
            inputs=input_image,
            outputs=[output_image, status_display],
            show_progress=True
        )
    
    return interface

# Create and configure the interface
demo = create_interface()

# Configuration for Hugging Face Spaces
if __name__ == "__main__":
    # Log system information
    config_info = get_marigold_config()
    logging.info("=== 360° Panoramic Depth Estimation Started ===")
    logging.info(f"Marigold Model: {config_info['model_config']['model_name']}")
    logging.info(f"Processing Device: {config_info['model_config']['device']}")
    logging.info(f"Max Image Size: {APP_CONFIG['max_image_size_mb']}MB")
    logging.info("Ready to process panoramic images!")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Required for Hugging Face Spaces
        server_port=7860,        # Standard port for Spaces
        show_error=True,         # Show detailed errors
        show_tips=True,          # Show usage tips
        enable_queue=True,       # Enable request queue
        max_threads=1,          # Limit concurrent processing for CPU
        favicon_path=None,       # Can add custom favicon
        ssl_verify=False         # For development
    )
else:
    # When imported as module
    logging.info("360° Panoramic Depth Estimation module loaded")