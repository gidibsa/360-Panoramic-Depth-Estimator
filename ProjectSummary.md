## Project Structure for Hugging Face Spaces

Here's the complete project structure with file names and functions:

### **Root Directory Structure**
```
360-panoramic-depth-estimator/
├── app.py                          # Main Gradio interface (your existing file)
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── config.py                     # Configuration constants
├── depth_processor.py            # Main depth processing pipeline
├── icosahedron_projections.py    # Icosahedral projection handling
├── image_utils.py                # Image processing utilities
└── marigold_wrapper.py           # Marigold model interface
```

### **File Functions Overview**

#### **1. app.py (Main Interface)**
- `process_image(input_img)` - Main processing function called by Gradio
- Gradio Blocks interface setup

#### **2. requirements.txt**
Dependencies list for HuggingFace Spaces

#### **3. `config.py`**
- `MARIGOLD_CONFIG` - Marigold model configuration constants
- `PROCESSING_CONFIG` - Processing parameters (ensemble_size=1, steps=1)
- `ICOSAHEDRON_CONFIG` - Icosahedron projection parameters

#### **4. `depth_processor.py` (Main Pipeline)**
- `process_depth(input_image)` - Main processing pipeline coordinator
- `validate_input_image(image)` - Input validation and preprocessing
- `convert_to_16bit_depth(depth_array)` - Convert depth to 16-bit format
- `save_depth_image(depth_array)` - Save final depth map

#### **5. `icosahedron_projections.py`**
- `generate_icosahedron_projections(equirectangular_image)` - Create 20 projections
- `project_equirect_to_face(image, face_index)` - Project single face
- `stitch_projections_to_equirect(depth_projections)` - Combine depth maps
- `get_icosahedron_vertices()` - Icosahedron geometry data
- `calculate_face_coordinates(face_index)` - Face coordinate mapping
- `interpolate_depth_boundaries(depth_maps)` - Handle projection overlaps

#### **6. `image_utils.py`**
- `load_and_validate_image(image_path)` - Image loading with validation
- `resize_maintain_aspect(image, target_size)` - Resolution handling
- `convert_pil_to_numpy(pil_image)` - PIL to numpy conversion
- `convert_numpy_to_pil(numpy_array)` - Numpy to PIL conversion
- `normalize_depth_map(depth_array)` - Depth normalization
- `apply_gamma_correction(image, gamma)` - Optional gamma correction

#### **7. `marigold_wrapper.py`**
- `initialize_marigold_model()` - Load and initialize Marigold
- `estimate_depth_single_image(image)` - Single image depth estimation
- `batch_process_projections(projection_list)` - Process projections serially
- `configure_cpu_processing()` - CPU optimization settings
- `clear_model_cache()` - Memory management

### **Key Function Relationships**

```
app.py:process_image()
    ↓
depth_processor.py:process_depth()
    ↓
icosahedron_projections.py:generate_icosahedron_projections()
    ↓ (20 projections)
marigold_wrapper.py:batch_process_projections()
    ↓ (20 depth maps)
icosahedron_projections.py:stitch_projections_to_equirect()
    ↓
depth_processor.py:convert_to_16bit_depth()
    ↓
Return to Gradio interface
```

### **Logging Points**
Each major function should include:
- Input validation logging
- Processing start/completion logging
- Error handling with detailed messages
- Progress updates for long operations
- Memory usage monitoring

This structure separates concerns clearly, making the code maintainable and suitable for HuggingFace Spaces deployment while ensuring all processing happens on CPU in a memory-efficient manner.