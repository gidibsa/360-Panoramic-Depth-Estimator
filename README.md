---
title: 360° Panoramic Depth Estimation
emoji: 🌍
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 360° Panoramic Depth Estimation

Generate high-quality depth maps from 360° panoramic images using state-of-the-art computer vision techniques.

## Features

- **360MonoDepth icosahedral projection** for optimal panoramic processing
- **Marigold depth estimation model** (latest v1.1) for accurate predictions
- **Native resolution processing** - no downscaling for maximum detail
- **16-bit depth output** for professional applications
- **CPU-optimized** for Hugging Face Spaces

## How to Use

1. Upload an equirectangular panoramic image (2:1 aspect ratio)
2. Click "Generate Depth Map" 
3. Wait for processing (typically 2-10 minutes depending on resolution)
4. Download the resulting 16-bit depth map

## Technical Details

- **Model**: Marigold Depth v1.1 (prs-eth/marigold-depth-v1-1)
- **Projection**: 20 icosahedral face projections
- **Processing**: Serial CPU processing optimized for Spaces
- **Output**: 16-bit grayscale PNG depth maps

## Supported Formats

- Input: JPEG, PNG, TIFF, BMP
- Output: 16-bit PNG

## Limitations

- Maximum file size: 50MB
- Processing timeout: 10 minutes
- CPU processing only (slower than GPU but more accessible)

## Citation

If you use this tool in your research, please cite the original papers:

```bibtex
@article{marigold2023,
  title={Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={...},
  journal={...},
  year={2023}
}

@article{360monodepth2023,
  title={360MonoDepth: High-Resolution 360° Monocular Depth Estimation},
  author={...},
  journal={...},
  year={2023}
}
```

## License

MIT License - see LICENSE file for details.