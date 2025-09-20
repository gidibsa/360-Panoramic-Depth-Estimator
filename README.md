# 360MonoDepth with Depth Anything V2

This project is a Hugging Face Space demo that estimates depth from **panoramic (equirectangular) images** using a simplified 360MonoDepth pipeline and the **Depth Anything V2 Large** model.

## How it works
1. The panorama is split into multiple tangent-plane "slices" (like snapping photos around you).
2. Each slice is passed through **Depth Anything V2 Large** to predict depth.
3. The depth maps are normalized and stitched horizontally back into a depth panorama.

## Running locally
Clone the repo and install dependencies:
```bash
git clone https://github.com/yourusername/360MonoDepth-DepthAnythingV2.git
cd 360MonoDepth-DepthAnythingV2
pip install -r requirements.txt
python app.py
