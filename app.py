import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# -------------------------------
# Load Depth Anything V2 Large
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large")
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Large"
).to(device)

# -------------------------------
# Step 1: Split panorama into tangent-plane views
# -------------------------------
def sample_views(equi_img, n_views=12):
    """
    Cut the panoramic image into n_views horizontal perspective slices.
    This avoids distortion by giving the depth model 'normal-looking' views.
    """
    h, w = equi_img.shape[:2]
    views = []
    view_w = w // n_views
    for i in range(n_views):
        crop = equi_img[:, i * view_w : (i + 1) * view_w]
        crop = cv2.resize(crop, (512, 512))  # upscale slices for better quality
        views.append(crop)
    return views

# -------------------------------
# Step 2: Run Depth Anything V2 on each slice
# -------------------------------
def predict_depth(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.predicted_depth[0].cpu().numpy()

    # Normalize to 0–255 for display
    d_min, d_max = prediction.min(), prediction.max()
    depth_vis = (255 * (prediction - d_min) / (d_max - d_min)).astype("uint8")
    return depth_vis

# -------------------------------
# Step 3: Recombine slices into panorama depth
# -------------------------------
def pano_depth(equi_img):
    img = np.array(equi_img)[:, :, ::-1]  # PIL->BGR
    views = sample_views(img, n_views=12)

    depth_views = []
    for v in views:
        depth_map = predict_depth(Image.fromarray(v[..., ::-1]))  # BGR->RGB
        depth_views.append(depth_map)

    # Concatenate horizontally to rebuild panorama depth
    stitched = np.concatenate(depth_views, axis=1)
    return Image.fromarray(stitched)

# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 360MonoDepth with Depth Anything V2 Large")

    with gr.Row():
        input_img = gr.Image(
            type="pil", label="Upload Panoramic Image (Equirectangular)"
        )
        output_img = gr.Image(type="pil", label="Predicted Depth Map")

    run_button = gr.Button("Run Depth Estimation")

    run_button.click(fn=pano_depth, inputs=input_img, outputs=output_img)

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    demo.launch()
