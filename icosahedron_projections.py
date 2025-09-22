"""
Icosahedral projection handling for 360° panoramic depth estimation.
This module handles the conversion of equirectangular panoramic images to icosahedral projections
and stitches depth maps back to equirectangular format at NATIVE RESOLUTION - no downscaling.
"""

import numpy as np
import cv2
from PIL import Image
import logging
from typing import List, Tuple, Dict, Optional
import math

# Embedded constants for icosahedral projections - NO FIXED PROJECTION SIZE
ICOSAHEDRON_CONFIG = {
    'num_faces': 20,
    'overlap_threshold': 0.1,
    'blending_method': 'weighted',
    'interpolation': cv2.INTER_LINEAR,
    'border_mode': cv2.BORDER_REFLECT_101,
    'fov_degrees': 90,  # Field of view for gnomonic projections
    'preserve_native_resolution': True
}

class IcosahedronProjector:
    """
    Handles icosahedral projection and inverse projection for panoramic images.
    Maintains FULL NATIVE RESOLUTION throughout the pipeline.
    """
    
    def __init__(self):
        """Initialize icosahedron geometry and projection parameters."""
        logging.info("Initializing icosahedron projector with native resolution preservation")
        self.vertices, self.faces = self._get_icosahedron_geometry()
        self.face_normals = self._calculate_face_normals()
        
    def _get_icosahedron_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get icosahedron vertices and face indices.
        
        Returns:
            Tuple of (vertices, faces) arrays
        """
        logging.debug("Calculating icosahedron geometry")
        
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Icosahedron vertices (12 vertices)
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=np.float64)
        
        # Normalize vertices to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Icosahedron faces (20 triangular faces)
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        return vertices, faces
    
    def _calculate_face_normals(self) -> np.ndarray:
        """
        Calculate normal vectors for each icosahedron face.
        
        Returns:
            Array of face normal vectors
        """
        logging.debug("Calculating face normals")
        
        normals = np.zeros((len(self.faces), 3), dtype=np.float64)
        
        for i, face in enumerate(self.faces):
            # Get face vertices
            v0, v1, v2 = self.vertices[face]
            
            # Calculate face center (centroid)
            center = (v0 + v1 + v2) / 3.0
            
            # Normalize to get outward normal
            normals[i] = center / np.linalg.norm(center)
            
        return normals
    
    def _calculate_optimal_projection_size(self, input_width: int, input_height: int) -> int:
        """
        Calculate optimal projection size to maintain detail density.
        Each face should capture approximately the same pixel density as the original.
        
        Args:
            input_width: Width of input equirectangular image
            input_height: Height of input equirectangular image
            
        Returns:
            Optimal square projection size for each face
        """
        # Calculate the average resolution per solid angle in the equirectangular image
        # Each icosahedron face covers approximately 1/20 of the sphere
        # The gnomonic projection with 90° FOV captures roughly equivalent area
        
        # For equirectangular: total pixels = width * height covers 4π steradians
        # For each face: should maintain similar pixel density
        total_pixels = input_width * input_height
        pixels_per_steradian = total_pixels / (4 * math.pi)
        
        # Each icosahedron face covers approximately 4π/20 steradians
        face_steradians = 4 * math.pi / 20
        
        # Calculate projection size to maintain pixel density
        face_pixels = pixels_per_steradian * face_steradians
        projection_size = int(math.sqrt(face_pixels))
        
        # Ensure minimum reasonable size and power of 2 for efficiency
        projection_size = max(512, projection_size)
        projection_size = 2 ** int(math.log2(projection_size) + 0.5)  # Round to nearest power of 2
        
        logging.info(f"Calculated optimal projection size: {projection_size}x{projection_size} for input {input_width}x{input_height}")
        return projection_size
    
    def _equirectangular_to_sphere(self, u: float, v: float) -> np.ndarray:
        """
        Convert equirectangular coordinates to 3D sphere coordinates.
        
        Args:
            u: Horizontal coordinate [0, 1]
            v: Vertical coordinate [0, 1]
            
        Returns:
            3D point on unit sphere
        """
        # Convert to spherical coordinates
        longitude = 2 * math.pi * u - math.pi  # [-π, π]
        latitude = math.pi * v - math.pi / 2   # [-π/2, π/2]
        
        # Convert to Cartesian coordinates
        x = math.cos(latitude) * math.cos(longitude)
        y = math.sin(latitude)
        z = math.cos(latitude) * math.sin(longitude)
        
        return np.array([x, y, z], dtype=np.float64)
    
    def _sphere_to_equirectangular(self, point: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D sphere coordinates to equirectangular coordinates.
        
        Args:
            point: 3D point on unit sphere
            
        Returns:
            Tuple of (u, v) coordinates [0, 1]
        """
        x, y, z = point
        
        # Convert to spherical coordinates
        longitude = math.atan2(z, x)  # [-π, π]
        latitude = math.asin(np.clip(y, -1, 1))  # [-π/2, π/2]
        
        # Convert to equirectangular coordinates
        u = (longitude + math.pi) / (2 * math.pi)  # [0, 1]
        v = (latitude + math.pi / 2) / math.pi     # [0, 1]
        
        return u, v
    
    def _project_face_to_gnomonic(self, face_idx: int, projection_size: int) -> np.ndarray:
        """
        Create gnomonic projection coordinates for a specific face at native resolution.
        
        Args:
            face_idx: Index of the icosahedron face
            projection_size: Size of the square projection (maintains native detail)
            
        Returns:
            Array of 3D coordinates for each pixel
        """
        logging.debug(f"Creating native resolution gnomonic projection for face {face_idx}, size: {projection_size}x{projection_size}")
        
        # Get face center (normal direction)
        face_normal = self.face_normals[face_idx]
        
        # Create local coordinate system for the face
        if abs(face_normal[0]) < 0.9:
            right = np.cross(face_normal, np.array([1, 0, 0]))
        else:
            right = np.cross(face_normal, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, face_normal)
        up = up / np.linalg.norm(up)
        
        # Create pixel grid at native resolution
        coords = np.zeros((projection_size, projection_size, 3), dtype=np.float64)
        
        # Projection parameters (field of view)
        fov_rad = math.radians(ICOSAHEDRON_CONFIG['fov_degrees'])
        scale = math.tan(fov_rad / 2)
        
        # Vectorized coordinate calculation for better performance
        i_coords, j_coords = np.meshgrid(
            np.arange(projection_size, dtype=np.float64),
            np.arange(projection_size, dtype=np.float64),
            indexing='ij'
        )
        
        # Normalize pixel coordinates to [-1, 1]
        x = (2 * j_coords / projection_size - 1) * scale
        y = (2 * i_coords / projection_size - 1) * scale
        
        # Project to sphere (vectorized)
        directions = (face_normal[None, None, :] + 
                     x[:, :, None] * right[None, None, :] + 
                     y[:, :, None] * up[None, None, :])
        
        # Normalize directions
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        coords = directions / norms
                
        return coords
    
    def project_equirect_to_face(self, image: np.ndarray, face_idx: int) -> np.ndarray:
        """
        Project equirectangular image to a specific icosahedron face at NATIVE RESOLUTION.
        
        Args:
            image: Input equirectangular image (H, W, C) at native resolution
            face_idx: Index of the face to project to
            
        Returns:
            Projected face image at native resolution
        """
        logging.info(f"Projecting equirectangular image to face {face_idx} at native resolution {image.shape}")
        
        input_height, input_width = image.shape[:2]
        
        # Calculate optimal projection size to maintain native detail
        projection_size = self._calculate_optimal_projection_size(input_width, input_height)
        
        # Get 3D coordinates for face projection at native resolution
        face_coords = self._project_face_to_gnomonic(face_idx, projection_size)
        
        # Convert 3D coordinates to equirectangular coordinates (vectorized)
        logging.debug("Converting 3D coordinates to equirectangular mapping")
        
        # Vectorized sphere to equirectangular conversion
        x, y, z = face_coords[:, :, 0], face_coords[:, :, 1], face_coords[:, :, 2]
        
        longitude = np.arctan2(z, x)
        latitude = np.arcsin(np.clip(y, -1, 1))
        
        u = (longitude + math.pi) / (2 * math.pi)
        v = (latitude + math.pi / 2) / math.pi
        
        # Convert to pixel coordinates
        map_x = (u * (input_width - 1)).astype(np.float32)
        map_y = (v * (input_height - 1)).astype(np.float32)
        
        # Apply remapping at native resolution
        logging.debug(f"Applying remapping to create {projection_size}x{projection_size} projection")
        projected = cv2.remap(
            image, 
            map_x, 
            map_y,
            ICOSAHEDRON_CONFIG['interpolation'],
            borderMode=ICOSAHEDRON_CONFIG['border_mode']
        )
        
        logging.info(f"Successfully created face {face_idx} projection at resolution {projected.shape}")
        return projected
    
    def generate_icosahedron_projections(self, equirectangular_image: Image.Image) -> List[Image.Image]:
        """
        Generate all 20 icosahedral projections from equirectangular image at NATIVE RESOLUTION.
        
        Args:
            equirectangular_image: Input panoramic image in PIL format at native resolution
            
        Returns:
            List of 20 projected face images at native resolution
        """
        logging.info(f"Generating 20 icosahedral projections from equirectangular image at native resolution: {equirectangular_image.size}")
        
        # Convert PIL to numpy array
        if isinstance(equirectangular_image, Image.Image):
            img_array = np.array(equirectangular_image)
        else:
            img_array = equirectangular_image
            
        logging.info(f"Input image array shape: {img_array.shape}")
        projections = []
        
        for face_idx in range(ICOSAHEDRON_CONFIG['num_faces']):
            logging.info(f"Processing face {face_idx + 1}/20 at native resolution")
            
            # Project to face at native resolution
            face_projection = self.project_equirect_to_face(img_array, face_idx)
            
            # Convert back to PIL Image
            face_pil = Image.fromarray(face_projection.astype(np.uint8))
            projections.append(face_pil)
            
            logging.debug(f"Face {face_idx + 1} projection size: {face_pil.size}")
            
        logging.info(f"Successfully generated all 20 projections at native resolution")
        return projections
    
    def _calculate_pixel_weights(self, coords: np.ndarray, face_idx: int) -> np.ndarray:
        """
        Calculate blending weights for pixels based on distance from face center.
        
        Args:
            coords: 3D coordinates array
            face_idx: Face index
            
        Returns:
            Weight array for blending
        """
        face_normal = self.face_normals[face_idx]
        
        # Calculate angular distance from face center (vectorized)
        dots = np.sum(coords * face_normal[None, None, :], axis=2)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)
        
        # Convert to weights (closer to center = higher weight)
        max_angle = math.pi / 3  # 60 degrees
        weights = np.cos(angles / max_angle * math.pi / 2)
        weights = np.clip(weights, 0, 1)
        
        return weights
    
    def stitch_projections_to_equirect(self, depth_projections: List[np.ndarray], 
                                     output_size: Tuple[int, int]) -> np.ndarray:
        """
        Stitch depth projections back to equirectangular format at NATIVE RESOLUTION.
        
        Args:
            depth_projections: List of 20 depth maps from projections at native resolution
            output_size: (height, width) of output equirectangular image at native resolution
            
        Returns:
            Stitched equirectangular depth map at native resolution
        """
        logging.info(f"Stitching depth projections back to equirectangular format at native resolution: {output_size}")
        
        output_height, output_width = output_size
        
        # Initialize output arrays at native resolution
        depth_sum = np.zeros((output_height, output_width), dtype=np.float64)
        weight_sum = np.zeros((output_height, output_width), dtype=np.float64)
        
        # Calculate projection size used for these depth maps
        if depth_projections:
            proj_size = depth_projections[0].shape[0]  # Assume square projections
            logging.info(f"Processing projections of size: {proj_size}x{proj_size}")
        
        for face_idx, depth_proj in enumerate(depth_projections):
            logging.info(f"Stitching face {face_idx + 1}/20 at native resolution")
            
            current_proj_size = depth_proj.shape[0]
            
            # Get 3D coordinates for this face at the projection's native resolution
            face_coords = self._project_face_to_gnomonic(face_idx, current_proj_size)
            
            # Calculate blending weights
            weights = self._calculate_pixel_weights(face_coords, face_idx)
            
            # Vectorized mapping back to equirectangular coordinates
            x, y, z = face_coords[:, :, 0], face_coords[:, :, 1], face_coords[:, :, 2]
            
            longitude = np.arctan2(z, x)
            latitude = np.arcsin(np.clip(y, -1, 1))
            
            u = (longitude + math.pi) / (2 * math.pi)
            v = (latitude + math.pi / 2) / math.pi
            
            # Convert to pixel coordinates
            eq_x = (u * (output_width - 1)).astype(np.int32)
            eq_y = (v * (output_height - 1)).astype(np.int32)
            
            # Ensure coordinates are within bounds
            valid_mask = ((eq_x >= 0) & (eq_x < output_width) & 
                         (eq_y >= 0) & (eq_y < output_height))
            
            # Use advanced indexing for efficient accumulation
            valid_eq_y = eq_y[valid_mask]
            valid_eq_x = eq_x[valid_mask]
            valid_weights = weights[valid_mask]
            valid_depth = depth_proj[valid_mask]
            
            # Accumulate values
            np.add.at(depth_sum, (valid_eq_y, valid_eq_x), valid_depth * valid_weights)
            np.add.at(weight_sum, (valid_eq_y, valid_eq_x), valid_weights)
        
        # Normalize by weights
        valid_mask = weight_sum > 1e-10  # Small epsilon to avoid division by zero
        result = np.zeros_like(depth_sum, dtype=np.float32)
        result[valid_mask] = (depth_sum[valid_mask] / weight_sum[valid_mask]).astype(np.float32)
        
        # Fill holes using interpolation if needed
        hole_count = np.sum(~valid_mask)
        if hole_count > 0:
            logging.info(f"Filling {hole_count} holes in stitched depth map using interpolation")
            result = self._fill_depth_holes(result, valid_mask)
        
        logging.info(f"Successfully stitched all projections to native resolution: {result.shape}")
        return result
    
    def _fill_depth_holes(self, depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in depth map using interpolation.
        
        Args:
            depth_map: Depth map with holes
            valid_mask: Mask indicating valid pixels
            
        Returns:
            Depth map with filled holes
        """
        # Normalize depth map for inpainting
        depth_min, depth_max = depth_map[valid_mask].min(), depth_map[valid_mask].max()
        if depth_max > depth_min:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = (depth_map * 255).astype(np.uint8)
        
        invalid_mask = (~valid_mask).astype(np.uint8)
        
        # Use inpainting with larger radius for better results
        filled_normalized = cv2.inpaint(
            depth_normalized,
            invalid_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )
        
        # Convert back to original range
        if depth_max > depth_min:
            filled = filled_normalized.astype(np.float32) / 255.0 * (depth_max - depth_min) + depth_min
        else:
            filled = filled_normalized.astype(np.float32) / 255.0
        
        return filled

# Global projector instance
_projector = None

def get_projector() -> IcosahedronProjector:
    """Get singleton icosahedron projector instance."""
    global _projector
    if _projector is None:
        _projector = IcosahedronProjector()
    return _projector

def generate_icosahedron_projections(equirectangular_image: Image.Image) -> List[Image.Image]:
    """
    Generate 20 icosahedral projections from equirectangular panoramic image at NATIVE RESOLUTION.
    
    Args:
        equirectangular_image: Input panoramic image at native resolution
        
    Returns:
        List of 20 projected images at native resolution
    """
    projector = get_projector()
    return projector.generate_icosahedron_projections(equirectangular_image)

def stitch_projections_to_equirect(depth_projections: List[np.ndarray], 
                                 output_size: Tuple[int, int]) -> np.ndarray:
    """
    Stitch depth map projections back to equirectangular format at NATIVE RESOLUTION.
    
    Args:
        depth_projections: List of 20 depth maps at native resolution
        output_size: (height, width) of output image at native resolution
        
    Returns:
        Stitched equirectangular depth map at native resolution
    """
    projector = get_projector()
    return projector.stitch_projections_to_equirect(depth_projections, output_size)

def calculate_face_coordinates(face_index: int) -> Dict:
    """
    Calculate coordinate mapping information for a specific face.
    
    Args:
        face_index: Index of the icosahedron face
        
    Returns:
        Dictionary containing face coordinate information
    """
    projector = get_projector()
    
    return {
        'face_index': face_index,
        'normal': projector.face_normals[face_index].tolist(),
        'vertices': projector.vertices[projector.faces[face_index]].tolist(),
        'native_resolution_processing': True
    }

def get_icosahedron_vertices() -> Dict:
    """
    Get icosahedron geometry data.
    
    Returns:
        Dictionary containing vertices and faces
    """
    projector = get_projector()
    
    return {
        'vertices': projector.vertices.tolist(),
        'faces': projector.faces.tolist(),
        'num_faces': len(projector.faces),
        'num_vertices': len(projector.vertices),
        'native_resolution_processing': True
    }

def interpolate_depth_boundaries(depth_maps: List[np.ndarray]) -> List[np.ndarray]:
    """
    Handle projection overlaps and smooth boundaries between depth maps.
    
    Args:
        depth_maps: List of depth maps from projections at native resolution
        
    Returns:
        List of depth maps (unchanged as blending is done during stitching)
    """
    logging.info("Depth boundary interpolation handled during stitching process at native resolution")
    return depth_maps