"""
Core Geometry Module - VinFast Perception ADAS
Implements Pinhole Camera Model, Ground Plane Intersection, and Bird's Eye View (BEV) Projection.
Trụ cột: Hình học (Geometry) & Biểu diễn (Representation)
"""

import numpy as np


class CameraGeometry:
    """
    Handles camera projection, monocular 3D distance estimation, and BEV mapping.
    Based on pinhole camera geometry and flat ground plane assumptions.
    """
    # Typical real-world object dimensions (meters) [height, width]
    CLASS_DIMENSIONS = {
        'person': {'height': 1.70, 'width': 0.50},
        'car': {'height': 1.50, 'width': 1.80},
        'truck': {'height': 2.80, 'width': 2.40},
        'bus': {'height': 3.20, 'width': 2.55},
        'motorcycle': {'height': 1.25, 'width': 0.80},
        'bicycle': {'height': 1.10, 'width': 0.60},
        'bottle': {'height': 0.25, 'width': 0.08},    # For desk/webcam testing with everyday objects
        'cell phone': {'height': 0.15, 'width': 0.07}, # For desk/webcam testing
        'default': {'height': 1.60, 'width': 1.00}
    }

    def __init__(self, img_w=640, img_h=480, fov_deg=60.0, cam_height=1.2, pitch_deg=0.0):
        """
        Args:
            img_w: Image width in pixels
            img_h: Image height in pixels
            fov_deg: Horizontal Field of View (FOV) in degrees
            cam_height: Height of camera above ground plane (meters)
            pitch_deg: Camera tilt angle (degrees, positive = looking down)
        """
        self.img_w = img_w
        self.img_h = img_h
        self.cam_height = cam_height
        self.pitch_rad = np.radians(pitch_deg)

        # Optical center (principal point)
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0

        # Focal lengths from FOV
        fov_rad = np.radians(fov_deg)
        self.fx = (img_w / 2.0) / np.tan(fov_rad / 2.0)
        # Assuming square pixels
        self.fy = self.fx

    def update_resolution(self, w, h):
        """Update intrinsics if video resolution changes."""
        if w != self.img_w or h != self.img_h:
            scale_x = w / self.img_w
            scale_y = h / self.img_h
            self.img_w = w
            self.img_h = h
            self.cx = w / 2.0
            self.cy = h / 2.0
            self.fx *= scale_x
            self.fy *= scale_y

    def estimate_3d_position(self, bbox, class_name='car', mode='fused'):
        """
        Estimate 3D metric coordinates (X, Y, Z) in camera/vehicle coordinate frame.
        X: Lateral offset (meters, positive = right, negative = left)
        Y: Vertical height (meters, positive = down towards ground)
        Z: Longitudinal distance (meters, forward ahead)

        Args:
            bbox: [x1, y1, x2, y2] pixel coordinates
            class_name: Name of detected class
            mode: 'fused', 'height', or 'ground'
        """
        x1, y1, x2, y2 = bbox
        bbox_w = max(1.0, x2 - x1)
        bbox_h = max(1.0, y2 - y1)
        u_center = (x1 + x2) / 2.0
        v_bottom = y2

        dims = self.CLASS_DIMENSIONS.get(class_name, self.CLASS_DIMENSIONS['default'])
        h_real = dims['height']

        # Method 1: Apparent Height (Pinhole Scaling)
        # Z = (f_y * H_real) / h_pixels
        z_height = (self.fy * h_real) / bbox_h

        # Method 2: Ground Plane Intersection (Ray-Ground Intersection)
        # Ray angle relative to camera optical axis
        alpha = np.arctan((v_bottom - self.cy) / self.fy)
        total_angle = alpha + self.pitch_rad

        if total_angle > np.radians(1.5): # Avoid division by zero near horizon
            z_ground = self.cam_height / np.tan(total_angle)
        else:
            z_ground = z_height

        # Sensor Fusion of height & ground models
        if mode == 'height':
            z = z_height
        elif mode == 'ground':
            z = z_ground
        else:
            # Fused: Ground plane is very reliable close to camera,
            # Height model is more stable at far distances.
            weight_ground = np.clip(1.0 - (z_height / 35.0), 0.2, 0.8)
            z = weight_ground * z_ground + (1.0 - weight_ground) * z_height

        # Boundary checks: distance cannot be negative or absurdly large
        z = float(np.clip(z, 0.3, 80.0))

        # Lateral position X from pinhole projection
        x = float(((u_center - self.cx) * z) / self.fx)

        # Vertical position Y of object base
        y = float(self.cam_height)

        return x, y, z

    def project_to_bev(self, x_metric, z_metric, bev_w=240, bev_h=300, max_x=5.0, max_z=25.0):
        """
        Map metric (X, Z) into 2D Bird's Eye View (BEV) radar pixel coordinates.
        Args:
            x_metric: lateral position in meters (-max_x to +max_x)
            z_metric: longitudinal distance in meters (0 to max_z)
            bev_w: Width of BEV widget in pixels
            bev_h: Height of BEV widget in pixels
        Returns:
            (u_bev, v_bev) pixel coordinates in BEV canvas
        """
        # Center x is at bev_w / 2
        # Vehicle origin (z=0) is at bottom of BEV canvas (bev_h - 20)
        bev_cx = bev_w / 2.0
        bev_base_y = bev_h - 25

        u_bev = int(bev_cx + (x_metric / max_x) * (bev_w / 2.0 - 15))
        v_bev = int(bev_base_y - (z_metric / max_z) * (bev_base_y - 20))

        u_bev = int(np.clip(u_bev, 5, bev_w - 5))
        v_bev = int(np.clip(v_bev, 5, bev_h - 5))

        return u_bev, v_bev
