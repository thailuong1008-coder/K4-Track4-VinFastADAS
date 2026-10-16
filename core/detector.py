"""
Core Detector Module - VinFast Perception ADAS
Wraps YOLOv8 with targeted classes, geometry calculation, and failure case simulation.
Trụ cột: Học (Learning), Dữ liệu (Data), Bằng chứng (Evidence).
"""

from ultralytics import YOLO
import numpy as np


class ADASDetector:
    # Target classes of interest in ADAS + desk demonstration objects
    TARGET_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        39: 'bottle',      # Useful for indoor desk demo
        67: 'cell phone'   # Useful for indoor desk demo
    }

    def __init__(self, model_path='yolov8n.pt', conf_thresh=0.35, imgsz=384):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        # Failure mode simulation flags
        self.simulate_occlusion = False
        self.simulate_pitch_error = False

    def detect(self, frame, geometry):
        """
        Runs object detection and projects each bounding box into 3D metric coordinates.
        Returns:
            list of dicts: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'class_name': str,
                    'conf': float,
                    'pos_3d': (x, y, z),
                    'is_occluded': bool
                }, ...
            ]
        """
        h, w = frame.shape[:2]
        geometry.update_resolution(w, h)

        # Run inference
        results = self.model(
            frame,
            conf=self.conf_thresh,
            imgsz=self.imgsz,
            verbose=False,
            classes=list(self.TARGET_CLASSES.keys())
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = self.TARGET_CLASSES.get(cls_id, 'car')

            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy

            is_occluded = False
            # If failure simulation is active: simulate partial occlusion
            # (e.g. bottom half of pedestrian obscured by car hood)
            if self.simulate_occlusion:
                orig_h = y2 - y1
                y1 = y1 + orig_h * 0.45  # Truncate top/bottom
                is_occluded = True

            # Estimate 3D metric position (X, Y, Z)
            x3d, y3d, z3d = geometry.estimate_3d_position([x1, y1, x2, y2], class_name=class_name)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': class_name,
                'conf': conf,
                'pos_3d': (x3d, y3d, z3d),
                'is_occluded': is_occluded
            })

        return detections
