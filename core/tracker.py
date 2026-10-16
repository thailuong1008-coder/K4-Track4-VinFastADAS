"""
Core Tracker & Kalman Filter Module - VinFast Perception ADAS
Handles 3D State Estimation, Measurement Uncertainty Adaptation, and Temporal Velocity Estimation.
Trụ cột: Bất định (Uncertainty), Thời gian (Temporal), Biểu diễn (Representation).
"""

import numpy as np


class KalmanTrack3D:
    """
    Kalman filter tracking a single 3D object state in vehicle coordinates:
    State: [x, z, vx, vz]^T
    x: Lateral position (m)
    z: Longitudinal distance (m)
    vx: Lateral velocity (m/s)
    vz: Longitudinal relative velocity (m/s) - negative means approaching!
    """
    _count = 0

    def __init__(self, init_x, init_z, class_name, confidence, dt=0.04):
        KalmanTrack3D._count += 1
        self.track_id = KalmanTrack3D._count
        self.class_name = class_name
        self.dt = dt
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        # State vector [x, z, vx, vz]
        self.x = np.array([init_x, init_z, 0.0, 0.0], dtype=np.float64)

        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Measurement matrix H (we measure x and z from camera)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Process noise covariance Q (vehicle dynamics uncertainty)
        q_pos = 0.05
        q_vel = 0.5
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # State covariance P
        self.P = np.diag([0.2, 0.5, 2.0, 2.0]).astype(np.float64)

        # Store latest bounding box and confidence
        self.bbox = None
        self.confidence = confidence

    def predict(self, dt=None):
        """State prediction step x = F*x, P = F*P*F^T + Q"""
        if dt is not None and dt > 0:
            self.dt = dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.x

    def update(self, meas_x, meas_z, bbox, confidence):
        """
        Measurement update step with Adaptive Uncertainty Covariance R.
        Trụ cột Bất định: R tăng khi confidence thấp hoặc khoảng cách xa.
        """
        self.time_since_update = 0
        self.hits += 1
        self.bbox = bbox
        self.confidence = confidence

        # Adaptive measurement noise R based on detector confidence and depth
        # Depth variance grows quadratically with distance in monocular vision
        var_x = 0.04 / max(0.2, confidence)
        var_z = (0.05 + 0.003 * (meas_z ** 2)) / max(0.2, confidence)
        R = np.diag([var_x, var_z]).astype(np.float64)

        # Innovation: y = z_meas - H*x
        z_meas = np.array([meas_x, meas_z], dtype=np.float64)
        y = z_meas - np.dot(self.H, self.x)

        # Innovation covariance: S = H*P*H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + R

        # Kalman gain: K = P*H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update state and covariance
        self.x = self.x + np.dot(K, y)
        I = np.eye(4, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    @property
    def position_x(self):
        return float(self.x[0])

    @property
    def position_z(self):
        return float(self.x[1])

    @property
    def velocity_x(self):
        return float(self.x[2])

    @property
    def velocity_z(self):
        """Relative longitudinal velocity (m/s). Negative = closing in."""
        return float(self.x[3])


class PerceptionTracker:
    """
    Multi-object tracker associating 3D observations with Kalman tracks.
    """
    def __init__(self, max_age=8, min_hits=2, distance_threshold=2.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracks = []

    def update(self, detections, dt=0.04):
        """
        detections: list of dicts:
            {'bbox': [x1, y1, x2, y2], 'class_name': str, 'conf': float, 'pos_3d': (x, y, z)}
        """
        # Step 1: Predict existing tracks
        for trk in self.tracks:
            trk.predict(dt)

        matched_tracks = set()
        matched_dets = set()

        # Step 2: Distance-based gating & association in 3D ground plane
        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, trk in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    dx = trk.position_x - det['pos_3d'][0]
                    dz = trk.position_z - det['pos_3d'][2]
                    dist_3d = np.sqrt(dx * dx + dz * dz)
                    cost_matrix[t_idx, d_idx] = dist_3d

            # Greedy matching for lowest distance
            while True:
                min_val = np.min(cost_matrix)
                if min_val > self.distance_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    cost_matrix[t_idx, d_idx] = 9999.0
                    continue

                # Associate
                trk = self.tracks[t_idx]
                det = detections[d_idx]
                trk.update(
                    meas_x=det['pos_3d'][0],
                    meas_z=det['pos_3d'][2],
                    bbox=det['bbox'],
                    confidence=det['conf']
                )
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                cost_matrix[t_idx, :] = 9999.0
                cost_matrix[:, d_idx] = 9999.0

        # Step 3: Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_trk = KalmanTrack3D(
                    init_x=det['pos_3d'][0],
                    init_z=det['pos_3d'][2],
                    class_name=det['class_name'],
                    confidence=det['conf'],
                    dt=dt
                )
                new_trk.bbox = det['bbox']
                self.tracks.append(new_trk)

        # Step 4: Prune dead tracks
        active_tracks = []
        for trk in self.tracks:
            if trk.time_since_update <= self.max_age:
                active_tracks.append(trk)
        self.tracks = active_tracks

        # Return confirmed tracks
        confirmed = [t for t in self.tracks if (t.hits >= self.min_hits or t.age == 1)]
        return confirmed
