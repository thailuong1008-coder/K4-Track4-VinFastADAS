"""
Core Collision Warning & TTC Engine - VinFast Perception ADAS
Calculates Time-To-Collision (TTC), corridor filtering, and multi-tier active safety alerts.
Trụ cột: Thời gian (Temporal) & Biểu diễn (Representation).
"""

import numpy as np


class CollisionEngine:
    """
    Evaluates collision risk, calculates TTC, and generates AEB triggers.
    """
    # Safety thresholds in seconds
    TTC_WARNING = 3.0       # Yellow warning
    TTC_CRITICAL = 1.5      # Red emergency brake
    MIN_DISTANCE_SAFE = 1.8 # Minimum stationary buffer (meters)

    # Ego vehicle drive corridor half-width (meters)
    CORRIDOR_WIDTH = 1.3    # Standard lane width around centerline

    def __init__(self, warning_ttc=3.0, critical_ttc=1.5):
        self.warning_ttc = warning_ttc
        self.critical_ttc = critical_ttc

    def evaluate_track(self, track):
        """
        Calculates TTC and alert status for a given Kalman track.
        Returns:
            dict with:
                'ttc': float (seconds, 99.0 if safe/moving away)
                'alert_level': 'SAFE' | 'WARNING' | 'CRITICAL'
                'in_corridor': bool
                'color': (B, G, R) tuple for OpenCV rendering
        """
        x = track.position_x
        z = track.position_z
        vz = track.velocity_z  # Negative = approaching

        in_corridor = abs(x) <= self.CORRIDOR_WIDTH

        # Relative velocity closing speed (m/s)
        closing_speed = -vz

        if closing_speed > 0.2 and z > 0.1:
            ttc = z / closing_speed
        else:
            # Moving away or stationary
            ttc = 99.0

        # Proximity override: if extremely close (< 1.5m), danger regardless of speed
        if z < self.MIN_DISTANCE_SAFE and in_corridor:
            alert_level = 'CRITICAL'
            color = (0, 0, 255) # Red
            ttc = min(ttc, 0.8)
        elif in_corridor and ttc <= self.critical_ttc:
            alert_level = 'CRITICAL'
            color = (0, 0, 255) # Red
        elif in_corridor and (ttc <= self.warning_ttc or z < 3.5):
            alert_level = 'WARNING'
            color = (0, 215, 255) # Yellow/Amber
        else:
            alert_level = 'SAFE'
            color = (0, 255, 128) # Green

        return {
            'ttc': float(np.round(ttc, 2)),
            'alert_level': alert_level,
            'in_corridor': in_corridor,
            'color': color,
            'closing_speed': float(np.round(closing_speed, 2))
        }

    def evaluate_all(self, tracks):
        """
        Evaluates all active tracks and determines the primary most-dangerous obstacle.
        """
        results = {}
        primary_track = None
        min_ttc = 99.0
        system_alert = 'SAFE'

        for trk in tracks:
            eval_res = self.evaluate_track(trk)
            results[trk.track_id] = eval_res

            # Check for most critical threat in driving corridor
            if eval_res['in_corridor']:
                if eval_res['alert_level'] == 'CRITICAL':
                    system_alert = 'CRITICAL'
                elif eval_res['alert_level'] == 'WARNING' and system_alert != 'CRITICAL':
                    system_alert = 'WARNING'

                if eval_res['ttc'] < min_ttc:
                    min_ttc = eval_res['ttc']
                    primary_track = trk

        return {
            'track_evals': results,
            'primary_track': primary_track,
            'min_ttc': min_ttc if min_ttc < 90.0 else None,
            'system_alert': system_alert
        }
