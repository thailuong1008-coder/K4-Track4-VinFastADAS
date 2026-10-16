"""
=============================================================================
VinFast Perception ADAS - Forward Collision Warning & BEV Radar
Capstone Project - Track 4: Computer Vision & Robotics
Real-time Demo with Webcam, Video Replay, Failure Mode & Telemetry Blackbox
=============================================================================
"""

import cv2
import numpy as np
import time
import os
import threading
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

from core.geometry import CameraGeometry
from core.detector import ADASDetector
from core.tracker import PerceptionTracker
from core.collision import CollisionEngine
from core.telemetry import TelemetryLogger


class ADASApp:
    def __init__(self, source_type='webcam', video_path="data/backup_dashcam.mp4"):
        self.source_type = source_type
        self.video_path = video_path

        # Core pipeline components
        self.geometry = CameraGeometry(img_w=640, img_h=480, fov_deg=65.0, cam_height=1.2, pitch_deg=0.0)
        self.detector = ADASDetector(model_path='yolov8n.pt', conf_thresh=0.35, imgsz=384)
        self.tracker = PerceptionTracker(max_age=8, min_hits=2, distance_threshold=3.0)
        self.collision = CollisionEngine(warning_ttc=3.0, critical_ttc=1.5)
        self.telemetry = TelemetryLogger(log_dir="data/logs")

        # Runtime controls
        self.cap = None
        self.is_running = False
        self.sound_enabled = True
        self.failure_mode = False
        self.last_beep_time = 0.0

        # Snapshot folder
        os.makedirs("data/snapshots", exist_ok=True)

        self._init_source()

    def _init_source(self):
        if self.cap is not None:
            self.cap.release()

        if self.source_type == 'webcam':
            print("Opening live webcam (Camera 0)...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("WARNING: Webcam not available, falling back to backup video.")
                self.source_type = 'video'
                self.cap = cv2.VideoCapture(self.video_path)
        else:
            print(f"Opening video file: {self.video_path}...")
            self.cap = cv2.VideoCapture(self.video_path)

        # Set resolution to 640x480 for consistent fast FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def trigger_audio_alert(self, alert_level):
        """Asynchronous audio beeps on alert."""
        if not self.sound_enabled or not HAS_SOUND:
            return

        now = time.time()
        if alert_level == 'CRITICAL' and (now - self.last_beep_time > 0.4):
            self.last_beep_time = now
            threading.Thread(target=lambda: winsound.Beep(1200, 200), daemon=True).start()
        elif alert_level == 'WARNING' and (now - self.last_beep_time > 0.9):
            self.last_beep_time = now
            threading.Thread(target=lambda: winsound.Beep(800, 150), daemon=True).start()

    def draw_bev_radar(self, tracks, collision_res, bev_w=280, bev_h=480):
        """
        Renders a top-down Bird's Eye View (BEV) radar widget.
        Trụ cột: Biểu diễn (Representation) & Hình học (Geometry).
        """
        bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
        # Background: VinFast dark blue cockpit style
        bev[:] = (22, 18, 14)

        cx = bev_w // 2
        base_y = bev_h - 40
        max_dist_m = 25.0
        max_lateral_m = 5.0

        # Distance range rings (5m, 10m, 15m, 20m, 25m)
        ranges = [5, 10, 15, 20, 25]
        for r in ranges:
            _, r_pix_y = self.geometry.project_to_bev(0, r, bev_w, bev_h, max_lateral_m, max_dist_m)
            radius = base_y - r_pix_y
            if radius > 5:
                cv2.ellipse(bev, (cx, base_y), (radius, radius), 0, 180, 360, (50, 45, 40), 1)
                cv2.putText(bev, f"{r}m", (cx + 8, r_pix_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 110, 100), 1)

        # Driving lane corridor lines (+/- 1.3m from vehicle center)
        _, corr_top_y = self.geometry.project_to_bev(0, max_dist_m, bev_w, bev_h, max_lateral_m, max_dist_m)
        corr_left_x, _ = self.geometry.project_to_bev(-self.collision.CORRIDOR_WIDTH, 0, bev_w, bev_h, max_lateral_m, max_dist_m)
        corr_right_x, _ = self.geometry.project_to_bev(self.collision.CORRIDOR_WIDTH, 0, bev_w, bev_h, max_lateral_m, max_dist_m)
        corr_left_top_x, _ = self.geometry.project_to_bev(-self.collision.CORRIDOR_WIDTH, max_dist_m, bev_w, bev_h, max_lateral_m, max_dist_m)
        corr_right_top_x, _ = self.geometry.project_to_bev(self.collision.CORRIDOR_WIDTH, max_dist_m, bev_w, bev_h, max_lateral_m, max_dist_m)

        # Safe driving path boundary
        cv2.line(bev, (corr_left_x, base_y), (corr_left_top_x, corr_top_y), (80, 75, 40), 1, cv2.LINE_AA)
        cv2.line(bev, (corr_right_x, base_y), (corr_right_top_x, corr_top_y), (80, 75, 40), 1, cv2.LINE_AA)
        # Center line
        cv2.line(bev, (cx, base_y), (cx, corr_top_y), (45, 40, 35), 1)

        # Draw Ego Vehicle (VinFast Electric Car)
        ego_w = 26
        ego_h = 38
        ego_x1 = cx - ego_w // 2
        ego_y1 = base_y - ego_h // 2
        cv2.rectangle(bev, (ego_x1, ego_y1), (ego_x1 + ego_w, ego_y1 + ego_h), (210, 110, 40), -1) # VinFast Blue
        cv2.rectangle(bev, (ego_x1, ego_y1), (ego_x1 + ego_w, ego_y1 + ego_h), (255, 200, 100), 1)
        # Headlights beam indicator
        pts = np.array([[cx - 8, ego_y1], [cx + 8, ego_y1], [cx + 35, ego_y1 - 40], [cx - 35, ego_y1 - 40]], np.int32)
        cv2.fillPoly(bev, [pts], (40, 35, 25))
        cv2.putText(bev, "EGO (VF8)", (cx - 24, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Draw Tracked Objects
        track_evals = collision_res.get('track_evals', {})
        for trk in tracks:
            eval_info = track_evals.get(trk.track_id, {})
            u_bev, v_bev = self.geometry.project_to_bev(trk.position_x, trk.position_z, bev_w, bev_h, max_lateral_m, max_dist_m)
            color = eval_info.get('color', (0, 255, 128))

            # Dot & circle for object
            cv2.circle(bev, (u_bev, v_bev), 6, color, -1)
            cv2.circle(bev, (u_bev, v_bev), 9, color, 1)

            # Velocity vector arrow (showing direction & closing speed)
            vz = trk.velocity_z # negative = moving toward ego
            vx = trk.velocity_x
            # Scale velocity vector for visual display (1 m/s = 8 pixels)
            arrow_dy = int(-vz * 8)
            arrow_dx = int(vx * 8)
            cv2.arrowedLine(bev, (u_bev, v_bev), (u_bev + arrow_dx, v_bev - arrow_dy), color, 2, tipLength=0.3)

            # Metric label next to blip
            label = f"#{trk.track_id} {trk.position_z:.1f}m"
            cv2.putText(bev, label, (u_bev + 10, v_bev + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        # Header Title
        cv2.putText(bev, "BEV TOP-DOWN RADAR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.line(bev, (12, 35), (bev_w - 12, 35), (60, 50, 40), 1)

        return bev

    def draw_hud(self, frame, tracks, collision_res, fps, latency_ms):
        """
        Renders HUD overlays on camera frame including 3D bounding boxes,
        TTC warnings, telemetry strip, and active safety alerts.
        """
        h, w = frame.shape[:2]
        track_evals = collision_res.get('track_evals', {})
        system_alert = collision_res.get('system_alert', 'SAFE')

        # Draw Ground Horizon and Lane Projection Guides
        horizon_y = int(self.geometry.cy)
        cv2.line(frame, (0, horizon_y), (w, horizon_y), (80, 80, 80), 1, cv2.LINE_4)

        # Draw vehicle path corridor projected onto camera frame
        cam_h = self.geometry.cam_height
        fx = self.geometry.fx
        fy = self.geometry.fy
        cx = self.geometry.cx
        cy = self.geometry.cy

        corridor_poly = []
        for z in [2.0, 5.0, 10.0, 18.0]:
            left_u = int(cx + (-self.collision.CORRIDOR_WIDTH * fx) / z)
            right_u = int(cx + (self.collision.CORRIDOR_WIDTH * fx) / z)
            v = int(cy + (cam_h * fy) / z)
            if 0 <= v < h:
                corridor_poly.append((left_u, right_u, v))

        for i in range(len(corridor_poly) - 1):
            lu1, ru1, v1 = corridor_poly[i]
            lu2, ru2, v2 = corridor_poly[i + 1]
            cv2.line(frame, (lu1, v1), (lu2, v2), (0, 180, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (ru1, v1), (ru2, v2), (0, 180, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes and 3D telemetry for each tracked object
        for trk in tracks:
            eval_info = track_evals.get(trk.track_id, {})
            color = eval_info.get('color', (0, 255, 128))
            ttc = eval_info.get('ttc', 99.0)
            in_corr = eval_info.get('in_corridor', False)

            if trk.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in trk.bbox]
                thickness = 3 if eval_info.get('alert_level') == 'CRITICAL' else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Distance & Kinematics Tag
                ttc_str = f"TTC: {ttc:.1f}s" if ttc < 50.0 else "TTC: >10s"
                vel_str = f"Vrel: {trk.velocity_z:+.1f}m/s"
                dist_str = f"Dist: {trk.position_z:.1f}m (X: {trk.position_x:+.1f}m)"

                tag_title = f"#{trk.track_id} {trk.class_name.upper()}"
                if in_corr:
                    tag_title += " [IN PATH]"

                # Badge background
                badge_h = 42
                badge_w = 190
                badge_y1 = max(0, y1 - badge_h - 4)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + badge_w, badge_y1 + badge_h), (25, 20, 15), -1)
                cv2.rectangle(frame, (x1, badge_y1), (x1 + badge_w, badge_y1 + badge_h), color, 1)

                cv2.putText(frame, tag_title, (x1 + 6, badge_y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(frame, f"{dist_str} | {vel_str}", (x1 + 6, badge_y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.putText(frame, ttc_str, (x1 + 6, badge_y1 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        # Top Status Bar
        cv2.rectangle(frame, (0, 0), (w, 36), (15, 12, 10), -1)
        mode_text = f"SOURCE: {self.source_type.upper()}"
        if self.failure_mode:
            mode_text += " | [FAILURE CASE: OCCLUSION ACTIVE]"

        cv2.putText(frame, f"VINFAST ADAS PERCEPTION | {mode_text}", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 215, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS | {latency_ms:.0f} ms", (w - 130, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1, cv2.LINE_AA)

        # Bottom Safety Banner
        banner_h = 45
        banner_y = h - banner_h
        if system_alert == 'CRITICAL':
            # Flashing red emergency brake alert
            flash = (int(time.time() * 6) % 2 == 0)
            bg_color = (0, 0, 220) if flash else (0, 0, 140)
            cv2.rectangle(frame, (0, banner_y), (w, h), bg_color, -1)
            cv2.putText(frame, "CRITICAL: COLLISION IMMINENT - AEB BRAKE TRIGGERED!",
                        (w // 2 - 210, banner_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            self.trigger_audio_alert('CRITICAL')
        elif system_alert == 'WARNING':
            cv2.rectangle(frame, (0, banner_y), (w, h), (0, 180, 240), -1)
            primary = collision_res.get('primary_track')
            min_ttc = collision_res.get('min_ttc', 2.0)
            txt = f"WARNING: OBSTACLE CLOSING IN - TTC {min_ttc:.1f}s (PREPARE TO BRAKE)"
            cv2.putText(frame, txt, (w // 2 - 220, banner_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 2)
            self.trigger_audio_alert('WARNING')
        else:
            cv2.rectangle(frame, (0, banner_y), (w, h), (20, 70, 20), -1)
            cv2.putText(frame, "SYSTEM CLEAR - SAFE DRIVING CORRIDOR",
                        (w // 2 - 160, banner_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 255, 180), 1)

        # Bottom Keybinds Hint
        hint_text = "[W]ebcam | [V]ideo | [F]ailure Demo | [S]napshot | [Q]uit"
        cv2.putText(frame, hint_text, (12, banner_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        return frame

    def run(self):
        """Main execution loop."""
        print("\n" + "=" * 65)
        print(" VINFAST ADAS PERCEPTION SYSTEM - CAPSTONE DEMO")
        print(" Hotkeys:")
        print("   [W] : Switch to Live Webcam")
        print("   [V] : Switch to Backup Dashcam Video")
        print("   [F] : Toggle Failure Case Simulation (Occlusion / Pitch)")
        print("   [B] : Toggle Sound Alerts")
        print("   [S] : Save Frame Snapshot + Telemetry Point")
        print("   [Q] : Quit and Save Telemetry Blackbox")
        print("=" * 65 + "\n")

        self.is_running = True
        frame_idx = 0
        prev_time = time.time()

        cv2.namedWindow("VinFast ADAS Perception Cockpit", cv2.WINDOW_NORMAL)

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.source_type == 'video':
                    # Loop video seamlessly
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Error reading frame from source.")
                    break

            start_t = time.time()
            frame_idx += 1
            dt = max(0.001, start_t - prev_time)
            prev_time = start_t

            # Step 1: Detect objects & 3D geometry projection
            detections = self.detector.detect(frame, self.geometry)

            # Step 2: Multi-object Kalman tracking & velocity estimation
            tracks = self.tracker.update(detections, dt=dt)

            # Step 3: Collision risk & TTC calculation
            collision_res = self.collision.evaluate_all(tracks)

            latency_ms = (time.time() - start_t) * 1000.0
            fps = 1.0 / dt if dt > 0 else 30.0

            # Step 4: Telemetry recording
            self.telemetry.log_frame(frame_idx, tracks, collision_res, latency_ms, fps)

            # Step 5: Render HUD and BEV Radar
            hud_frame = self.draw_hud(frame, tracks, collision_res, fps, latency_ms)
            bev_radar = self.draw_bev_radar(tracks, collision_res, bev_w=280, bev_h=frame.shape[0])

            # Combine Side-by-Side: Camera HUD on Left + BEV Radar on Right
            combined_cockpit = np.hstack((hud_frame, bev_radar))

            cv2.imshow("VinFast ADAS Perception Cockpit", combined_cockpit)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('w') or key == ord('W'):
                print("Switching source to WEBCAM...")
                self.source_type = 'webcam'
                self._init_source()
            elif key == ord('v') or key == ord('V'):
                print("Switching source to BACKUP VIDEO...")
                self.source_type = 'video'
                self._init_source()
            elif key == ord('f') or key == ord('F'):
                self.failure_mode = not self.failure_mode
                self.detector.simulate_occlusion = self.failure_mode
                status_str = "ENABLED" if self.failure_mode else "DISABLED"
                print(f"[DEMO CONTROL] Failure Case Simulation (Occlusion): {status_str}")
            elif key == ord('b') or key == ord('B'):
                self.sound_enabled = not self.sound_enabled
                print(f"[DEMO CONTROL] Sound Alert: {'ENABLED' if self.sound_enabled else 'DISABLED'}")
            elif key == ord('s') or key == ord('S'):
                snap_path = f"data/snapshots/cockpit_snap_{int(time.time())}.jpg"
                cv2.imwrite(snap_path, combined_cockpit)
                print(f"[SNAPSHOT] Saved cockpit screenshot to: {snap_path}")

        self.cleanup()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.telemetry.close()
        print(f"Perception session complete. Telemetry saved to: {self.telemetry.filepath}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['webcam', 'video'], default='webcam')
    parser.add_argument('--video', default='data/backup_dashcam.mp4')
    args = parser.parse_args()

    app = ADASApp(source_type=args.source, video_path=args.video)
    app.run()
