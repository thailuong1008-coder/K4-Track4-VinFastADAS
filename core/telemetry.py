"""
Core Telemetry & Blackbox Module - VinFast Perception ADAS
Records frame-by-frame telemetry, metric states, TTC, and system latencies to CSV.
Trụ cột: Dữ liệu (Data) & Bằng chứng (Evidence).
"""

import csv
import os
import time


class TelemetryLogger:
    def __init__(self, log_dir="data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = None
        self.csv_writer = None
        self.start_time = time.time()
        self.record_count = 0
        self.init_new_session()

    def init_new_session(self):
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.log_dir, f"telemetry_{timestr}.csv")
        self.log_file = open(self.filepath, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.log_file)
        # Header matching ADAS blackbox standards
        self.csv_writer.writerow([
            "timestamp_rel_s", "frame_idx", "track_id", "class_name",
            "pos_x_m", "pos_z_m", "vel_x_mps", "vel_z_mps",
            "ttc_s", "alert_level", "confidence", "latency_ms", "fps"
        ])
        self.log_file.flush()

    def log_frame(self, frame_idx, tracks, collision_res, latency_ms, fps):
        rel_time = round(time.time() - self.start_time, 3)
        track_evals = collision_res.get('track_evals', {})

        if not tracks:
            # Log empty frame heartbeat
            self.csv_writer.writerow([
                rel_time, frame_idx, -1, "none",
                0.0, 0.0, 0.0, 0.0,
                99.0, "SAFE", 0.0, round(latency_ms, 1), round(fps, 1)
            ])
            self.record_count += 1
            return

        for trk in tracks:
            eval_info = track_evals.get(trk.track_id, {})
            ttc = eval_info.get('ttc', 99.0)
            alert = eval_info.get('alert_level', 'SAFE')

            self.csv_writer.writerow([
                rel_time,
                frame_idx,
                trk.track_id,
                trk.class_name,
                round(trk.position_x, 2),
                round(trk.position_z, 2),
                round(trk.velocity_x, 2),
                round(trk.velocity_z, 2),
                ttc,
                alert,
                round(trk.confidence, 2),
                round(latency_ms, 1),
                round(fps, 1)
            ])
            self.record_count += 1

        if self.record_count % 15 == 0:
            self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()
