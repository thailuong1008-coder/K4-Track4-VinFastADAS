"""
Generate Backup Sample Video for VinFast ADAS Perception Demo.
Synthesizes a realistic forward dashcam video with road perspective, lane lines,
an approaching vehicle decelerating/accelerating, and a pedestrian crossing.
This ensures a 100% reliable backup video ready for the live demo.
"""

import cv2
import numpy as np
import os


def generate_backup_video(output_path="data/sample_video.mp4", duration_sec=15, fps=25):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = int(duration_sec * fps)
    print(f"Generating {total_frames} frames of backup ADAS video...")

    # Horizon line
    horizon_y = int(h * 0.45)

    for i in range(total_frames):
        # Progress 0 to 1
        t = (i % (fps * 7)) / (fps * 7)

        # Background: Sky and asphalt
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Sky: gradient blue/gray
        frame[:horizon_y, :] = [180, 160, 140]
        # Asphalt road
        frame[horizon_y:, :] = [45, 45, 50]

        # Draw road perspective lines (trapezoid)
        road_poly = np.array([
            [int(w * 0.44), horizon_y],
            [int(w * 0.56), horizon_y],
            [w, h],
            [0, h]
        ], np.int32)
        cv2.fillPoly(frame, [road_poly], (55, 55, 60))

        # Lane markings
        # Left lane boundary
        cv2.line(frame, (int(w * 0.44), horizon_y), (0, h), (255, 255, 255), 2)
        # Right lane boundary
        cv2.line(frame, (int(w * 0.56), horizon_y), (w, h), (255, 255, 255), 2)

        # Center dashed lane
        dash_offset = int((i * 8) % 60)
        for dy in range(horizon_y + 10, h, 40):
            y_start = dy + (dash_offset % 20)
            if y_start < h:
                y_end = min(h, y_start + 18)
                x_center = int(w * 0.5)
                # perspective width
                frac = (y_start - horizon_y) / (h - horizon_y)
                thickness = max(1, int(1 + 3 * frac))
                cv2.line(frame, (x_center, y_start), (x_center, y_end), (240, 240, 240), thickness)

        # Scenario 1: Approaching vehicle in front
        # Distance oscillates from 30m down to 4m, then moves away
        cycle_t = (i % 250) / 250.0
        if cycle_t < 0.65:
            # Approaching
            z = 35.0 - (cycle_t / 0.65) * 29.0 # from 35m to 6m
        else:
            # Pulling away / stationary
            z = 6.0 + ((cycle_t - 0.65) / 0.35) * 20.0

        # Scale car by 1/z
        car_scale = 18.0 / z
        car_w = int(60 * car_scale)
        car_h = int(45 * car_scale)

        # Center car in lane with slight curve
        car_cx = int(w * 0.5 + 20 * np.sin(i * 0.05))
        # Car ground contact y
        car_y2 = int(horizon_y + (h - horizon_y) * (18.0 / (z + 8.0)))
        car_y1 = car_y2 - car_h
        car_x1 = car_cx - car_w // 2
        car_x2 = car_x1 + car_w

        if car_y2 < h and car_w > 10 and car_h > 10:
            # Draw synthetic car body (VinFast Red VF8 look)
            cv2.rectangle(frame, (car_x1, car_y1 + car_h // 3), (car_x2, car_y2), (40, 30, 180), -1) # Red body
            # Cabin / roof
            roof_x1 = car_x1 + car_w // 6
            roof_x2 = car_x2 - car_w // 6
            cv2.rectangle(frame, (roof_x1, car_y1), (roof_x2, car_y1 + car_h // 3), (30, 20, 130), -1)
            # Rear windshield
            cv2.rectangle(frame, (roof_x1 + 2, car_y1 + 2), (roof_x2 - 2, car_y1 + car_h // 3 - 2), (180, 200, 220), -1)
            # Taillights (VinFast signature LED bar)
            tl_y = car_y1 + car_h // 2
            cv2.line(frame, (car_x1 + 3, tl_y), (car_x2 - 3, tl_y), (0, 0, 255), max(1, int(2 * car_scale)))
            # Wheels
            wheel_w = max(2, int(car_w * 0.16))
            wheel_h = max(2, int(car_h * 0.25))
            cv2.rectangle(frame, (car_x1, car_y2 - wheel_h), (car_x1 + wheel_w, car_y2), (15, 15, 15), -1)
            cv2.rectangle(frame, (car_x2 - wheel_w, car_y2 - wheel_h), (car_x2, car_y2), (15, 15, 15), -1)

        # Scenario 2: Pedestrian crossing periodically
        if (i % 200) > 80 and (i % 200) < 170:
            ped_t = ((i % 200) - 80) / 90.0
            ped_x = int(w * 0.15 + ped_t * (w * 0.4))
            ped_z = 12.0
            ped_scale = 18.0 / ped_z
            ped_w = int(22 * ped_scale)
            ped_h = int(60 * ped_scale)
            ped_y2 = int(horizon_y + (h - horizon_y) * (18.0 / (ped_z + 8.0)))
            ped_y1 = ped_y2 - ped_h

            # Draw pedestrian (head + body)
            head_r = max(2, int(ped_w * 0.35))
            cv2.circle(frame, (ped_x, ped_y1 + head_r), head_r, (200, 180, 160), -1)
            cv2.rectangle(frame, (ped_x - ped_w // 2, ped_y1 + 2 * head_r), (ped_x + ped_w // 2, ped_y2 - head_r), (80, 120, 50), -1)
            cv2.line(frame, (ped_x - 3, ped_y2 - head_r), (ped_x - 3, ped_y2), (40, 40, 40), 2)
            cv2.line(frame, (ped_x + 3, ped_y2 - head_r), (ped_x + 3, ped_y2), (40, 40, 40), 2)

        # Watermark
        cv2.putText(frame, "VINFAST ADAS DASHCAM BACKUP STREAM (1080p -> 480p)", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Sample video created successfully at: {output_path}")


if __name__ == '__main__':
    generate_backup_video()
