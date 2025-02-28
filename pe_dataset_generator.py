from pathlib import Path
import math
import csv

import numpy as np
from ultralytics import YOLO
import cv2 as cv

model = YOLO("models/yolo11n-pose.pt")
SKELETON_LINE = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
SKELETON_ANGLE = {
    "left_shoulder": [11, 5, 7],
    "right_shoulder": [8, 6, 12],
    "left_elbow": [5, 7, 9],
    "right_elbow": [6, 8, 10],
    "left_hip": [5, 11, 13],
    "right_hip": [6, 12, 14],
    "left_knee": [11, 13, 15],
    "right_knee": [12, 14, 16],
}
SKELETON_DISTANCE = {
    "left_hip_to_ankle": [11, 15],
    "right_hip_to_ankle": [12, 16],
    "left_shoulder_to_knee": [5, 13],
    "right_shoulder_to_knee": [6, 14],
}

KEY_POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 0, 255)


def calculate_angle(A, B, C):
    BA = (int(A[0]) - int(B[0]), int(A[1]) - int(B[1]))
    BC = (int(C[0]) - int(B[0]), int(C[1]) - int(B[1]))

    angle_BA = math.atan2(BA[1], BA[0])
    angle_BC = math.atan2(BC[1], BC[0])

    angle_radians = angle_BC - angle_BA

    angle_radians = (angle_radians + 2 * math.pi) % (2 * math.pi)

    angle_degrees = round(math.degrees(angle_radians), 2)

    return angle_degrees


def calculate_distance(A, B):
    A = A.tolist()
    B = B.tolist()
    return round(
        math.sqrt((int(B[0]) - int(A[0])) ** 2 + (int(B[1]) - int(A[1])) ** 2), 2
    )


images_directory = Path("images")

with open("dataset/dataset.csv", "w", newline="") as file:
    csv_writer = csv.writer(file)

    for image_path in images_directory.iterdir():
        if image_path.is_file():
            image_label = int(image_path.name.split(".")[0].split("-")[1])
            img = cv.imread(image_path)
            results = model(img)

            key_points = results[0].keypoints.data.cpu().numpy()

            for person in key_points:
                dataset_record = []
                for key_point in person:
                    key_point = key_point[:2].astype(np.uint16)
                    dataset_record.extend(key_point.tolist())
                    img = cv.circle(img, (key_point), 2, KEY_POINT_COLOR, 2)

                for connection in SKELETON_LINE:
                    kp1_idx, kp2_idx = connection

                    kp1 = person[kp1_idx][:2].astype(np.uint16)
                    kp2 = person[kp2_idx][:2].astype(np.uint16)

                    if kp1.sum() > 0 or kp2.sum() > 0:
                        img = cv.line(img, kp1, kp2, CONNECTION_COLOR, 2)

                for angle in SKELETON_ANGLE:
                    kp1_idx, kp2_idx, kp3_idx = SKELETON_ANGLE[angle]

                    kp1 = person[kp1_idx][:2].astype(np.uint16)
                    kp2 = person[kp2_idx][:2].astype(np.uint16)
                    kp3 = person[kp3_idx][:2].astype(np.uint16)

                    kp_angle = calculate_angle(kp1, kp2, kp3)
                    dataset_record.append(kp_angle)

                for distance in SKELETON_DISTANCE:
                    kp1_idx, kp2_idx = SKELETON_DISTANCE[distance]

                    kp1 = person[kp1_idx][:2].astype(np.uint16)
                    kp2 = person[kp2_idx][:2].astype(np.uint16)

                    kp_distance = calculate_distance(kp1, kp2)
                    dataset_record.append(kp_distance)

                dataset_record.append(image_label)

                csv_writer.writerow(dataset_record)
