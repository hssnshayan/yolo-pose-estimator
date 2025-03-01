import math
import joblib

import numpy as np
from ultralytics import YOLO
from skeletons.skeletons import SKELETON_DISTANCE, SKELETON_ANGLE
import cv2 as cv

model = YOLO("models/yolo11n-pose.pt")
knn_model = joblib.load("models/best_knn.joblib")
capture = cv.VideoCapture(0)

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


while True:
    # live_img = cv.imread("images/test/02-0.jpg")
    ret, live_img = capture.read()
    results = model(live_img)

    key_points = results[0].keypoints.data.cpu().numpy()

    for person in key_points:
        X_record = []
        for key_point in person:
            key_point = key_point[:2].astype(np.uint16)
            X_record.extend(key_point.tolist())
            live_img = cv.circle(live_img, (key_point), 2, KEY_POINT_COLOR, 2)

        for connection in SKELETON_LINE:
            kp1_idx, kp2_idx = connection

            kp1 = person[kp1_idx][:2].astype(np.uint16)
            kp2 = person[kp2_idx][:2].astype(np.uint16)

            if kp1.sum() > 0 or kp2.sum() > 0:
                live_img = cv.line(live_img, kp1, kp2, CONNECTION_COLOR, 2)

        for angle in SKELETON_ANGLE:
            kp1_idx, kp2_idx, kp3_idx = SKELETON_ANGLE[angle]

            kp1 = person[kp1_idx][:2].astype(np.uint16)
            kp2 = person[kp2_idx][:2].astype(np.uint16)
            kp3 = person[kp3_idx][:2].astype(np.uint16)

            kp_angle = calculate_angle(kp1, kp2, kp3)
            X_record.append(kp_angle)

        for distance in SKELETON_DISTANCE:
            kp1_idx, kp2_idx = SKELETON_DISTANCE[distance]

            kp1 = person[kp1_idx][:2].astype(np.uint16)
            kp2 = person[kp2_idx][:2].astype(np.uint16)

            kp_distance = calculate_distance(kp1, kp2)
            X_record.append(kp_distance)

        X_record = np.array(X_record[10:]).reshape(1, -1)
        pe_predict = knn_model.predict(X_record)

        result = "Standing" if int(pe_predict[0]) else "Sitting"
        text = str(result)

        x = int(person[0][0])
        y = int(person[0][1])

        offset = 50

        position = (int(x - offset), int(y - offset))
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2

        cv.putText(live_img, text, position, font, font_scale, color, thickness)

    cv.imshow("Live", live_img)
    if cv.waitKey(1) == 13:
        break

cv.destroyAllWindows()
capture.release()
