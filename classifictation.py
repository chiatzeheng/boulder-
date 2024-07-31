import cv2
import numpy as np


def detect_holds(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    holds = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hold_type = classify_hold(contour)
                holds.append((cx, cy, hold_type))

    return holds, mask


def classify_hold(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity > 0.8:
        return "Jug"
    elif circularity < 0.3:
        return "Edge"
    elif area < 500:
        return "Crimp"
    elif 0.3 <= circularity <= 0.6:
        return "Pinch"
    elif area > 2000:
        return "Sloper"
    else:
        return "Pocket"
