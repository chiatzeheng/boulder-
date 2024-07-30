import cv2
import numpy as np
import networkx as nx
from scipy.spatial import distance

# Hold type instructions
HOLD_INSTRUCTIONS = {
    "Jug": "Wrap your whole hand around for a strong grip",
    "Edge": "Use fingertips and pull perpendicular to the hold",
    "Crimp": "Use fingertips, keep body close to wall",
    "Pinch": "Squeeze with thumb on one side, fingers on the other",
    "Sloper": "Use whole hand, keep weight low and opposed",
    "Pocket": "Insert fingers, pull in direction of the pocket",
    "Undercling": "Grip from below, find high footholds",
    "Horn": "Wrap hand around protrusion for a secure grip",
}


def create_color_range(color, tolerance=20):
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_color[0][0][0] - tolerance, 100, 100])
    upper = np.array([hsv_color[0][0][0] + tolerance, 255, 255])
    return lower, upper


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
        return "Pocket"  # Default classification


def construct_graph(holds, max_distance=150):
    G = nx.Graph()
    for i, hold in enumerate(holds):
        G.add_node(i, pos=(hold[0], hold[1]))

    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            dist = distance.euclidean(holds[i][:2], holds[j][:2])
            if dist <= max_distance:
                G.add_edge(i, j, weight=dist)

    return G


def find_optimal_path(G, start, end):
    try:
        path = nx.shortest_path(G, start, end, weight="weight")
        return path
    except nx.NetworkXNoPath:
        return None


def enhance_route(frame, mask, holds, path):
    # Create a blank mask for the route
    route_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Draw the holds and path on the route mask
    for hold in holds:
        cv2.circle(route_mask, (hold[0], hold[1]), 20, 255, -1)

    if path:
        for i in range(len(path) - 1):
            start = holds[path[i]][:2]
            end = holds[path[i + 1]][:2]
            cv2.line(route_mask, start, end, 255, 10)

    # Combine the hold mask and route mask
    combined_mask = cv2.bitwise_or(mask, route_mask)

    # Create a dimmed version of the original frame
    dimmed_frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)

    # Create a brightened version of the original frame for the route
    brightened_frame = cv2.addWeighted(frame, 1.5, np.zeros_like(frame), 0, 0)

    # Combine the dimmed and brightened frames using the mask
    result = np.where(combined_mask[:, :, None] == 255, brightened_frame, dimmed_frame)

    return result.astype(np.uint8)


def visualize_path(frame, holds, path, mask):
    result = enhance_route(frame, mask, holds, path)

    for i, hold in enumerate(holds):
        cv2.circle(result, (hold[0], hold[1]), 5, (0, 255, 0), -1)
        label = f"{hold[2]}: {HOLD_INSTRUCTIONS[hold[2]]}"
        cv2.putText(
            result,
            label,
            (hold[0] + 10, hold[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if path:
        for i in range(len(path) - 1):
            start = holds[path[i]][:2]
            end = holds[path[i + 1]][:2]
            cv2.line(result, start, end, (0, 0, 255), 2)

    return result


def main():
    cap = cv2.VideoCapture(0)

    color = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if color is None:
            cv2.putText(
                frame,
                "Press 'c' to select color",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Bouldering Assistant", frame)
        else:
            lower_color, upper_color = create_color_range(color)
            holds, mask = detect_holds(frame, lower_color, upper_color)

            if len(holds) > 1:
                graph = construct_graph(holds)
                start = min(range(len(holds)), key=lambda i: holds[i][1])
                end = max(range(len(holds)), key=lambda i: holds[i][1])
                path = find_optimal_path(graph, start, end)
                result = visualize_path(frame, holds, path, mask)
                cv2.imshow("Bouldering Assistant", result)
            else:
                cv2.putText(
                    frame,
                    "No route detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Bouldering Assistant", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            r = cv2.selectROI("Select Color", frame)
            color = frame[int(r[1] + r[3] / 2), int(r[0] + r[2] / 2)]
            cv2.destroyWindow("Select Color")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
