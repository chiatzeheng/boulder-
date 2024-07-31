import cv2
import numpy as np
import networkx as nx
from scipy.spatial import distance


# Hold type instructions and difficulty ratings (1-10, 10 being most difficult)
HOLD_INFO = {
    "Jug": {
        "instruction": "Wrap your whole hand around for a strong grip",
        "difficulty": 2,
    },
    "Edge": {
        "instruction": "Use fingertips and pull perpendicular to the hold",
        "difficulty": 5,
    },
    "Crimp": {
        "instruction": "Use fingertips, keep body close to wall",
        "difficulty": 8,
    },
    "Pinch": {
        "instruction": "Squeeze with thumb on one side, fingers on the other",
        "difficulty": 6,
    },
    "Sloper": {
        "instruction": "Use whole hand, keep weight low and opposed",
        "difficulty": 7,
    },
    "Pocket": {
        "instruction": "Insert fingers, pull in direction of the pocket",
        "difficulty": 6,
    },
    "Undercling": {
        "instruction": "Grip from below, find high footholds",
        "difficulty": 7,
    },
    "Horn": {
        "instruction": "Wrap hand around protrusion for a secure grip",
        "difficulty": 4,
    },
}


def create_color_range(color, tolerance=32):
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_color[0][0][0] - tolerance, 100, 100])
    upper = np.array([hsv_color[0][0][0] + tolerance, 255, 255])
    return lower, upper


def construct_graph(holds, max_distance=150):
    G = nx.Graph()
    for i, hold in enumerate(holds):
        G.add_node(i, pos=(hold[0], hold[1]), type=hold[2])

    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            dist = distance.euclidean(holds[i][:2], holds[j][:2])
            if dist <= max_distance:
                difficulty = (
                    HOLD_INFO[holds[i][2]]["difficulty"]
                    + HOLD_INFO[holds[j][2]]["difficulty"]
                ) / 2
                G.add_edge(i, j, weight=dist, difficulty=difficulty)

    return G


def find_viable_path(G, start, end, max_difficulty=8):
    def dfs_path(current, visited, path):
        if current == end:
            return path

        visited.add(current)
        neighbors = sorted(
            G.neighbors(current),
            key=lambda n: (G.nodes[n]["pos"][1], G[current][n]["difficulty"]),
        )

        for neighbor in neighbors:
            if (
                neighbor not in visited
                and G[current][neighbor]["difficulty"] <= max_difficulty
            ):
                new_path = dfs_path(neighbor, visited.copy(), path + [neighbor])
                if new_path:
                    return new_path

        return None

    return dfs_path(start, set(), [start])


def enhance_route(frame, mask, holds, path):
    route_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for hold in holds:
        cv2.circle(route_mask, (hold[0], hold[1]), 20, 255, -1)

    if path:
        for i in range(len(path) - 1):
            start = holds[path[i]][:2]
            end = holds[path[i + 1]][:2]
            cv2.line(route_mask, start, end, 255, 10)

    combined_mask = cv2.bitwise_or(mask, route_mask)

    dimmed_frame = cv2.addWeighted(frame, 0.3, np.zeros_like(frame), 0.7, 0)
    brightened_frame = cv2.addWeighted(frame, 1.5, np.zeros_like(frame), 0, 0)

    result = np.where(combined_mask[:, :, None] == 255, brightened_frame, dimmed_frame)

    return result.astype(np.uint8)


def visualize_path(frame, holds, path, mask):
    result = enhance_route(frame, mask, holds, path)

    for i, hold in enumerate(holds):
        cv2.circle(result, (hold[0], hold[1]), 5, (0, 255, 0), -1)
        label = f"{hold[2]}: {HOLD_INFO[hold[2]]['instruction']}"
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
