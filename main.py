import cv2
from classifictation import detect_holds
from visualizer import (
    construct_graph,
    create_color_range,
    find_viable_path,
    visualize_path,
)


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
                path = find_viable_path(graph, start, end)
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
