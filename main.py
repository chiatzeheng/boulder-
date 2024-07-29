import cv2
import numpy as np
import networkx as nx
from scipy.spatial import distance


# function to determine clr range with tolerance
def create_color_range(color, tolerance=20):
    color = np.uint8([[color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_color[0][0][0] - tolerance, 100, 100])
    upper = np.array([hsv_color[0][0][0] + tolerance, 255, 255])
    return lower, upper

#detecting holds as nodes 
def detect_holds(frame, lower_color, upper_color):
    #convert to hsv 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    holds = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Adjust these values based on your specific case
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                holds.append((cx, cy))
    
    return holds

def construct_graph(holds, max_distance=150):
    G = nx.Graph()
    for i, hold in enumerate(holds):
        G.add_node(i, pos=hold)
    
    for i in range(len(holds)):
        for j in range(i + 1, len(holds)):
            dist = distance.euclidean(holds[i], holds[j])
            if dist <= max_distance:
                G.add_edge(i, j, weight=dist)
    
    return G

def find_optimal_path(G, start, end):
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def visualize_path(frame, holds, path):
    result = frame.copy()
    
    # Draw all holds
    for hold in holds:
        cv2.circle(result, hold, 5, (0, 255, 0), -1)
    
    # Draw optimal path
    if path:
        for i in range(len(path) - 1):
            start = holds[path[i]]
            end = holds[path[i + 1]]
            cv2.line(result, start, end, (0, 0, 255), 2)
    
    return result

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, or provide a URL for IP camera
    
    color = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if color is None:
            cv2.putText(frame, "Press 'c' to select color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Bouldering Assistant', frame)
        else:
            lower_color, upper_color = create_color_range(color)
            holds = detect_holds(frame, lower_color, upper_color)
            
            if len(holds) > 1:
                graph = construct_graph(holds)
                start = min(range(len(holds)), key=lambda i: holds[i][1])
                end = max(range(len(holds)), key=lambda i: holds[i][1])
                path = find_optimal_path(graph, start, end)
                result = visualize_path(frame, holds, path)
                cv2.imshow('Bouldering Assistant', result)
            else:
                cv2.putText(frame, "No route detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Bouldering Assistant', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            r = cv2.selectROI("Select Color", frame)
            color = frame[int(r[1]+r[3]/2), int(r[0]+r[2]/2)]
            cv2.destroyWindow("Select Color")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
