import io
import PIL.Image
import numpy as np
from math import sqrt

def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())

def img_qt_to_arr(img_qt):
    w, h, d = img_qt.size().width(), img_qt.size().height(), img_qt.depth()
    bytes_ = img_qt.bits().asstring(w * h * d // 8)
    img_arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
    return img_arr

def img_pil_to_data(img_pil):
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_data = f.getvalue()
    return img_data

def img_arr_to_data(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    img_data = img_pil_to_data(img_pil)
    return img_data

def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    if np.linalg.norm(p2 - p1) == 0:
        return np.linalg.norm(p3 - p1)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def find_box_coordinates(box):
    # Extract x and y coordinates from the input array
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    
    # Find xmin, ymin, xmax, ymax
    xmin = min(x_values)
    ymin = min(y_values)
    xmax = max(x_values)
    ymax = max(y_values)
    
    return [xmin, ymin, xmax, ymax]

def img_qt_to_arr(img_qt):
    w, h, d = img_qt.size().width(), img_qt.size().height(), img_qt.depth()
    bytes_ = img_qt.bits().tobytes()  # Use tobytes() instead of asstring()
    img_arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
    img_arr = img_arr[:,:, :3]

    return img_arr.copy()