import cv2
import numpy as np
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(landmarks, image_shape):
    h, w, _ = image_shape
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # Points 3D du modèle (en mm)
    model_points = np.array([
        [0.0, 0.0, 0.0],         # Nez (1)
        [0.0, -330.0, -65.0],    # Menton (199)
        [-225.0, 170.0, -135.0], # Oeil gauche ext (33)
        [225.0, 170.0, -135.0],  # Oeil droit ext (263)
        [-150.0, -150.0, -125.0],# Bouche gauche (61)
        [150.0, -150.0, -125.0], # Bouche droite (291)
        [-150.0, 170.0, -135.0], # Oeil gauche int (133)
        [150.0, 170.0, -135.0]   # Oeil droit int (362)
    ])

    # Points 2D correspondants (indices MediaPipe)
    image_points = np.array([
        [landmarks.landmark[1].x * w, landmarks.landmark[1].y * h],
        [landmarks.landmark[199].x * w, landmarks.landmark[199].y * h],
        [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],
        [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],
        [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h],
        [landmarks.landmark[133].x * w, landmarks.landmark[133].y * h],
        [landmarks.landmark[362].x * w, landmarks.landmark[362].y * h]
    ], dtype=np.float32)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)

    # Calcul des angles (yaw, pitch, roll) en degrés
    # Méthode robuste à partir de la matrice de rotation
    sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rmat[2,1], rmat[2,2])  # pitch
        y = np.arctan2(-rmat[2,0], sy)        # yaw
        z = np.arctan2(rmat[1,0], rmat[0,0])  # roll
    else:
        x = np.arctan2(-rmat[1,2], rmat[1,1])
        y = np.arctan2(-rmat[2,0], sy)
        z = 0

    # Conversion en degrés
    pitch = np.degrees(x)
    yaw = np.degrees(y)
    roll = np.degrees(z)

    return yaw, pitch, roll