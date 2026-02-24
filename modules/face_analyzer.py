import mediapipe as mp
import cv2
from .utils import eye_aspect_ratio

class FaceAnalyzer:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # Indices des points pour les yeux (mêmes qu'avant)
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def get_eye_landmarks(self, landmarks, frame_shape):
        h, w, _ = frame_shape
        left_eye = []
        right_eye = []
        for idx in self.LEFT_EYE_INDICES:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            left_eye.append((x, y))
        for idx in self.RIGHT_EYE_INDICES:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            right_eye.append((x, y))
        return left_eye, right_eye

    def compute_ear(self, left_eye, right_eye):
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        return (left_ear + right_ear) / 2.0