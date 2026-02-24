from .utils import get_head_pose

class HeadPoseEstimator:
    def estimate(self, landmarks, frame_shape):
        return get_head_pose(landmarks, frame_shape)