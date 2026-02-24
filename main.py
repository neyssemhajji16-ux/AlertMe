import cv2
import json
from modules.camera import Camera
from modules.face_analyzer import FaceAnalyzer
from modules.head_pose import HeadPoseEstimator
from modules.alarm import Alarm
from modules.logger import Logger

def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    config = load_config()

    cam = Camera(config.get('camera_id', 0))
    face_analyzer = FaceAnalyzer(
        detection_confidence=config.get('face_detection_confidence', 0.5),
        tracking_confidence=config.get('face_tracking_confidence', 0.5)
    )
    head_pose_estimator = HeadPoseEstimator() if config.get('use_head_pose', True) else None
    alarm = Alarm(
        sound_file=config.get('alarm_sound'),
        video_url=config.get('alarm_video_url'),
        use_sound=config.get('use_sound', True),
        use_video=config.get('use_video', True)
    )
    logger = Logger(log_file=config.get('log_file', 'logs/detector.log'))

    EAR_THRESHOLD = config.get('ear_threshold', 0.2)
    EAR_CONSEC_FRAMES = config.get('ear_consec_frames', 20)
    HEAD_POSE_THRESHOLD_YAW = config.get('head_pose_threshold_yaw', 40)
    HEAD_POSE_THRESHOLD_PITCH = config.get('head_pose_threshold_pitch', 25)
    HEAD_POSE_CONSEC_FRAMES = config.get('head_pose_consec_frames', 15)
    NO_FACE_CONSEC_FRAMES = config.get('no_face_consec_frames', 30)
    INVERT_PITCH = config.get('head_pose_invert_pitch', False)

    ear_counter = 0
    head_pose_counter = 0
    no_face_counter = 0
    alarm_active = False

    logger.info("Drowsiness Detector started")

    while True:
        frame = cam.get_frame()
        if frame is None:
            logger.error("Failed to grab frame")
            break

        results = face_analyzer.process(frame)
        face_detected = results and results.multi_face_landmarks

        if face_detected:
            # Réinitialiser le compteur d'absence
            no_face_counter = 0

            landmarks = results.multi_face_landmarks[0]
            left_eye, right_eye = face_analyzer.get_eye_landmarks(landmarks, frame.shape)
            ear = face_analyzer.compute_ear(left_eye, right_eye)

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Détection yeux fermés
            eyes_closed = ear < EAR_THRESHOLD
            ear_counter = ear_counter + 1 if eyes_closed else 0
            eyes_closed_long = ear_counter >= EAR_CONSEC_FRAMES

            # Détection tête
            head_away = False
            if head_pose_estimator:
                yaw, pitch, roll = head_pose_estimator.estimate(landmarks, frame.shape)
                if INVERT_PITCH:
                    pitch = -pitch
                cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f} (inv:{INVERT_PITCH})", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if abs(yaw) > HEAD_POSE_THRESHOLD_YAW or pitch < -HEAD_POSE_THRESHOLD_PITCH:
                    head_pose_counter += 1
                else:
                    head_pose_counter = 0
                head_away = head_pose_counter >= HEAD_POSE_CONSEC_FRAMES

            # Condition de déclenchement (visage présent)
            trigger = eyes_closed_long or head_away

        else:
            # Aucun visage détecté
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            no_face_counter += 1
            # Réinitialiser les compteurs liés au visage
            ear_counter = 0
            head_pose_counter = 0
            # Déclenchement si absence prolongée
            trigger = no_face_counter >= NO_FACE_CONSEC_FRAMES

        # Logique commune d'alarme
        if trigger:
            if not alarm_active:
                logger.warning("ALARM triggered - No face or inattention detected")
                alarm.trigger()
                alarm_active = True
        else:
            if alarm_active:
                alarm.stop()
                alarm_active = False

        cv2.imshow("Drowsiness Detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            alarm.stop()
            alarm_active = False
            logger.info("Alarme stoppée manuellement")

    cam.release()
    cv2.destroyAllWindows()
    alarm.cleanup()
    logger.info("Drowsiness Detector stopped")

if __name__ == "__main__":
    main()