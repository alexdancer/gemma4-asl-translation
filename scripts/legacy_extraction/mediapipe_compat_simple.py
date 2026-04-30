"""Simple compatibility layer for MediaPipe 0.10+ using new tasks API."""

import mediapipe as mp
from mediapipe.tasks import python
from typing import NamedTuple, Optional, List


class NormalizedLandmark:
    def __init__(self, x, y, z, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class NormalizedLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class HolisticResults:
    def __init__(self, pose_landmarks=None, left_hand_landmarks=None, 
                 right_hand_landmarks=None, face_landmarks=None):
        self.pose_landmarks = pose_landmarks
        self.left_hand_landmarks = left_hand_landmarks
        self.right_hand_landmarks = right_hand_landmarks
        self.face_landmarks = face_landmarks


class Holistic:
    """Wrapper for MediaPipe pose detection using tasks API."""

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        from mediapipe.tasks.python import vision
        
        self.vision = vision
        
        BaseOptions = vision.BaseOptions
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.VisionRunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(),
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
        )
        
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.hand_landmarker = None
        
        try:
            HandLandmarkerOptions = vision.HandLandmarkerOptions
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(),
                running_mode=VisionRunningMode.IMAGE,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        except Exception as e:
            print(f"Warning: Could not initialize hand landmarker: {e}")
        
    def process(self, rgb_image):
        """Process image and return results in old API format."""
        import numpy as np
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        pose_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None
        face_landmarks = None
        
        try:
            # Get pose landmarks
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result and pose_result.landmarks and len(pose_result.landmarks) > 0:
                landmarks = []
                for landmark in pose_result.landmarks[0]:
                    landmarks.append(NormalizedLandmark(landmark.x, landmark.y, landmark.z, 1.0))
                pose_landmarks = NormalizedLandmarkList(landmarks)
            
            # Get hand landmarks
            if self.hand_landmarker:
                hand_result = self.hand_landmarker.detect(mp_image)
                if hand_result and hand_result.landmarks:
                    for i, hand_landmarks_list in enumerate(hand_result.landmarks):
                        landmarks = []
                        for landmark in hand_landmarks_list:
                            landmarks.append(NormalizedLandmark(landmark.x, landmark.y, landmark.z, 1.0))
                        hand_list = NormalizedLandmarkList(landmarks)
                        # Assume first is right, second is left
                        if i == 0:
                            right_hand_landmarks = hand_list
                        elif i == 1:
                            left_hand_landmarks = hand_list
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
        
        return HolisticResults(
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks
        )
    
    def close(self):
        """Clean up resources."""
        try:
            if self.pose_landmarker:
                self.pose_landmarker.close()
        except:
            pass
        try:
            if self.hand_landmarker:
                self.hand_landmarker.close()
        except:
            pass


# Monkey-patch mediapipe
class Solutions:
    class Holistic_NS:
        Holistic = Holistic
    holistic = Holistic_NS()

mp.solutions = Solutions()
