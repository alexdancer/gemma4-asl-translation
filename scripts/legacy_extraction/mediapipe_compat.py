"""Compatibility layer for MediaPipe 0.10+ using new tasks API."""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from typing import NamedTuple, Optional, List


class NormalizedLandmark(NamedTuple):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class NormalizedLandmarkList(NamedTuple):
    landmark: List[NormalizedLandmark]


class HolisticResults(NamedTuple):
    pose_landmarks: Optional[NormalizedLandmarkList] = None
    left_hand_landmarks: Optional[NormalizedLandmarkList] = None
    right_hand_landmarks: Optional[NormalizedLandmarkList] = None
    face_landmarks: Optional[NormalizedLandmarkList] = None


class Holistic:
    """Wrapper around MediaPipe Holistic pose detector using new tasks API."""

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize Holistic with the new MediaPipe tasks API."""
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        
        # Use the new pose landmarker from tasks API
        BaseOptions = mp_python.vision.BaseOptions
        PoseLandmarkerOptions = mp_python.vision.PoseLandmarkerOptions
        VisionRunningMode = mp_python.vision.VisionRunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=None),  # Uses default model
            running_mode=VisionRunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
        )
        
        self.pose_landmarker = mp_python.vision.PoseLandmarker.create_from_options(options)
        
        # For hands and face, we'll create separate landmarkers
        HandLandmarkerOptions = mp_python.vision.HandLandmarkerOptions
        self.hand_landmarker = None
        try:
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=None),
                running_mode=VisionRunningMode.IMAGE,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self.hand_landmarker = mp_python.vision.HandLandmarker.create_from_options(hand_options)
        except:
            pass  # Hand landmarker may not be available
        
    def process(self, rgb_image):
        """Process image and return landmarks in old API format."""
        import cv2
        
        # Convert RGB numpy to MediaPipe Image
        if isinstance(rgb_image, np.ndarray):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        else:
            mp_image = rgb_image
        
        # Get results
        pose_result = self.pose_landmarker.detect(mp_image)
        
        # Convert to old API format
        pose_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None
        face_landmarks = None
        
        if pose_result and pose_result.landmarks:
            landmarks_list = pose_result.landmarks[0]
            pose_landmarks = NormalizedLandmarkList(
                landmark=[NormalizedLandmark(l.x, l.y, l.z, getattr(l, 'visibility', 1.0)) 
                         for l in landmarks_list]
            )
        
        # Try to get hand landmarks
        if self.hand_landmarker:
            try:
                hand_result = self.hand_landmarker.detect(mp_image)
                if hand_result and hand_result.landmarks:
                    # Separate left and right hands (assuming first is right, second is left)
                    if len(hand_result.landmarks) > 0:
                        landmarks_list = hand_result.landmarks[0]
                        right_hand_landmarks = NormalizedLandmarkList(
                            landmark=[NormalizedLandmark(l.x, l.y, l.z, 1.0) for l in landmarks_list]
                        )
                    if len(hand_result.landmarks) > 1:
                        landmarks_list = hand_result.landmarks[1]
                        left_hand_landmarks = NormalizedLandmarkList(
                            landmark=[NormalizedLandmark(l.x, l.y, l.z, 1.0) for l in landmarks_list]
                        )
            except:
                pass
        
        return HolisticResults(
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks
        )
    
    def close(self):
        """Clean up resources."""
        if self.pose_landmarker:
            try:
                self.pose_landmarker.close()
            except:
                pass
        if self.hand_landmarker:
            try:
                self.hand_landmarker.close()
            except:
                pass


# Monkey-patch mediapipe to add solutions namespace
class SolutionsNamespace:
    class HolisticNamespace:
        Holistic = Holistic
    
    holistic = HolisticNamespace()


mp.solutions = SolutionsNamespace()
