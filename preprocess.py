import numpy as np
import torch
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
EXPECTED_LENGTH = 188

def extract_landmarks(results):
    def lm_to_array(lm_list, expected):
        if lm_list is None:
            return np.zeros((expected, 3))
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark])
        if pts.shape[0] < expected:
            pts = np.pad(pts, ((0, expected - pts.shape[0]), (0, 0)))
        return pts

    lh = lm_to_array(results.left_hand_landmarks, 21)
    rh = lm_to_array(results.right_hand_landmarks, 21)
    pose = lm_to_array(results.pose_landmarks, 33)

    combined = np.concatenate([lh, rh, pose], axis=0).flatten()

    if combined.shape[0] < EXPECTED_LENGTH:
        combined = np.pad(combined, (0, EXPECTED_LENGTH - combined.shape[0]))

    return combined

def preprocess_frame(pil_frame):
    img = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    features = extract_landmarks(results)
    return torch.tensor(features, dtype=torch.float32)