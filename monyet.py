import os
# Force XCB before any Qt/OpenCV import on Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from collections import deque
import time
import platform

# Hand landmark indices (MediaPipe standard 21-point hand model)
INDEX_FINGER_TIP = 8
THUMB_TIP = 4


class MonkeyExpressionMatcher:
    def __init__(self):
        # ── Download/verify task models ────────────────────────────────
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_model = os.path.join(model_dir, "face_landmarker_v2.task")
        self.hand_model = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(self.face_model):
            raise RuntimeError(
                f"Face landmark model not found at {self.face_model}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )
        if not os.path.exists(self.hand_model):
            raise RuntimeError(
                f"Hand landmark model not found at {self.hand_model}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        # ── Initialise MediaPipe Tasks API ─────────────────────────────
        face_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.face_model),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # ── Define monkey expressions ──────────────────────────────────
        self.monkey_expressions = {
            "Thinking": {
                "description": "Finger close to mouth",
                "color": (255, 180, 0),  # Orange
            },
            "Happy": {
                "description": "Mouth open + finger away from face",
                "color": (0, 255, 0),  # Green
            },
            "Shocked": {
                "description": "Mouth open with no hand",
                "color": (255, 0, 255),  # Magenta
            },
        }

        # ── Match history for smoothing ────────────────────────────────
        self.match_history = deque(maxlen=8)
        self.current_match = None
        self.match_confidence = 0.0
        self.match_start_time = None
        self.display_duration = 2.5

        # ── Load monkey images ─────────────────────────────────────────
        self.monkey_images = self.load_monkey_images()

        # ── FPS tracking ───────────────────────────────────────────────
        self.last_time = time.time()

    # ── Image loading ─────────────────────────────────────────────────

    def load_monkey_images(self):
        monkey_imgs = {}
        image_files = {
            "Thinking": "Monkey_Thinking.jpg",
            "Happy": "Monkey_Happy.jpg",
            "Shocked": "Monkey_Shocked.jpg",
        }

        for expr, filename in image_files.items():
            if os.path.exists(filename):
                img = cv2.imread(filename)
                if img is not None:
                    monkey_imgs[expr] = img
                    print(f"✓ Loaded: {filename}")
                else:
                    print(f"✗ Failed to load: {filename}")
                    monkey_imgs[expr] = self.create_placeholder(expr)
            else:
                print(f"✗ File not found: {filename}")
                monkey_imgs[expr] = self.create_placeholder(expr)

        return monkey_imgs

    def create_placeholder(self, expr):
        colors = {
            "Thinking": (180, 150, 100),
            "Happy": (100, 200, 100),
            "Shocked": (200, 100, 200),
        }
        img = np.ones((300, 300, 3), dtype=np.uint8)
        img[:] = colors.get(expr, (128, 128, 128))
        cv2.putText(img, expr, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    # ── Landmark helpers ──────────────────────────────────────────────

    @staticmethod
    def calculate_distance(point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def get_mouth_aspect_ratio(self, landmarks):
        """MAR: vertical / horizontal mouth opening."""
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        left_corner = landmarks[61]
        right_corner = landmarks[291]

        vertical = self.calculate_distance(upper_lip, lower_lip)
        horizontal = self.calculate_distance(left_corner, right_corner)
        return vertical / horizontal if horizontal > 0 else 0

    def detect_tongue_out(self, landmarks):
        return self.get_mouth_aspect_ratio(landmarks) > 0.3

    # ── Gesture detection ────────────────────────────────────────────

    def detect_finger_near_mouth(self, face_landmarks, hand_landmarks_list):
        if not hand_landmarks_list:
            return False

        mouth_center = face_landmarks[13]
        mouth_x, mouth_y = mouth_center.x, mouth_center.y

        for hand_landmarks in hand_landmarks_list:
            index_tip = hand_landmarks[INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks[THUMB_TIP]

            index_dist = np.sqrt((index_tip.x - mouth_x)**2 + (index_tip.y - mouth_y)**2)
            thumb_dist = np.sqrt((thumb_tip.x - mouth_x)**2 + (thumb_tip.y - mouth_y)**2)

            if index_dist < 0.10 or thumb_dist < 0.10:
                return True

        return False

    def detect_finger_away_from_face(self, face_landmarks, hand_landmarks_list):
        if not hand_landmarks_list:
            return False

        face_center = face_landmarks[1]  # Nose bridge
        face_x, face_y = face_center.x, face_center.y

        for hand_landmarks in hand_landmarks_list:
            index_tip = hand_landmarks[INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks[THUMB_TIP]

            index_dist = np.sqrt((index_tip.x - face_x)**2 + (index_tip.y - face_y)**2)
            thumb_dist = np.sqrt((thumb_tip.x - face_x)**2 + (thumb_tip.y - face_y)**2)

            if index_dist > 0.20 or thumb_dist > 0.20:
                return True

        return False

    # ── Feature extraction ──────────────────────────────────────────

    def extract_features(self, landmarks, hand_landmarks_list):
        features = {}

        # Mouth
        mouth_openness = self.get_mouth_aspect_ratio(landmarks)
        features["mouth_openness"] = mouth_openness
        features["mouth_open"] = mouth_openness > 0.20
        features["tongue_out"] = self.detect_tongue_out(landmarks)

        # Hands
        hand_detected = bool(hand_landmarks_list)
        features["hand_detected"] = hand_detected
        features["no_hand_detected"] = not hand_detected

        features["finger_near_mouth"] = False
        features["finger_away_from_face"] = False

        if hand_detected:
            features["finger_near_mouth"] = self.detect_finger_near_mouth(
                landmarks, hand_landmarks_list
            )
            features["finger_away_from_face"] = self.detect_finger_away_from_face(
                landmarks, hand_landmarks_list
            )

        return features

    # ── Expression matching ─────────────────────────────────────────

    def match_expression(self, features):
        # Rule 1: Thinking
        if features.get("finger_near_mouth", False):
            return "Thinking", 0.9

        # Rule 2: Happy
        if features.get("mouth_open", False) and features.get("finger_away_from_face", False):
            return "Happy", 0.9

        # Rule 3: Shocked
        if features.get("mouth_open", False) and features.get("no_hand_detected", False):
            return "Shocked", 0.9

        # Rule 4: Tongue out → Shocked
        if features.get("tongue_out", False):
            return "Shocked", 0.9

        return None, 0.0

    # ── Temporal smoothing ─────────────────────────────────────────

    def update_match_history(self, match, confidence):
        threshold = 0.40

        if confidence > threshold:
            self.match_history.append(match)
        else:
            self.match_history.append("Neutral")

        if len(self.match_history) > 0:
            valid = [m for m in self.match_history if m != "Neutral"]
            if len(valid) >= 2:
                most_common = max(set(valid), key=valid.count)
                if valid.count(most_common) >= 2:
                    if self.current_match != most_common:
                        self.current_match = most_common
                        self.match_start_time = time.time()
                    self.match_confidence = confidence
                    return

        if list(self.match_history).count("Neutral") >= 4:
            self.current_match = None
            self.match_confidence = 0.0

    # ── Overlay ────────────────────────────────────────────────────

    def draw_overlay(self, frame, expression, confidence):
        h, w = frame.shape[:2]

        if expression and expression in self.monkey_images:
            monkey_img = self.monkey_images[expression]
            mh, mw = monkey_img.shape[:2]

            target_height = int(h * 0.35)
            scale = target_height / mh
            new_w, new_h = int(mw * scale), target_height
            monkey_resized = cv2.resize(monkey_img, (new_w, new_h))

            x_off, y_off = w - new_w - 20, 20

            if y_off + new_h > h:
                new_h = h - y_off - 20
                new_w = int(new_h * (mw / mh))
                monkey_resized = cv2.resize(monkey_img, (new_w, new_h))

            roi = frame[y_off:y_off + new_h, x_off:x_off + new_w]
            if roi.shape[0] == new_h and roi.shape[1] == new_w:
                blended = cv2.addWeighted(roi, 0.05, monkey_resized, 0.95, 0)
                frame[y_off:y_off + new_h, x_off:x_off + new_w] = blended

                color = self.monkey_expressions[expression]["color"]
                cv2.rectangle(frame, (x_off - 3, y_off - 3),
                              (x_off + new_w + 3, y_off + new_h + 3), color, 4)

        if expression:
            label = f"MATCHED: {expression}!"
            conf_text = f"Confidence: {confidence:.1%}"

            (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            (cw, _), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            box_w, box_h = max(lw, cw) + 30, 85

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (box_w, box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            color = self.monkey_expressions[expression]["color"]
            cv2.rectangle(frame, (10, 10), (box_w, box_h), color, 3)
            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 2)

        return frame

    # ── HUD ────────────────────────────────────────────────────────

    def draw_status_info(self, frame, finger_near_mouth, finger_away_from_face, mouth_open, no_hand_detected):
        lines = []
        if finger_near_mouth:
            lines.append("THINKING: Finger near mouth")
        if finger_away_from_face and mouth_open:
            lines.append("HAPPY: Mouth open + finger away")
        elif finger_away_from_face:
            lines.append("Finger away from face")
        if mouth_open and no_hand_detected:
            lines.append("SHOCKED: Mouth open + no hand")
        elif mouth_open:
            lines.append("Mouth open")

        for i, text in enumerate(lines):
            tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            tx, ty = frame.shape[1] - tw - 20, 80 + i * 35

            cv2.rectangle(frame, (tx - 10, ty - 25),
                          (frame.shape[1] - 10, ty + 5), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)

            if "THINKING" in text:
                color = (255, 180, 0)
            elif "HAPPY" in text:
                color = (0, 255, 0)
            elif "SHOCKED" in text:
                color = (255, 0, 255)
            else:
                color = (200, 200, 200)

            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_expression_instructions(self, frame):
        instructions = [
            "THINKING: Finger close to mouth",
            "HAPPY: Mouth open + finger away from face",
            "SHOCKED: Mouth open + no hand detected",
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (15, 120 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Camera ─────────────────────────────────────────────────────

    def initialize_camera(self):
        print("Initializing camera...")
        print("Scanning for available cameras...")
        available = []

        for i in range(3):
            cap_test = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap_test.isOpened():
                ok, f = cap_test.read()
                if ok and f is not None:
                    available.append(i)
                    print(f"  Camera {i}: OK ({f.shape[1]}x{f.shape[0]})")
                else:
                    print(f"  Camera {i}: opened but no frame")
                cap_test.release()
            else:
                print(f"  Camera {i}: not available")
            time.sleep(0.3)

        if not available:
            print("No working cameras found!")
            return None

        idx = available[0]
        print(f"Using camera {idx}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(1)

        for _ in range(10):
            ok, f = cap.read()
            if ok and f is not None:
                print("Camera OK")
                return cap
            time.sleep(0.1)

        print("Camera test failed")
        cap.release()
        return None

    # ── Main loop ──────────────────────────────────────────────────

    def run(self):
        cap = self.initialize_camera()
        if cap is None:
            print("Camera initialization failed.")
            return

        print("\n" + "=" * 70)
        print("  MONKEY EXPRESSION MATCHER")
        print("=" * 70)
        print("\n  TRIGGERS:")
        print("    THINKING: Finger close to mouth")
        print("    HAPPY:    Mouth open + finger away from face")
        print("    SHOCKED:  Mouth open + no hand detected")
        print("\n" + "=" * 70)
        print("  Controls:  q=quit  r=reset  f=fullscreen")
        print("=" * 70 + "\n")

        fullscreen = False
        frame_count = 0
        last_fps_time = time.time()
        fps = 0

        cv2.namedWindow("Monkey Expression Matcher", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            frame_count += 1

            if not ok:
                print("Frame grab failed, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = self.initialize_camera()
                if cap is None:
                    break
                continue

            # FPS
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = now

            # Mirror
            frame = cv2.flip(frame, 1)

            # ── MediaPipe Tasks API inference ──────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_result = self.face_landmarker.detect(mp_image)
            hand_result = self.hand_landmarker.detect(mp_image)

            face_landmarks_list = face_result.face_landmarks if face_result else None
            hand_landmarks_list = hand_result.hand_landmarks if hand_result else None

            finger_near_mouth = False
            finger_away_from_face = False
            mouth_open = False
            no_hand_detected = True

            if face_landmarks_list:
                landmarks = face_landmarks_list[0]  # first face
                features = self.extract_features(landmarks, hand_landmarks_list)

                finger_near_mouth = features.get("finger_near_mouth", False)
                finger_away_from_face = features.get("finger_away_from_face", False)
                mouth_open = features.get("mouth_open", False)
                no_hand_detected = features.get("no_hand_detected", True)

                match, confidence = self.match_expression(features)
                self.update_match_history(match, confidence)

            # ── Draw HUD ───────────────────────────────────────────────
            self.draw_status_info(frame, finger_near_mouth, finger_away_from_face,
                                  mouth_open, no_hand_detected)
            self.draw_expression_instructions(frame)

            if self.current_match:
                frame = self.draw_overlay(frame, self.current_match, self.match_confidence)

            # FPS + status
            cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            n_faces = len(face_landmarks_list) if face_landmarks_list else 0
            n_hands = len(hand_landmarks_list) if hand_landmarks_list else 0
            status = f"Faces: {n_faces} | Hands: {n_hands}"
            cv2.putText(frame, status, (15, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Monkey Expression Matcher", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("Resetting camera...")
                cap.release()
                time.sleep(1)
                cap = self.initialize_camera()
                if cap is None:
                    break
            elif key == ord("f"):
                fullscreen = not fullscreen
                prop = cv2.WND_PROP_FULLSCREEN
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty("Monkey Expression Matcher", prop, mode)

        cap.release()
        cv2.destroyAllWindows()
        print("\nThanks for playing with the monkeys!\n")


if __name__ == "__main__":
    try:
        print("Starting Monkey Expression Matcher...")
        matcher = MonkeyExpressionMatcher()
        matcher.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
