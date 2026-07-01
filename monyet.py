from collections import deque
import argparse
import time
import os
import platform

cv2 = None
mp = None
np = None
torch = None
F = None
transforms = None
Image = None
load_model_with_metadata = None
get_monkey_map = None


def load_runtime_dependencies():
    global cv2, mp, np, torch, F, transforms, Image, load_model_with_metadata, get_monkey_map
    if cv2 is not None:
        return

    import cv2 as cv2_module
    import mediapipe as mp_module
    import numpy as np_module
    import torch as torch_module
    import torch.nn.functional as functional_module
    from torchvision import transforms as transforms_module
    from PIL import Image as image_module
    from model import load_model_with_metadata as load_model_with_metadata_fn
    from dataset import get_monkey_map as get_monkey_map_fn

    cv2 = cv2_module
    mp = mp_module
    np = np_module
    torch = torch_module
    F = functional_module
    transforms = transforms_module
    Image = image_module
    load_model_with_metadata = load_model_with_metadata_fn
    get_monkey_map = get_monkey_map_fn


class MonkeyExpressionMatcher:
    def __init__(
        self,
        model_path="best_model.pth",
        confidence_threshold=0.50,
        shocked_threshold=0.80,
        happy_threshold=0.55,
        camera_index=None,
        debug=False,
        debug_dir="debug_frames",
        mirror=True,
        camera_width=1280,
        camera_height=720,
    ):
        load_runtime_dependencies()
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.shocked_threshold = shocked_threshold
        self.happy_threshold = happy_threshold
        self.camera_index = camera_index
        self.debug = debug
        self.debug_dir = debug_dir
        self.mirror = mirror
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.last_probs = None
        self.last_prediction = ""
        self.last_decision = ""
        self.last_face_bbox = None
        self.debug_frame_index = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Run training first: python train.py --data data/fer2013.csv")
            raise FileNotFoundError(model_path)
        self.expression_model, self.model_metadata = load_model_with_metadata(model_path, self.device)
        self.class_names = self.model_metadata["class_names"]
        self.label_mode = self.model_metadata["label_mode"]
        self.image_size = self.model_metadata["image_size"]
        self.monkey_map = get_monkey_map(self.label_mode)
        print(f"Model loaded from {model_path}")
        print(f"  Label mode: {self.label_mode}")
        print(f"  Classes: {', '.join(self.class_names)}")
        if self.model_metadata.get("val_acc") is not None:
            print(f"  Validation accuracy: {self.model_metadata['val_acc']:.2%}")
        if self.model_metadata.get("test_acc") is not None:
            print(f"  Test accuracy: {self.model_metadata['test_acc']:.2%}")

        # Preprocessing transform (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Face/hand detection. Newer MediaPipe builds may expose only Tasks API,
        # so keep an OpenCV face detector fallback for Python 3.13 environments.
        self.face_backend = "mediapipe" if hasattr(mp, "solutions") else "opencv"
        self.thinking_enabled = self.face_backend == "mediapipe"
        if self.face_backend == "mediapipe":
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            if self.face_detector.empty():
                raise RuntimeError(f"Failed to load OpenCV face detector: {cascade_path}")
            self.face_mesh = None
            self.hands = None
            self.mp_face_mesh = None
            self.mp_hands = None
            print("  MediaPipe solutions API not available; using OpenCV face detector.")
            print("  Thinking gesture detection is disabled in this environment.")

        # Monkey expression images
        self.monkey_images = self._load_monkey_images()

        # Temporal smoothing
        self.match_history = deque(maxlen=8)
        self.current_match = None
        self.match_confidence = 0.0
        self.current_emotion = ""
        self.neutral_streak = 0

        # FPS tracking
        self.last_time = time.time()

    def _load_monkey_images(self):
        image_files = {
            "Thinking": "Monkey_Thinking.jpg",
            "Happy": "Monkey_Happy.jpg",
            "Shocked": "Monkey_Shocked.jpg",
        }
        imgs = {}
        for expr, filename in image_files.items():
            if os.path.exists(filename):
                img = cv2.imread(filename)
                if img is not None:
                    imgs[expr] = img
                    print(f"  Loaded: {filename}")
                else:
                    print(f"  Failed to load: {filename}")
                    imgs[expr] = self._create_placeholder(expr)
            else:
                print(f"  File not found: {filename}")
                imgs[expr] = self._create_placeholder(expr)
        return imgs

    def _create_placeholder(self, expr):
        colors = {"Thinking": (180, 150, 100), "Happy": (100, 200, 100), "Shocked": (200, 100, 200)}
        img = np.ones((300, 300, 3), dtype=np.uint8)
        img[:] = colors.get(expr, (128, 128, 128))
        cv2.putText(img, expr, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def _get_face_bbox(self, landmarks, frame_w, frame_h, padding=0.25):
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w, h = x_max - x_min, y_max - y_min
        pad_x, pad_y = w * padding, h * padding
        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y * 1.3))
        x2 = min(frame_w, int(x_max + pad_x))
        y2 = min(frame_h, int(y_max + pad_y * 0.8))
        return x1, y1, x2, y2

    def _pad_bbox(self, x, y, width, height, frame_w, frame_h, padding=0.20):
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y * 1.2))
        x2 = min(frame_w, int(x + width + pad_x))
        y2 = min(frame_h, int(y + height + pad_y * 0.8))
        return x1, y1, x2, y2

    def _calibrate_expression_decision(self, predicted_class, probs):
        class_name = self.class_names[predicted_class]
        confidence = probs[predicted_class].item()

        if self.label_mode != "expression":
            if confidence < self.confidence_threshold:
                return None, confidence, class_name
            return self.monkey_map[predicted_class], confidence, class_name

        if class_name == "Happy":
            if confidence >= self.happy_threshold:
                return "Happy", confidence, class_name
            return None, confidence, f"{class_name} (low confidence)"

        if class_name == "Shocked":
            if confidence >= self.shocked_threshold:
                return "Shocked", confidence, class_name
            return None, confidence, f"{class_name} (not confident)"

        return None, confidence, class_name

    def _predict_expression(self, face_crop):
        pil_image = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.expression_model(tensor)
            probs = F.softmax(output, dim=1)[0]
            predicted_class = probs.argmax().item()

        monkey_expr, confidence, class_name = self._calibrate_expression_decision(predicted_class, probs)
        self.last_probs = probs.detach().cpu().numpy()
        self.last_prediction = self.class_names[predicted_class]
        self.last_decision = class_name if monkey_expr else "Neutral/no match"
        return monkey_expr, confidence, class_name, probs

    def _detect_finger_near_mouth(self, face_landmarks, hand_landmarks_list):
        if not hand_landmarks_list:
            return False
        mouth = face_landmarks[13]
        mouth_x, mouth_y = mouth.x, mouth.y
        for hand_landmarks in hand_landmarks_list:
            for tip_id in [
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.THUMB_TIP,
            ]:
                tip = hand_landmarks.landmark[tip_id]
                dist = np.sqrt((tip.x - mouth_x) ** 2 + (tip.y - mouth_y) ** 2)
                if dist < 0.10:
                    return True
        return False

    def _update_match_history(self, match, confidence, emotion_name=""):
        if confidence >= self.confidence_threshold and match is not None:
            self.match_history.append((match, confidence, emotion_name))
            self.neutral_streak = 0
        else:
            self.match_history.append(("Neutral", confidence, emotion_name))
            self.neutral_streak += 1

        valid_matches = [entry for entry in self.match_history if entry[0] != "Neutral"]
        if len(valid_matches) >= 2:
            names = [entry[0] for entry in valid_matches]
            most_common = max(set(names), key=names.count)
            if names.count(most_common) >= 2:
                latest = next(entry for entry in reversed(valid_matches) if entry[0] == most_common)
                self.current_match = most_common
                self.match_confidence = latest[1]
                self.current_emotion = latest[2]
                return

        if self.neutral_streak >= 4:
            self.current_match = None
            self.match_confidence = 0.0
            self.current_emotion = ""

    def _clear_match_history(self):
        self.match_history.clear()
        self.current_match = None
        self.match_confidence = 0.0
        self.current_emotion = ""
        self.neutral_streak = 0

    def _open_camera(self, camera_index):
        """Open camera with platform-appropriate backend.

        On Linux: first wakes the camera sensor via v4l2-ctl (required on some
        Chicony integrated cameras that fail VIDIOC_REQBUFS after GPU init),
        then opens with CAP_ANY which auto-selects the working backend."""
        system = platform.system()

        if system == "Windows":
            return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if system == "Darwin":
            return cv2.VideoCapture(camera_index)

        # Linux: wake camera sensor with v4l2-ctl before OpenCV
        # Some integrated cameras (e.g. Chicony) go to sleep during GPU init
        # and fail VIDIOC_REQBUFS when OpenCV tries to allocate V4L2 buffers.
        import subprocess
        for wake_dev in (f"/dev/video{camera_index}",):
            try:
                subprocess.run([
                    "v4l2-ctl", "-d", wake_dev,
                    "--set-fmt-video=width=640,height=480,pixelformat=MJPG",
                    "--stream-mmap", "--stream-count=1", "--stream-to=/dev/null"
                ], capture_output=True, timeout=5)
            except Exception:
                pass

        cap = cv2.VideoCapture(camera_index)
        return cap
    def _draw_overlay(self, frame, expression, confidence, emotion_name):
        h, w = frame.shape[:2]

        # Monkey image in top-right
        if expression and expression in self.monkey_images:
            monkey_img = self.monkey_images[expression]
            mh, mw = monkey_img.shape[:2]
            target_height = int(h * 0.35)
            scale = target_height / mh
            new_w, new_h = int(mw * scale), int(mh * scale)
            monkey_resized = cv2.resize(monkey_img, (new_w, new_h))

            x_offset = w - new_w - 20
            y_offset = 20

            if y_offset + new_h > h:
                new_h = h - y_offset - 20
                new_w = int(new_h * (mw / mh))
                monkey_resized = cv2.resize(monkey_img, (new_w, new_h))

            roi = frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w]
            if roi.shape[:2] == (new_h, new_w):
                blended = cv2.addWeighted(roi, 0.05, monkey_resized, 0.95, 0)
                frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = blended

                colors = {"Thinking": (255, 180, 0), "Happy": (0, 255, 0), "Shocked": (255, 0, 255)}
                color = colors.get(expression, (255, 255, 255))
                cv2.rectangle(frame, (x_offset - 3, y_offset - 3),
                              (x_offset + new_w + 3, y_offset + new_h + 3), color, 4)

        # Match label box
        if expression:
            label = f"MATCHED: {expression}!"
            emotion_text = f"Emotion: {emotion_name}"
            conf_text = f"Confidence: {confidence:.1%}"

            label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
            emotion_w = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            conf_w = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
            box_w = max(label_w, emotion_w, conf_w) + 30
            box_h = 110

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (box_w, box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            colors = {"Thinking": (255, 180, 0), "Happy": (0, 255, 0), "Shocked": (255, 0, 255)}
            color = colors.get(expression, (255, 255, 255))
            cv2.rectangle(frame, (10, 10), (box_w, box_h), color, 3)
            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, emotion_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(frame, conf_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return frame

    def _draw_debug_overlay(self, frame):
        if not self.debug:
            return

        h, w = frame.shape[:2]
        y = max(10, h - 145)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y - 10), (390, h - 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        cv2.putText(frame, f"Raw: {self.last_prediction or '-'} | Use: {self.last_decision or '-'}",
                    (20, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        if self.last_probs is not None:
            for i, (name, prob) in enumerate(zip(self.class_names, self.last_probs)):
                text = f"{name}: {prob:.1%}"
                cv2.putText(frame, text, (20, y + 36 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        if self.last_face_bbox is not None:
            x1, y1, x2, y2 = self.last_face_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

    def _save_debug_frame(self, frame):
        if not self.debug:
            return
        self.debug_frame_index += 1
        filename = os.path.join(self.debug_dir, f"frame_{self.debug_frame_index:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved debug frame: {filename}")

    def _draw_instructions(self, frame):
        instructions = [
            "THINKING: Finger close to mouth (gesture)",
            "HAPPY: Smile / happy face (ML)",
            "SHOCKED: Surprise / angry / sad face (ML)",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (15, 140 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def initialize_camera(self):
        print("Initializing camera...")
        if self.camera_index is not None:
            camera_candidates = [self.camera_index]
        else:
            camera_candidates = range(3)

        # Scan cameras without releasing the working one (avoids V4L2 re-open bug)
        for i in camera_candidates:
            cap = self._open_camera(i)
            if not cap.isOpened():
                cap.release()
                time.sleep(0.3)
                continue
            success, frame = cap.read()
            if success and frame is not None:
                print(f"  Camera {i}: OK ({frame.shape[1]}x{frame.shape[0]})")
                # Use 640x480 during init phase for compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                time.sleep(0.5)
                # Burn-in: drain 10 frames
                for _ in range(10):
                    cap.read()
                    time.sleep(0.05)
                # Attempt target resolution
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                except Exception:
                    pass
                return cap
            cap.release()
            time.sleep(0.3)

        print("No working cameras found!")
        return None
    def run(self):
        cap = self.initialize_camera()
        if cap is None:
            print("Camera initialization failed. Check webcam connection and permissions.")
            return

        print("\n" + "=" * 70)
        print("  MONKEY EXPRESSION MATCHER (ML Edition)")
        print("=" * 70)
        print(f"  Model: ResNet50 trained on FER2013 ({self.label_mode} mode)")
        if self.label_mode == "expression":
            print(f"  Thresholds: Shocked >= {self.shocked_threshold:.2f}, Happy >= {self.happy_threshold:.2f}")
        print("  THINKING: Finger near mouth (gesture rule)")
        print("  HAPPY:    Happy face (ML prediction)")
        print("  SHOCKED:  Angry/Disgust/Fear/Sad/Surprise face (ML prediction)")
        print("-" * 70)
        print("Controls: q=quit  r=reset camera  f=fullscreen  c=clear match  s=save debug frame")
        print("=" * 70 + "\n")

        fullscreen = False
        frame_count = 0
        last_fps_time = time.time()
        fps = 0

        cv2.namedWindow("Monkey Expression Matcher", cv2.WINDOW_NORMAL)

        while True:
            success, frame = cap.read()
            frame_count += 1

            if not success:
                print("Frame capture failed, reconnecting...")
                cap.release()
                time.sleep(1)
                cap = self.initialize_camera()
                if cap is None:
                    break
                continue

            # FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time

            if self.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            finger_near_mouth = False
            monkey_expr = None
            confidence = 0.0
            emotion_name = ""
            self.last_face_bbox = None
            n_faces = 0
            n_hands = 0

            if self.face_backend == "mediapipe":
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_mesh.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                n_faces = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
                n_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark

                        if hand_results.multi_hand_landmarks:
                            finger_near_mouth = self._detect_finger_near_mouth(
                                landmarks, hand_results.multi_hand_landmarks
                            )

                        if finger_near_mouth:
                            monkey_expr = "Thinking"
                            confidence = 0.95
                            emotion_name = "Thinking (gesture)"
                        else:
                            x1, y1, x2, y2 = self._get_face_bbox(landmarks, w, h)
                            if x2 > x1 and y2 > y1:
                                self.last_face_bbox = (x1, y1, x2, y2)
                                face_crop = frame[y1:y2, x1:x2]
                                if face_crop.size > 0:
                                    monkey_expr, confidence, emotion_name, _ = self._predict_expression(face_crop)

                        self._update_match_history(monkey_expr, confidence, emotion_name)
                else:
                    self._update_match_history(None, 0.0, "")
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80),
                )
                n_faces = len(faces)
                if n_faces:
                    x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
                    x1, y1, x2, y2 = self._pad_bbox(x, y, fw, fh, w, h)
                    self.last_face_bbox = (x1, y1, x2, y2)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        monkey_expr, confidence, emotion_name, _ = self._predict_expression(face_crop)
                    self._update_match_history(monkey_expr, confidence, emotion_name)
                else:
                    self._update_match_history(None, 0.0, "")

            # Draw UI
            self._draw_instructions(frame)
            if self.current_match:
                frame = self._draw_overlay(
                    frame,
                    self.current_match,
                    self.match_confidence,
                    self.current_emotion,
                )
            self._draw_debug_overlay(frame)

            # FPS + status
            cv2.putText(frame, f"FPS: {fps}", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = (
                f"Faces: {n_faces} | Hands: {n_hands} | "
                f"Detector: {self.face_backend} | Model: ResNet50/{self.label_mode}"
            )
            cv2.putText(frame, status, (15, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Monkey Expression Matcher", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self._clear_match_history()
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
            elif key == ord("s"):
                self._save_debug_frame(frame)

        cap.release()
        cv2.destroyAllWindows()
        print("\nThanks for playing with the monkeys!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monkey Expression Matcher (ML Edition)")
    parser.add_argument("--model", default="best_model.pth", help="Path to trained .pth checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.50,
                        help="Minimum softmax confidence to accept a 7-class prediction (default mode). Lower = more matches.")
    parser.add_argument("--shocked-threshold", type=float, default=0.80,
                        help="Minimum confidence before showing Shocked for expression checkpoints")
    parser.add_argument("--happy-threshold", type=float, default=0.55,
                        help="Minimum confidence before showing Happy for expression checkpoints")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use. By default the first working camera is selected.")
    parser.add_argument("--debug", action="store_true",
                        help="Show model probabilities, face crop box, and enable saving frames with s.")
    parser.add_argument("--debug-dir", default="debug_frames",
                        help="Directory for frames saved with s in debug mode.")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Do not mirror the camera preview.")
    parser.add_argument("--width", type=int, default=1280,
                        help="Requested camera width after initialization.")
    parser.add_argument("--height", type=int, default=720,
                        help="Requested camera height after initialization.")
    args = parser.parse_args()

    try:
        print("Starting Monkey Expression Matcher (ML Edition)...")
        matcher = MonkeyExpressionMatcher(
            model_path=args.model,
            confidence_threshold=args.confidence_threshold,
            shocked_threshold=args.shocked_threshold,
            happy_threshold=args.happy_threshold,
            camera_index=args.camera,
            debug=args.debug,
            debug_dir=args.debug_dir,
            mirror=not args.no_mirror,
            camera_width=args.width,
            camera_height=args.height,
        )
        matcher.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
