import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os
import platform

class MonkeyExpressionMatcher:
    def __init__(self):
        # Initialize Mediapipe Face Mesh (but don't draw it)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize Mediapipe Hands (but don't draw them)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define monkey expressions
        self.monkey_expressions = {
            "Thinking": {
                "description": "Finger close to mouth",
                "color": (255, 180, 0),  # Orange
                "features": {
                    "finger_near_mouth": True,
                    "mouth_openness": 0.15,
                }
            },
            "Happy": {
                "description": "Mouth open + finger away from face",
                "color": (0, 255, 0),  # Green
                "features": {
                    "finger_away_from_face": True,
                    "mouth_open": True,
                }
            },
            "Shocked": {
                "description": "Mouth open with no hand",
                "color": (255, 0, 255),  # Magenta
                "features": {
                    "mouth_open": True,
                    "no_hand_detected": True,
                }
            }
        }
        
        # Match history for smoothing
        self.match_history = deque(maxlen=8)
        self.current_match = None
        self.match_confidence = 0.0
        self.match_start_time = None
        self.display_duration = 2.5  # seconds
        
        # Load monkey images
        self.monkey_images = self.load_monkey_images()
        
        # For FPS calculation
        self.last_time = time.time()
        
    def load_monkey_images(self):
        """Load the actual monkey images"""
        monkey_imgs = {}
        image_files = {
            "Thinking": "Monkey_Thinking.jpg",
            "Happy": "Monkey_Happy.jpg",
            "Shocked": "Monkey_Shocked.jpg"
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
        """Create placeholder if image not found"""
        colors = {
            "Thinking": (180, 150, 100),
            "Happy": (100, 200, 100),
            "Shocked": (200, 100, 200)
        }
        
        img = np.ones((300, 300, 3), dtype=np.uint8)
        img[:] = colors.get(expr, (128, 128, 128))
        cv2.putText(img, expr, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def get_mouth_aspect_ratio(self, landmarks):
        """Calculate mouth openness"""
        # Upper and lower lip (center)
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        # Mouth corners
        left_corner = landmarks[61]
        right_corner = landmarks[291]
        
        vertical = self.calculate_distance(upper_lip, lower_lip)
        horizontal = self.calculate_distance(left_corner, right_corner)
        
        mar = vertical / horizontal if horizontal > 0 else 0
        return mar
    
    def detect_tongue_out(self, landmarks):
        """Detect if tongue is out (very open mouth with specific shape)"""
        mouth_open = self.get_mouth_aspect_ratio(landmarks)
        return mouth_open > 0.3
    
    def detect_finger_near_mouth(self, face_landmarks, hand_landmarks_list):
        """Detect if finger is near mouth for Thinking expression"""
        if not hand_landmarks_list:
            return False
        
        # Get mouth position
        mouth_center = face_landmarks[13]  # Upper lip center
        mouth_x, mouth_y = mouth_center.x, mouth_center.y
        
        for hand_landmarks in hand_landmarks_list:
            # Check index finger tip
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # Check thumb tip
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculate distances to mouth
            index_distance = np.sqrt((index_tip.x - mouth_x)**2 + (index_tip.y - mouth_y)**2)
            thumb_distance = np.sqrt((thumb_tip.x - mouth_x)**2 + (thumb_tip.y - mouth_y)**2)
            
            # If finger is close to mouth (within 10% of screen distance)
            if index_distance < 0.10 or thumb_distance < 0.10:
                return True
        
        return False
    
    def detect_finger_away_from_face(self, face_landmarks, hand_landmarks_list):
        """Detect if finger is away from face for Happy expression"""
        if not hand_landmarks_list:
            return False
        
        # Get face center
        face_center = face_landmarks[1]  # Nose bridge
        face_x, face_y = face_center.x, face_center.y
        
        for hand_landmarks in hand_landmarks_list:
            # Check index finger tip
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # Check thumb tip
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            
            # Calculate distances to face center
            index_distance = np.sqrt((index_tip.x - face_x)**2 + (index_tip.y - face_y)**2)
            thumb_distance = np.sqrt((thumb_tip.x - face_x)**2 + (thumb_tip.y - face_y)**2)
            
            # If finger is far from face (more than 20% of screen distance)
            if index_distance > 0.20 or thumb_distance > 0.20:
                return True
        
        return False
    
    def extract_features(self, landmarks, hand_results):
        """Extract facial features from landmarks and hand gestures"""
        features = {}
        
        # Mouth features
        mouth_openness = self.get_mouth_aspect_ratio(landmarks)
        features['mouth_openness'] = mouth_openness
        features['mouth_open'] = mouth_openness > 0.20
        features['tongue_out'] = self.detect_tongue_out(landmarks)
        
        # Hand gesture features
        hand_detected = hand_results and hand_results.multi_hand_landmarks
        features['hand_detected'] = hand_detected
        features['no_hand_detected'] = not hand_detected
        
        features['finger_near_mouth'] = False
        features['finger_away_from_face'] = False
        
        if hand_detected:
            features['finger_near_mouth'] = self.detect_finger_near_mouth(
                landmarks, hand_results.multi_hand_landmarks
            )
            features['finger_away_from_face'] = self.detect_finger_away_from_face(
                landmarks, hand_results.multi_hand_landmarks
            )
        
        return features
    
    def match_expression(self, user_features):
        """Match user features to monkey expressions"""
        # Rule-based matching for clear logic
        
        # Rule 1: Thinking - finger near mouth
        if user_features.get('finger_near_mouth', False):
            return "Thinking", 0.9
        
        # Rule 2: Happy - mouth open AND finger away from face
        if (user_features.get('mouth_open', False) and 
            user_features.get('finger_away_from_face', False)):
            return "Happy", 0.9
        
        # Rule 3: Shocked - mouth open AND no hand detected
        if (user_features.get('mouth_open', False) and 
            user_features.get('no_hand_detected', False)):
            return "Shocked", 0.9
        
        # Rule 4: Shocked - tongue out (regardless of hands)
        if user_features.get('tongue_out', False):
            return "Shocked", 0.9
        
        # No match
        return None, 0.0
    
    def update_match_history(self, match, confidence):
        """Update match history with smoothing"""
        threshold = 0.40
        
        if confidence > threshold:
            self.match_history.append(match)
        else:
            self.match_history.append("Neutral")
        
        # Get most common match in history
        if len(self.match_history) > 0:
            valid_matches = [m for m in self.match_history if m != "Neutral"]
            if len(valid_matches) >= 2:
                most_common = max(set(valid_matches), key=valid_matches.count)
                if valid_matches.count(most_common) >= 2:
                    if self.current_match != most_common:
                        self.current_match = most_common
                        self.match_start_time = time.time()
                    self.match_confidence = confidence
                    return
        
        # Only clear if we've been neutral for a while
        neutral_count = list(self.match_history).count("Neutral")
        if neutral_count >= 4:
            self.current_match = None
            self.match_confidence = 0.0
    
    def draw_overlay(self, frame, expression, confidence):
        """Draw expression match overlay on frame"""
        h, w = frame.shape[:2]
        
        # Draw monkey image in corner
        if expression and expression in self.monkey_images:
            monkey_img = self.monkey_images[expression]
            mh, mw = monkey_img.shape[:2]
            
            # Resize monkey image to fit nicely
            target_height = int(h * 0.35)
            scale = target_height / mh
            new_w, new_h = int(mw * scale), int(mh * scale)
            monkey_resized = cv2.resize(monkey_img, (new_w, new_h))
            
            # Position in top-right corner
            x_offset = w - new_w - 20
            y_offset = 20
            
            # Ensure it fits in frame
            if y_offset + new_h > h:
                new_h = h - y_offset - 20
                new_w = int(new_h * (mw / mh))
                monkey_resized = cv2.resize(monkey_img, (new_w, new_h))
            
            # Alpha blending for smooth overlay
            alpha = 0.95
            roi = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
            
            if roi.shape[0] == new_h and roi.shape[1] == new_w:
                blended = cv2.addWeighted(roi, 1-alpha, monkey_resized, alpha, 0)
                frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = blended
                
                # Draw border
                color = self.monkey_expressions[expression]['color']
                cv2.rectangle(frame, (x_offset-3, y_offset-3), 
                             (x_offset+new_w+3, y_offset+new_h+3), color, 4)
        
        # Draw match label
        if expression:
            label = f"MATCHED: {expression}!"
            conf_text = f"Confidence: {confidence:.1%}"
            
            # Background for text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            box_w = max(label_w, conf_w) + 30
            box_h = 85
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (box_w, box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Border
            color = self.monkey_expressions[expression]['color']
            cv2.rectangle(frame, (10, 10), (box_w, box_h), color, 3)
            
            # Draw text
            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (200, 200, 200), 2)
        
        return frame
    
    def draw_status_info(self, frame, finger_near_mouth, finger_away_from_face, mouth_open, no_hand_detected):
        """Draw clean status information without landmarks"""
        info_lines = []
        
        if finger_near_mouth:
            info_lines.append("🤔 THINKING: Finger near mouth")
        
        if finger_away_from_face and mouth_open:
            info_lines.append("😃 HAPPY: Mouth open + finger away")
        elif finger_away_from_face:
            info_lines.append("👉 Finger away from face")
        
        if mouth_open and no_hand_detected:
            info_lines.append("😮 SHOCKED: Mouth open + no hand")
        elif mouth_open:
            info_lines.append("😮 Mouth open")
        
        # Display all info lines
        for i, text in enumerate(info_lines):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 20
            text_y = 80 + (i * 35)
            
            # Background
            cv2.rectangle(frame, (text_x - 10, text_y - 25), 
                         (frame.shape[1] - 10, text_y + 5), (0, 0, 0), -1)
            cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
            
            # Text
            color = (255, 180, 0) if "THINKING" in text else (
                    (0, 255, 0) if "HAPPY" in text else (
                    (255, 0, 255) if "SHOCKED" in text else (200, 200, 200)))
            
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_expression_instructions(self, frame):
        """Draw simple instructions for each expression"""
        instructions = [
            "🤔 THINKING: Finger close to mouth",
            "😃 HAPPY: Mouth open + finger away from face", 
            "😮 SHOCKED: Mouth open + no hand detected"
        ]
        
        y_start = 120
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (15, y_start + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def initialize_camera(self):
        """Initialize camera with robust error handling"""
        print("🔍 Initializing camera...")
        
        # Diagnostic: Check available cameras
        print("Scanning for available cameras...")
        available_cameras = []
        
        for i in range(3):
            if platform.system() == "Windows":
                cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap_test = cv2.VideoCapture(i)
                
            if cap_test.isOpened():
                success, frame = cap_test.read()
                if success and frame is not None:
                    available_cameras.append(i)
                    print(f"✅ Camera {i}: Working (Resolution: {frame.shape[1]}x{frame.shape[0]})")
                else:
                    print(f"⚠️  Camera {i}: Opened but no frame")
                cap_test.release()
            else:
                print(f"❌ Camera {i}: Not available")
            time.sleep(0.5)
        
        if not available_cameras:
            print("❌ No working cameras found!")
            return None
        
        # Try to open the best camera
        camera_index = available_cameras[0]
        print(f"🎥 Attempting to use camera {camera_index}...")
        
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)
        
        # Start with lower resolution for stability
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Wait for camera to initialize
        time.sleep(2)
        
        # Test camera
        test_success = False
        for i in range(10):
            success, frame = cap.read()
            if success and frame is not None:
                test_success = True
                break
            time.sleep(0.1)
        
        if not test_success:
            print("❌ Camera test failed - cannot read frames")
            cap.release()
            return None
        
        print("✅ Camera initialized successfully!")
        
        # Gradually increase resolution
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"📷 Final resolution: 1280x720")
        except:
            print("⚠️  Could not set high resolution, using 640x480")
        
        return cap
    
    def run(self):
        """Main loop to run the expression matcher with clean display"""
        cap = self.initialize_camera()
        
        if cap is None:
            print("""
            ❌ Camera initialization failed! Please check:
            
            1. 🔌 Webcam is connected and not being used by another application
            2. 📷 Webcam drivers are installed and up to date
            3. 🔒 Camera permissions are granted
            4. 💻 Try a different USB port
            5. 🚫 Close other applications that might use the camera (Zoom, Teams, etc.)
            """)
            return
        
        print("\n" + "=" * 70)
        print("🐵  MONKEY EXPRESSION MATCHER - CLEAN VERSION  🐵")
        print("=" * 70)
        print("\n🎯 CLEAN TRIGGERS:")
        print("  🤔 THINKING: Finger close to mouth")
        print("  😃 HAPPY: Mouth open + finger away from face") 
        print("  😮 SHOCKED: Mouth open + no hand detected")
        print("\n" + "=" * 70)
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset camera")
        print("  'f' - Toggle fullscreen")
        print("=" * 70 + "\n")
        
        fullscreen = False
        frame_count = 0
        last_fps_time = time.time()
        fps = 0
        
        # Create window first
        cv2.namedWindow('🐵 Monkey Expression Matcher', cv2.WINDOW_NORMAL)
        
        while True:
            success, frame = cap.read()
            frame_count += 1
            
            if not success:
                print("❌ Failed to capture frame - attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = self.initialize_camera()
                if cap is None:
                    break
                continue
            
            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process both face and hands (but don't draw landmarks)
            face_results = self.face_mesh.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # REMOVED: All landmark drawing code
            
            finger_near_mouth = False
            finger_away_from_face = False
            mouth_open = False
            no_hand_detected = False
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # REMOVED: Face mesh drawing
                    
                    # Extract features (including hand gestures)
                    landmarks = face_landmarks.landmark
                    features = self.extract_features(landmarks, hand_results)
                    
                    # Check for gestures
                    finger_near_mouth = features.get('finger_near_mouth', False)
                    finger_away_from_face = features.get('finger_away_from_face', False)
                    mouth_open = features.get('mouth_open', False)
                    no_hand_detected = features.get('no_hand_detected', False)
                    
                    # Match expression
                    match, confidence = self.match_expression(features)
                    self.update_match_history(match, confidence)
            
            # Draw clean status info
            self.draw_status_info(frame, finger_near_mouth, finger_away_from_face, mouth_open, no_hand_detected)
            
            # Draw expression instructions
            self.draw_expression_instructions(frame)
            
            # Draw overlay
            if self.current_match:
                frame = self.draw_overlay(frame, self.current_match, self.match_confidence)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display status
            status = f"Faces: {len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0} | Hands: {len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0}"
            cv2.putText(frame, status, (15, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('🐵 Monkey Expression Matcher', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting camera...")
                cap.release()
                time.sleep(1)
                cap = self.initialize_camera()
                if cap is None:
                    break
            elif key == ord('f'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('🐵 Monkey Expression Matcher', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('🐵 Monkey Expression Matcher', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Thanks for playing with the monkeys!\n")

if __name__ == "__main__":
    try:
        print("🚀 Starting Monkey Expression Matcher...")
        matcher = MonkeyExpressionMatcher()
        matcher.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleaning up...")
        cv2.destroyAllWindows()