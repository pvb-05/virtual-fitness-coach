"""
Công cụ kiểm tra nhanh Computer Vision, Rep Counter & KNN Classifier với Webcam (Giai đoạn 4).
Tích hợp trọn vẹn:
- MediaPipe Pose: Nhận diện 33 landmarks và vẽ khung xương
- Biomechanics: Trích xuất 7 đặc trưng cơ sinh học chuẩn hóa
- Rep Counter: State Machine đếm số rep độc lập, đo độ sâu
- Pose Classifier (KNN): Phân loại tư thế Chuẩn xác (correct) vs Lỗi (insufficient_depth, knee_valgus)
- Phím 'r': Reset số rep về 0 | Phím 'q' hoặc ESC: Thoát
"""

import sys
import time
import cv2
from core.pose_detector import PoseDetector
from core.biomechanics import extract_squat_features
from core.rep_counter import RepCounter, RepCounterConfig, SquatState
from core.pose_classifier import PoseClassifier


def run_webcam_demo(camera_index: int = 0):
    print("=" * 65)
    print("Virtual Fitness Coach - Kiểm tra Trực quan CV + Rep Counter + KNN")
    print("Đang nạp mô hình KNN và kết nối tới Webcam...")
    print("Phím 'r': Reset số rep về 0 | Phím 'q' hoặc ESC: Thoát")
    print("=" * 65)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở webcam ở index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = PoseDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        visibility_threshold=0.5,
        model_complexity=1
    )

    rep_counter = RepCounter(RepCounterConfig(smoothing_factor=0.6))
    classifier = PoseClassifier()

    prev_time = time.time()
    fps = 0.0
    flash_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)

            # Tính FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # 1. Phát hiện tư thế
            result = detector.process_frame(frame)
            annotated_frame = detector.draw_skeleton(frame, result)

            # 2. Trích xuất đặc trưng
            features = None
            classification_res = None
            if result.detected and result.is_valid:
                features = extract_squat_features(result.landmarks)
                # Chỉ phân loại tư thế khi người dùng đang ở pha hạ người hoặc đáy squat
                classification_res = classifier.classify(features)

            # 3. Cập nhật Rep Counter
            rep_res = rep_counter.update(features=features, is_valid_pose=result.is_valid)
            if rep_res.rep_just_completed:
                flash_frames = 12

            # 4. Giao diện HUD
            h, w = annotated_frame.shape[:2]

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (15, 15), (360, 260), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.85, annotated_frame, 0.15, 0, annotated_frame)
            cv2.rectangle(annotated_frame, (15, 15), (360, 260), (51, 65, 85), 1)

            # Tiêu đề & FPS
            cv2.putText(annotated_frame, f"SQUAT COACH AI - FPS: {fps:.1f}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 185, 129), 2, cv2.LINE_AA)

            # Rep Counter lớn
            rep_box_color = (16, 185, 129) if flash_frames > 0 else (255, 255, 255)
            if flash_frames > 0:
                flash_frames -= 1
                cv2.rectangle(annotated_frame, (0, 0), (w, h), (16, 185, 129), 8)

            cv2.putText(annotated_frame, f"REPS: {rep_res.rep_count}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rep_box_color, 3, cv2.LINE_AA)

            # Trạng thái State Machine
            state_colors = {
                SquatState.STANDING: (100, 200, 100),
                SquatState.DESCENDING: (255, 200, 50),
                SquatState.BOTTOM: (0, 220, 255),
                SquatState.ASCENDING: (255, 150, 0)
            }
            state_color = state_colors.get(rep_res.state, (200, 200, 200))
            cv2.putText(annotated_frame, f"State: {rep_res.state.value}", (190, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)

            # Góc gối & độ sâu
            cv2.putText(annotated_frame, f"Goc goi: {rep_res.current_knee_angle:.1f} deg", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            # Phân loại tư thế bằng KNN
            if classification_res and rep_res.state != SquatState.STANDING:
                if classification_res.label == "correct":
                    label_color = (16, 185, 129)
                    tag = f"[CHUAN] Correct ({int(classification_res.confidence * 100)}%)"
                elif classification_res.label == "insufficient_depth":
                    label_color = (0, 165, 255)
                    tag = f"[LOI] Squat nong ({int(classification_res.confidence * 100)}%)"
                elif classification_res.label == "knee_valgus":
                    label_color = (0, 0, 255)
                    tag = f"[LOI] Chum dau goi ({int(classification_res.confidence * 100)}%)"
                else:
                    label_color = (160, 160, 160)
                    tag = "[T/T] Chua xac dinh"

                cv2.putText(annotated_frame, tag, (30, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, classification_res.description_vi[:35], (30, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(annotated_frame, "Tu the: San sang...", (30, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

            # Tiến độ độ sâu (Progress Bar)
            cv2.putText(annotated_frame, "Tien do do sau:", (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = 30, 205, 300, 15
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int((rep_res.progress_pct / 100.0) * bar_w)
            progress_color = (16, 185, 129) if rep_res.progress_pct >= 95 else (255, 180, 0)
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), progress_color, -1)
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)

            cv2.putText(annotated_frame, "Nhan 'r': Reset | 'q': Thoat", (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.imshow("Virtual Fitness Coach - AI Trainer Demo", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                rep_counter.reset()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[THÀNH CÔNG] Đã giải phóng webcam.")


if __name__ == "__main__":
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam_demo(cam_id)
