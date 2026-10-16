"""
Công cụ thu thập và gán nhãn dữ liệu tư thế Squat từ Webcam thực tế (Giai đoạn 4).
Hỗ trợ người dùng mở rộng dataset bằng dữ liệu người tập thật:
- Chạy: python data/collect_data.py --label correct
- Chạy: python data/collect_data.py --label insufficient_depth
- Chạy: python data/collect_data.py --label knee_valgus
- Nhấn phím SPACE: Lưu mẫu tư thế hiện tại vào data/dataset_squat.csv
- Nhấn phím 'q' hoặc ESC: Lưu và thoát
"""

import os
import sys
import argparse
import csv
import cv2
import numpy as np

# Đảm bảo in ký tự tiếng Việt an toàn trên console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào path để import core module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pose_detector import PoseDetector
from core.biomechanics import extract_squat_features, FEATURE_NAMES

CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset_squat.csv")


def main():
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu tư thế Squat từ Webcam.")
    parser.add_argument(
        "--label",
        type=str,
        default="correct",
        choices=["correct", "insufficient_depth", "knee_valgus"],
        help="Nhãn tư thế cần thu thập (correct | insufficient_depth | knee_valgus)"
    )
    parser.add_argument("--cam", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    target_label = args.label
    print("=" * 65)
    print(f"BỘ THU THẬP DỮ LIỆU SQUAT — ĐANG GÁN NHÃN: [{target_label}]")
    print("Hướng dẫn:")
    print(" - Đứng vào vị trí squat mong muốn.")
    print(" - Nhấn phím SPACE (phím Cách) để lưu 1 mẫu đặc trưng vào CSV.")
    print(" - Nhấn phím 'q' hoặc ESC để kết thúc.")
    print("=" * 65)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở camera index {args.cam}")
        return

    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            result = detector.process_frame(frame)
            annotated = detector.draw_skeleton(frame, result)

            # Bảng hướng dẫn thu thập
            cv2.rectangle(annotated, (15, 15), (420, 110), (15, 23, 42), -1)
            cv2.rectangle(annotated, (15, 15), (420, 110), (51, 65, 85), 1)

            cv2.putText(annotated, f"DANG THU THAP: {target_label.upper()}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"So mau da ghi them: {saved_count}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, "Nhan SPACE: Ghi mau | 'q': Thoat", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow("Thu thap du lieu Squat", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 32:  # Phím SPACE
                if result.detected and result.is_valid:
                    features = extract_squat_features(result.landmarks)
                    row_data = [features[col] for col in FEATURE_NAMES] + [target_label]

                    # Ghi thêm dòng vào CSV
                    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)

                    saved_count += 1
                    print(f" -> [ĐÃ LƯU] Mẫu #{saved_count} với nhãn '{target_label}': Gối={features['knee_angle_left']:.1f}deg, Valgus={features['knee_alignment_left']:.2f}")
                else:
                    print(f" -> [BỎ QUA] Tư thế chưa hợp lệ hoặc bị khuất: {result.error_message}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"[HOÀN THÀNH] Đã thu thập và lưu thành công {saved_count} mẫu mới vào '{CSV_PATH}'.")


if __name__ == "__main__":
    main()
