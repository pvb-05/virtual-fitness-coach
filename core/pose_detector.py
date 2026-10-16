"""
Core Module: Pose Detector
Sử dụng MediaPipe Pose để phát hiện 33 điểm mốc cơ thể (landmarks) từ webcam/video frame.
Kiểm tra độ tin cậy (visibility/presence), xử lý trường hợp bị khuất hoặc ngoài khung hình,
và cung cấp tiện ích vẽ khung xương trực quan.
"""

import logging
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger("PoseDetector")


class LandmarkPoint(NamedTuple):
    """Điểm mốc chuẩn hóa (x, y thuộc [0, 1] theo tỷ lệ khung hình, z là chiều sâu tương đối)."""
    x: float
    y: float
    z: float
    visibility: float


class PoseDetectionResult:
    """Kết quả phát hiện tư thế trả về cho từng frame."""
    def __init__(
        self,
        detected: bool,
        is_valid: bool,
        landmarks: Optional[List[LandmarkPoint]] = None,
        raw_landmarks: Any = None,
        error_message: str = "",
        frame_width: int = 0,
        frame_height: int = 0
    ):
        self.detected = detected
        self.is_valid = is_valid
        self.landmarks = landmarks or []
        self.raw_landmarks = raw_landmarks
        self.error_message = error_message
        self.frame_width = frame_width
        self.frame_height = frame_height

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển phục vụ serialize sang JSON qua WebSocket."""
        return {
            "detected": self.detected,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "landmark_count": len(self.landmarks),
            "keypoints": [
                {
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 3)
                }
                for lm in self.landmarks
            ]
        }


class PoseDetector:
    """Bộ nhận diện tư thế người bằng MediaPipe Pose."""

    # Chỉ số các landmarks thiết yếu cho bài tập Squat theo chuẩn MediaPipe Pose
    ESSENTIAL_LANDMARKS = {
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
    }

    # Các cặp kết nối xương để vẽ khung xương
    SKELETON_CONNECTIONS = [
        (11, 12),  # Vai trái - Vai phải
        (11, 23),  # Vai trái - Hông trái
        (12, 24),  # Vai phải - Hông phải
        (23, 24),  # Hông trái - Hông phải
        (23, 25),  # Hông trái - Gối trái
        (24, 26),  # Hông phải - Gối phải
        (25, 27),  # Gối trái - Cổ chân trái
        (26, 28),  # Gối phải - Cổ chân phải
        (27, 29),  # Cổ chân trái - Gót trái
        (28, 30),  # Cổ chân phải - Gót phải
        (29, 31),  # Gót trái - Mũi chân trái
        (30, 32),  # Gót phải - Mũi chân phải
        (11, 13),  # Vai trái - Cùi chỏ trái
        (13, 15),  # Cùi chỏ trái - Cổ tay trái
        (12, 14),  # Vai phải - Cùi chỏ phải
        (14, 16),  # Cùi chỏ phải - Cổ tay phải
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_threshold: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Khởi tạo MediaPipe Pose.
        :param min_detection_confidence: Ngưỡng tự tin phát hiện ban đầu (0.0 -> 1.0)
        :param min_tracking_confidence: Ngưỡng tự tin bám vết theo thời gian (0.0 -> 1.0)
        :param visibility_threshold: Ngưỡng visibility tối thiểu để một khớp được coi là hợp lệ
        :param model_complexity: 0: Lite (nhanh), 1: Full (cân bằng, mặc định), 2: Heavy (chính xác hơn nhưng nặng)
        """
        self.visibility_threshold = visibility_threshold
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info(
            f"Đã khởi tạo MediaPipe Pose (complexity={model_complexity}, "
            f"detect_conf={min_detection_confidence}, track_conf={min_tracking_confidence})"
        )

    def process_frame(self, frame: np.ndarray) -> PoseDetectionResult:
        """
        Xử lý 1 khung hình BGR từ webcam và trả về danh sách landmarks đã kiểm định.
        :param frame: Khung hình chuẩn OpenCV định dạng BGR (H, W, C)
        :return: PoseDetectionResult
        """
        if frame is None or frame.size == 0:
            return PoseDetectionResult(
                detected=False,
                is_valid=False,
                error_message="Khung hình rỗng hoặc không hợp lệ"
            )

        h, w = frame.shape[:2]

        # Chuyển đổi BGR sang RGB theo yêu cầu của MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Thực thi mô hình phát hiện tư thế
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return PoseDetectionResult(
                detected=False,
                is_valid=False,
                error_message="Không phát hiện người trong khung hình",
                frame_width=w,
                frame_height=h
            )

        # Trích xuất 33 landmarks chuẩn hóa
        extracted_landmarks: List[LandmarkPoint] = []
        for lm in results.pose_landmarks.landmark:
            extracted_landmarks.append(
                LandmarkPoint(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility)
                )
            )

        # Kiểm tra độ tin cậy của các khớp thiết yếu cho bài tập Squat
        missing_joints = []
        for name, idx in self.ESSENTIAL_LANDMARKS.items():
            if idx >= len(extracted_landmarks):
                missing_joints.append(name)
            elif extracted_landmarks[idx].visibility < self.visibility_threshold:
                missing_joints.append(name)

        if missing_joints:
            readable_names = ", ".join(missing_joints[:3])
            err_msg = (
                f"Không thấy rõ toàn thân (khuất {readable_names}). "
                "Vui lòng lùi xa camera khoảng 2-2.5m để thấy rõ từ vai đến bàn chân."
            )
            return PoseDetectionResult(
                detected=True,
                is_valid=False,
                landmarks=extracted_landmarks,
                raw_landmarks=results.pose_landmarks,
                error_message=err_msg,
                frame_width=w,
                frame_height=h
            )

        return PoseDetectionResult(
            detected=True,
            is_valid=True,
            landmarks=extracted_landmarks,
            raw_landmarks=results.pose_landmarks,
            error_message="",
            frame_width=w,
            frame_height=h
        )

    def draw_skeleton(
        self,
        image: np.ndarray,
        result: PoseDetectionResult,
        color_valid: Tuple[int, int, int] = (16, 185, 129),    # Xanh Emerald BGR: (129, 185, 16)
        color_invalid: Tuple[int, int, int] = (68, 68, 239)   # Đỏ BGR: (68, 68, 239)
    ) -> np.ndarray:
        """
        Vẽ khung xương và các khớp nối trực tiếp lên ảnh.
        :param image: Ảnh BGR để vẽ đè lên
        :param result: Kết quả từ hàm process_frame
        :param color_valid: Màu khi tư thế hợp lệ (RGB)
        :param color_invalid: Màu cảnh báo khi tư thế bị khuất hoặc lỗi (RGB)
        :return: Ảnh sau khi đã vẽ overlay
        """
        if not result.detected or not result.landmarks:
            return image

        annotated_image = image.copy()
        h, w = image.shape[:2]

        # Chọn màu sắc theo trạng thái hợp lệ
        line_color = (129, 185, 16) if result.is_valid else (68, 68, 239)
        joint_color = (255, 255, 255)

        # 1. Vẽ các đường nối xương
        for start_idx, end_idx in self.SKELETON_CONNECTIONS:
            if start_idx < len(result.landmarks) and end_idx < len(result.landmarks):
                p1 = result.landmarks[start_idx]
                p2 = result.landmarks[end_idx]

                if p1.visibility >= self.visibility_threshold and p2.visibility >= self.visibility_threshold:
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(annotated_image, pt1, pt2, line_color, thickness=3, lineType=cv2.LINE_AA)

        # 2. Vẽ các điểm khớp tròn
        for idx, lm in enumerate(result.landmarks):
            # Chỉ vẽ các khớp chính phần thân và chân để tránh rối mắt
            if idx in self.ESSENTIAL_LANDMARKS.values() or idx in (13, 14, 15, 16, 0):
                if lm.visibility >= self.visibility_threshold:
                    center = (int(lm.x * w), int(lm.y * h))
                    # Vòng tròn phát sáng bên ngoài
                    cv2.circle(annotated_image, center, radius=6, color=line_color, thickness=-1, lineType=cv2.LINE_AA)
                    # Chấm trắng trung tâm
                    cv2.circle(annotated_image, center, radius=3, color=joint_color, thickness=-1, lineType=cv2.LINE_AA)

        # 3. Vẽ thông báo lỗi nếu không hợp lệ
        if not result.is_valid and result.error_message:
            cv2.putText(
                annotated_image,
                result.error_message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        return annotated_image

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        if hasattr(self, "pose") and self.pose:
            self.pose.close()
            logger.info("Đã đóng phiên làm việc MediaPipe Pose.")
