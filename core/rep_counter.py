"""
Core Module: Rep Counter (State Machine)
Bộ đếm số lần lặp (repetitions/reps) cho bài tập Squat sử dụng máy trạng thái hữu hạn (Finite State Machine).
Hoạt động hoàn toàn độc lập với mô hình Machine Learning KNN.

Các trạng thái chính của chu trình Squat:
- STANDING: Người tập đứng thẳng ở vị trí chuẩn bị (góc gối >= 160 độ).
- DESCENDING: Người tập đang hạ thấp trọng tâm người (góc gối giảm xuống dưới 140 độ).
- BOTTOM: Đạt độ sâu của squat theo cấu hình (góc gối <= 95 độ).
- ASCENDING: Người tập đang đạp chân đứng dậy (góc gối vượt lên trên 115 độ).

Tính năng:
- Ngưỡng góc cấu hình linh hoạt thông qua RepCounterConfig.
- Cơ chế trễ (Hysteresis) và làm mịn góc (Exponential Moving Average) để chống rung lắc (jitter).
- Xử lý mất dấu landmarks an toàn, không cộng rep sai khi camera mất tín hiệu.
- Ghi nhận độ sâu tối đa (min knee angle) và thời gian thực hiện từng rep.
"""

from enum import Enum
import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


class SquatState(str, Enum):
    """Các trạng thái của máy trạng thái Squat."""
    STANDING = "STANDING"          # Đứng thẳng
    DESCENDING = "DESCENDING"      # Đang hạ người
    BOTTOM = "BOTTOM"              # Đáy squat (độ sâu hợp lệ)
    ASCENDING = "ASCENDING"        # Đang đứng lên


@dataclass
class RepCounterConfig:
    """Cấu hình các ngưỡng góc và tham số chống rung cho State Machine."""
    standing_angle: float = 160.0          # Góc gối coi là đứng thẳng chuẩn bị
    descending_angle: float = 140.0        # Góc gối kích hoạt trạng thái hạ người
    bottom_angle: float = 95.0             # Ngưỡng góc gối đạt đáy squat chuẩn
    ascending_angle: float = 115.0         # Ngưỡng bắt đầu pha đẩy người lên (Hysteresis)
    standing_complete_angle: float = 155.0 # Ngưỡng hoàn thành rep khi trở về tư thế đứng
    smoothing_factor: float = 0.5          # Hệ số làm mịn EMA (0.0: rất mượt, 1.0: không làm mịn)
    max_lost_frames_reset: int = 30        # Số frame mất landmarks liên tiếp trước khi reset về STANDING


class RepCounterResult:
    """Kết quả trả về sau mỗi lần cập nhật frame."""
    def __init__(
        self,
        rep_count: int,
        state: SquatState,
        progress_pct: float,
        current_knee_angle: float,
        rep_just_completed: bool = False,
        rep_duration: float = 0.0,
        min_depth_angle: float = 180.0,
        feedback_event: Optional[str] = None
    ):
        self.rep_count = rep_count
        self.state = state
        self.progress_pct = progress_pct
        self.current_knee_angle = current_knee_angle
        self.rep_just_completed = rep_just_completed
        self.rep_duration = rep_duration
        self.min_depth_angle = min_depth_angle
        self.feedback_event = feedback_event

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rep_count": self.rep_count,
            "state": self.state.value,
            "progress_pct": round(self.progress_pct, 1),
            "current_knee_angle": round(self.current_knee_angle, 1),
            "rep_just_completed": self.rep_just_completed,
            "rep_duration": round(self.rep_duration, 2),
            "min_depth_angle": round(self.min_depth_angle, 1),
            "feedback_event": self.feedback_event
        }


class RepCounter:
    """
    Máy trạng thái đếm số lần squat độc lập.
    Đảm bảo 1 rep chỉ được cộng khi người tập hoàn thành đầy đủ chu trình:
    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING.
    """

    def __init__(self, config: Optional[RepCounterConfig] = None):
        self.config = config or RepCounterConfig()
        self.state = SquatState.STANDING
        self.rep_count = 0
        
        # Các biến theo dõi chu trình hiện tại
        self.has_reached_bottom = False
        self.rep_start_time: Optional[float] = None
        self.min_knee_angle_in_rep = 180.0
        self.last_rep_duration = 0.0
        
        # Bộ lọc làm mịn góc đầu vào
        self.smoothed_knee_angle: Optional[float] = None
        
        # Bộ đếm xử lý frame mất landmarks
        self.consecutive_invalid_frames = 0

    def reset(self):
        """Đặt lại toàn bộ trạng thái và số đếm về ban đầu."""
        self.state = SquatState.STANDING
        self.rep_count = 0
        self.has_reached_bottom = False
        self.rep_start_time = None
        self.min_knee_angle_in_rep = 180.0
        self.last_rep_duration = 0.0
        self.smoothed_knee_angle = None
        self.consecutive_invalid_frames = 0

    def _smooth_angle(self, raw_angle: float) -> float:
        """Làm mịn góc bằng Exponential Moving Average (EMA) để lọc nhiễu."""
        if self.smoothed_knee_angle is None:
            self.smoothed_knee_angle = raw_angle
        else:
            alpha = self.config.smoothing_factor
            self.smoothed_knee_angle = alpha * raw_angle + (1.0 - alpha) * self.smoothed_knee_angle
        return self.smoothed_knee_angle

    def _calculate_progress(self, angle: float) -> float:
        """
        Tính % tiến độ hoàn thành một lần squat dựa trên góc gối (0% khi đứng, 100% khi đạt đáy).
        Hỗ trợ hiển thị thanh tiến độ (progress bar) trên giao diện.
        """
        standing = self.config.standing_angle
        bottom = self.config.bottom_angle

        if angle >= standing:
            return 0.0
        elif angle <= bottom:
            return 100.0
        else:
            progress = ((standing - angle) / (standing - bottom)) * 100.0
            return max(0.0, min(100.0, progress))

    def update(
        self,
        features: Optional[Dict[str, float]],
        is_valid_pose: bool = True,
        timestamp: Optional[float] = None
    ) -> RepCounterResult:
        """
        Cập nhật máy trạng thái dựa trên các đặc trưng cơ sinh học của frame mới.
        
        :param features: Dictionary chứa ít nhất knee_angle_left và knee_angle_right
        :param is_valid_pose: Trạng thái landmarks có hợp lệ hay không từ PoseDetector
        :param timestamp: Thời điểm của frame (mặc định lấy time.time())
        :return: RepCounterResult
        """
        now = timestamp if timestamp is not None else time.time()
        rep_just_completed = False
        feedback_event = None

        # 1. Xử lý trường hợp mất landmarks hoặc tư thế không hợp lệ
        if not is_valid_pose or not features:
            self.consecutive_invalid_frames += 1
            if self.consecutive_invalid_frames >= self.config.max_lost_frames_reset:
                # Nếu mất tín hiệu quá lâu giữa chừng, hủy chu trình dở dang để tránh đếm sai
                if self.state != SquatState.STANDING:
                    self.state = SquatState.STANDING
                    self.has_reached_bottom = False
                    self.rep_start_time = None
                    self.min_knee_angle_in_rep = 180.0

            angle_to_report = self.smoothed_knee_angle or 180.0
            return RepCounterResult(
                rep_count=self.rep_count,
                state=self.state,
                progress_pct=self._calculate_progress(angle_to_report),
                current_knee_angle=angle_to_report,
                rep_just_completed=False,
                rep_duration=self.last_rep_duration,
                min_depth_angle=self.min_knee_angle_in_rep,
                feedback_event="LOST_LANDMARKS"
            )

        # Đã nhận lại frame hợp lệ
        self.consecutive_invalid_frames = 0

        # 2. Lấy góc gối đại diện (trung bình 2 gối hoặc gối hợp lệ)
        knee_left = features.get("knee_angle_left", 180.0)
        knee_right = features.get("knee_angle_right", 180.0)
        raw_knee_angle = (knee_left + knee_right) / 2.0
        knee_angle = self._smooth_angle(raw_knee_angle)

        # Cập nhật độ sâu thấp nhất đạt được trong rep hiện tại
        if self.state != SquatState.STANDING:
            if knee_angle < self.min_knee_angle_in_rep:
                self.min_knee_angle_in_rep = knee_angle

        # 3. Chuyển đổi trạng thái (State Transitions)
        cfg = self.config

        if self.state == SquatState.STANDING:
            # Điều kiện chuyển: Bắt đầu gập gối hạ người
            if knee_angle < cfg.descending_angle:
                self.state = SquatState.DESCENDING
                self.rep_start_time = now
                self.has_reached_bottom = False
                self.min_knee_angle_in_rep = knee_angle

        elif self.state == SquatState.DESCENDING:
            # Điều kiện 1: Đạt độ sâu đáy squat hợp lệ
            if knee_angle <= cfg.bottom_angle:
                self.state = SquatState.BOTTOM
                self.has_reached_bottom = True
            # Điều kiện 2: Người tập bỏ cuộc nửa chừng, đứng thẳng dậy mà chưa chạm đáy
            elif knee_angle >= cfg.standing_complete_angle:
                self.state = SquatState.STANDING
                feedback_event = "INCOMPLETE_DEPTH"  # Chưa đủ độ sâu
                self.has_reached_bottom = False
                self.rep_start_time = None
                self.min_knee_angle_in_rep = 180.0

        elif self.state == SquatState.BOTTOM:
            # Điều kiện chuyển: Bắt đầu đẩy người đứng lên (có hysteresis >= ascending_angle)
            # Người tập có giữ nguyên ở đáy squat bao lâu thì vẫn ở trạng thái BOTTOM, KHÔNG cộng rep
            if knee_angle >= cfg.ascending_angle:
                self.state = SquatState.ASCENDING

        elif self.state == SquatState.ASCENDING:
            # Điều kiện 1: Hoàn thành trở về vị trí đứng thẳng
            if knee_angle >= cfg.standing_complete_angle:
                if self.has_reached_bottom:
                    self.rep_count += 1
                    rep_just_completed = True
                    feedback_event = "REP_COMPLETED"
                    if self.rep_start_time is not None:
                        self.last_rep_duration = now - self.rep_start_time

                # Reset cho lần lặp tiếp theo
                self.state = SquatState.STANDING
                self.has_reached_bottom = False
                self.rep_start_time = None
                self.min_knee_angle_in_rep = 180.0

            # Điều kiện 2: Đang đứng lên lại thụt xuống đáy lần nữa
            elif knee_angle <= cfg.bottom_angle:
                self.state = SquatState.BOTTOM

        progress = self._calculate_progress(knee_angle)

        return RepCounterResult(
            rep_count=self.rep_count,
            state=self.state,
            progress_pct=progress,
            current_knee_angle=knee_angle,
            rep_just_completed=rep_just_completed,
            rep_duration=self.last_rep_duration,
            min_depth_angle=self.min_knee_angle_in_rep,
            feedback_event=feedback_event
        )
