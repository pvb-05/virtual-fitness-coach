"""
Services Module: LLM Agent (Google Gemini API)
Tích hợp Google Gemini API thông qua SDK `google-genai` để phân tích lỗi tư thế kéo dài
và sinh lời khuyên sửa tư thế bằng tiếng Việt ngắn gọn, súc tích.

Nguyên tắc hoạt động:
1. Chỉ gọi LLM khi lỗi duy trì liên tục qua thời gian thực tế (mặc định: >= 1.0 giây).
2. Áp dụng thời gian chờ (Cooldown, mặc định: 10.0 giây) để tránh spam API và tránh làm rối người tập.
3. Không để LLM làm chậm pipeline Computer Vision: chạy tác vụ ngầm qua ThreadPoolExecutor.
4. Cơ chế dự phòng ngoại tuyến (Offline Fallback): nếu thiếu API key, mất mạng hoặc vượt quota,
   hệ thống tự động sử dụng lời khuyên chuyên gia chuẩn mực đã định nghĩa sẵn.
5. Bảo mật: đọc GEMINI_API_KEY từ biến môi trường, không in hoặc lưu key vào log.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Any

logger = logging.getLogger("LLMAgent")

# Các câu hướng dẫn dự phòng chuẩn mực từ chuyên gia thể hình
FALLBACK_ADVICE: Dict[str, str] = {
    "ERR_DEPTH": "Hãy hạ hông sâu hơn một chút để đùi song song với mặt sàn, giữ lưng thẳng và mắt nhìn về phía trước.",
    "ERR_VALGUS": "Hãy chủ động mở rộng đầu gối ra ngoài theo hướng mũi chân khi hạ người để bảo vệ khớp gối.",
    "ERR_OCCLUDED": "Hãy lùi xa camera khoảng 2 mét để hệ thống nhìn rõ toàn thân từ vai đến bàn chân.",
    "ERR_UNKNOWN": "Hãy giữ thăng bằng, siết nhẹ cơ bụng và thực hiện động tác với biên độ kiểm soát."
}


class LLMAgent:
    """Agent quản lý gọi Gemini API và theo dõi lỗi tư thế kéo dài."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        min_error_duration_sec: float = 1.0,
        cooldown_sec: float = 10.0
    ):
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY", "")

        # Kiểm tra key hợp lệ (loại trừ giá trị mẫu hoặc offline)
        if not self.api_key or self.api_key in ("your_gemini_api_key_here", "", "offline"):
            self.api_key = None
            logger.info("Chưa cấu hình GEMINI_API_KEY. LLMAgent sẽ hoạt động ở chế độ Dự phòng Offline.")
        else:
            logger.info("Đã phát hiện GEMINI_API_KEY. Sẵn sàng kết nối Google Gemini API.")

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.min_error_duration_sec = min_error_duration_sec
        self.cooldown_sec = cooldown_sec

        # Bộ theo dõi trạng thái lỗi
        self.current_error_code: Optional[str] = None
        self.error_start_time: Optional[float] = None
        self.last_advice_timestamps: Dict[str, float] = {}  # error_code -> last_trigger_timestamp
        self.is_processing_request = False

        # Lời khuyên mới nhất được sinh ra
        self.latest_advice: Optional[str] = None
        self.latest_advice_timestamp: float = 0.0

        # Luồng thực thi ngầm tránh chặn vòng lặp video
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GeminiWorker")

    def process_error(
        self,
        error_code: Optional[str],
        error_description: str,
        confidence: float,
        rep_count: int,
        timestamp: Optional[float] = None,
        on_advice_ready_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[str]:
        """
        Tiếp nhận mã lỗi từ từng frame, tính toán thời gian tồn tại thực tế và kích hoạt LLM.

        :param error_code: Mã lỗi (ví dụ: 'ERR_DEPTH', 'ERR_VALGUS'), hoặc None nếu tư thế đúng
        :param error_description: Mô tả ngắn gọn về lỗi
        :param confidence: Độ tự tin của mô hình phân loại (0.0 -> 1.0)
        :param rep_count: Số rep hiện tại của buổi tập
        :param timestamp: Thời điểm của frame (mặc định time.time())
        :param on_advice_ready_callback: Callback tùy chọn khi lời khuyên sẵn sàng (advice, error_code)
        :return: Lời khuyên mới nhất (nếu có trong vòng 5 giây qua), hoặc None
        """
        now = timestamp if timestamp is not None else time.time()

        # 1. Nếu không có lỗi (tư thế đúng hoặc đứng nghỉ), reset bộ theo dõi lỗi hiện tại
        if not error_code or error_code in ("None", "ERR_UNKNOWN"):
            self.current_error_code = None
            self.error_start_time = None
            # Trả về lời khuyên gần nhất nếu vừa sinh trong vòng 4 giây
            if now - self.latest_advice_timestamp <= 4.0:
                return self.latest_advice
            return None

        # 2. Theo dõi thời lượng lỗi
        if self.current_error_code != error_code:
            # Lỗi mới xuất hiện: Bắt đầu bấm giờ
            self.current_error_code = error_code
            self.error_start_time = now
            return None

        # Lỗi cũ đang tiếp tục duy trì: Tính thời gian kéo dài thực tế
        error_duration = now - (self.error_start_time or now)

        # 3. Kiểm tra điều kiện kích hoạt:
        # - Đã kéo dài đủ min_error_duration_sec (ví dụ: 1.0s)
        # - Đã qua thời gian Cooldown kể từ lần nhắc lỗi này trước đó
        # - Không có tác vụ Gemini nào khác đang chạy dở
        last_triggered = self.last_advice_timestamps.get(error_code, 0.0)
        is_cooldown_passed = (now - last_triggered) >= self.cooldown_sec

        if error_duration >= self.min_error_duration_sec and is_cooldown_passed and not self.is_processing_request:
            self.last_advice_timestamps[error_code] = now
            self.is_processing_request = True

            logger.info(
                f"Lỗi [{error_code}] đã kéo dài {error_duration:.2f}s (ngưỡng {self.min_error_duration_sec}s). "
                f"Kích hoạt sinh lời khuyên..."
            )

            # Chạy tác vụ sinh lời khuyên trong luồng ngầm
            self.executor.submit(
                self._generate_advice_task,
                error_code,
                error_description,
                confidence,
                rep_count,
                on_advice_ready_callback
            )

        # Trả về lời khuyên gần nhất nếu vừa sinh trong vòng 4 giây
        if now - self.latest_advice_timestamp <= 4.0:
            return self.latest_advice
        return None

    def _generate_advice_task(
        self,
        error_code: str,
        error_description: str,
        confidence: float,
        rep_count: int,
        callback: Optional[Callable[[str, str], None]] = None
    ):
        """Tác vụ ngầm: Gọi Gemini API hoặc lấy lời khuyên dự phòng."""
        advice_text = ""

        try:
            if self.api_key:
                advice_text = self._call_gemini_api(error_code, error_description, confidence, rep_count)
            else:
                advice_text = FALLBACK_ADVICE.get(error_code, FALLBACK_ADVICE["ERR_UNKNOWN"])
                logger.info(f"Sử dụng lời khuyên dự phòng cho [{error_code}]: '{advice_text}'")

        except Exception as e:
            logger.warning(f"Lỗi khi gọi Gemini API ({e}). Kích hoạt lời khuyên dự phòng.")
            advice_text = FALLBACK_ADVICE.get(error_code, FALLBACK_ADVICE["ERR_UNKNOWN"])

        finally:
            self.is_processing_request = False
            self.latest_advice = advice_text
            self.latest_advice_timestamp = time.time()

            if callback and advice_text:
                try:
                    callback(advice_text, error_code)
                except Exception as cb_err:
                    logger.warning(f"Lỗi thực thi callback lời khuyên: {cb_err}")

    def _call_gemini_api(
        self,
        error_code: str,
        error_description: str,
        confidence: float,
        rep_count: int
    ) -> str:
        """Gửi prompt tinh gọn đến Google Gemini API bằng SDK google-genai."""
        from google import genai

        prompt = (
            f"Bạn là huấn luyện viên thể hình cá nhân (Virtual Fitness Coach).\n"
            f"Hệ thống Computer Vision vừa phát hiện người tập mắc lỗi tư thế kéo dài trong bài tập Squat:\n"
            f"- Loại bài tập: Squat (Gánh đùi)\n"
            f"- Mã lỗi phát hiện: {error_code}\n"
            f"- Mô tả lỗi: {error_description}\n"
            f"- Độ tin cậy nhận diện: {int(confidence * 100)}%\n"
            f"- Lần lặp hiện tại (Rep): {rep_count}\n\n"
            f"Yêu cầu:\n"
            f"1. Đưa ra duy nhất 1 hoặc 2 câu hướng dẫn ngắn gọn bằng tiếng Việt (tối đa 30 từ).\n"
            f"2. Tập trung vào ĐÚNG MỘT hành động chỉnh sửa cơ thể ngay lập tức để khắc phục lỗi trên.\n"
            f"3. Không đưa ra chẩn đoán y tế, không dùng từ ngữ nguy hiểm, giọng điệu khích lệ tích cực.\n"
            f"4. Chỉ trả về nội dung lời khuyên, không thêm tiêu đề hay dấu gạch đầu dòng."
        )

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        if response and response.text:
            cleaned = response.text.strip().replace("\n", " ")
            logger.info(f"Gemini API phản hồi thành công: '{cleaned}'")
            return cleaned

        raise ValueError("Phản hồi rỗng từ Gemini API")

    def reset(self):
        """Đặt lại trạng thái bộ theo dõi."""
        self.current_error_code = None
        self.error_start_time = None
        self.last_advice_timestamps.clear()
        self.latest_advice = None
        self.latest_advice_timestamp = 0.0
        self.is_processing_request = False

    def shutdown(self):
        """Dọn dẹp luồng khi dừng ứng dụng."""
        self.executor.shutdown(wait=False)
