"""
Services Module: Text-to-Speech (TTS)
Chuyển đổi văn bản phản hồi và hướng dẫn sửa tư thế thành âm thanh tiếng Việt.
Tính năng:
- Sử dụng gTTS với ngôn ngữ tiếng Việt ('vi')
- Cơ chế băm (SHA256/MD5) lưu đệm (cache) âm thanh vào static/audios/ để tái sử dụng ngay lập tức
- Chạy bất đồng bộ qua ThreadPoolExecutor để không bao giờ làm nghẽn vòng lặp xử lý video
- Chống tấn công path traversal: chỉ lưu và phục vụ file .mp3 an toàn trong thư mục cache
- Cơ chế dự phòng: nếu mất mạng hoặc lỗi dịch vụ, trả về None để giao diện chỉ hiển thị chữ
"""

import os
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

logger = logging.getLogger("TTSService")

DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "audios"))


class TTSService:
    """Dịch vụ tổng hợp giọng nói tiếng Việt với bộ nhớ đệm an toàn."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_workers: int = 2,
        language: str = "vi"
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.language = language
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TTSWorker")
        self.memory_cache: Dict[str, str] = {}  # text_hash -> relative_audio_url

        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(self.cache_dir, exist_ok=True)
        self._warmup_cache_index()

    def _warmup_cache_index(self):
        """Quét các file audio đã có sẵn trong thư mục để nạp vào cache bộ nhớ."""
        try:
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".mp3"):
                    hash_key = fname[:-4]
                    self.memory_cache[hash_key] = f"/static/audios/{fname}"
            logger.info(f"Đã nạp {len(self.memory_cache)} file audio từ cache đĩa.")
        except Exception as e:
            logger.warning(f"Không thể quét thư mục audio cache: {e}")

    def _hash_text(self, text: str) -> str:
        """Tạo mã băm ngắn gọn, an toàn từ chuỗi văn bản."""
        cleaned = text.strip().lower()
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]

    def get_cached_audio_url(self, text: str) -> Optional[str]:
        """Kiểm tra xem câu văn bản này đã có sẵn file âm thanh chưa."""
        if not text or not text.strip():
            return None
        text_hash = self._hash_text(text)
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]

        expected_file = os.path.join(self.cache_dir, f"{text_hash}.mp3")
        if os.path.exists(expected_file):
            url = f"/static/audios/{text_hash}.mp3"
            self.memory_cache[text_hash] = url
            return url
        return None

    def _generate_audio_sync(self, text: str, text_hash: str) -> Optional[str]:
        """Tạo file mp3 đồng bộ bằng gTTS (hàm này được gọi bên trong ThreadPoolExecutor)."""
        target_path = os.path.join(self.cache_dir, f"{text_hash}.mp3")
        
        # Phòng thủ: Kiểm tra nếu tiến trình khác vừa tạo xong
        if os.path.exists(target_path):
            url = f"/static/audios/{text_hash}.mp3"
            self.memory_cache[text_hash] = url
            return url

        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(target_path)

            url = f"/static/audios/{text_hash}.mp3"
            self.memory_cache[text_hash] = url
            logger.info(f"Đã tạo file âm thanh mới: {url} cho câu: '{text[:30]}...'")
            return url

        except Exception as err:
            logger.warning(f"Lỗi tạo âm thanh gTTS cho câu '{text[:30]}...': {err}")
            return None

    def request_audio(self, text: str) -> Optional[str]:
        """
        Yêu cầu đường dẫn âm thanh cho câu văn bản:
        - Nếu đã có trong cache: Trả về URL ngay lập tức (0ms delay).
        - Nếu chưa có: Đẩy vào hàng đợi nền để tạo file và trả về None cho frame hiện tại
          (Frontend sẽ hiển thị text trước, và phát âm thanh ở các frame sau khi file sẵn sàng).
        """
        if not text or not text.strip():
            return None

        # 1. Kiểm tra cache
        cached_url = self.get_cached_audio_url(text)
        if cached_url:
            return cached_url

        # 2. Đẩy tác vụ sinh audio vào luồng ngầm không chặn (Non-blocking)
        text_hash = self._hash_text(text)
        self.executor.submit(self._generate_audio_sync, text, text_hash)

        return None

    def synthesize_blocking_for_test(self, text: str) -> Optional[str]:
        """Hàm hỗ trợ chạy đồng bộ dành riêng cho Unit Test."""
        text_hash = self._hash_text(text)
        return self._generate_audio_sync(text, text_hash)

    def shutdown(self):
        """Dọn dẹp luồng khi đóng ứng dụng."""
        self.executor.shutdown(wait=False)
