"""
Unit Tests cho Services Module (Giai đoạn 6).
Kiểm tra:
- LLMAgent: Cơ chế đếm thời gian lỗi kéo dài (>= 1.0s)
- LLMAgent: Cơ chế Cooldown chống spam
- LLMAgent: Lời khuyên dự phòng Offline khi không có API key
- TTSService: Băm văn bản và lưu đệm âm thanh (Audio Caching)
- TTSService: Tạo file âm thanh tiếng Việt an toàn
"""

import os
import time
import pytest
from services.llm_agent import LLMAgent, FALLBACK_ADVICE
from services.tts_service import TTSService


def test_llm_agent_offline_fallback_content():
    """Kiểm tra từ điển lời khuyên dự phòng có đủ các mã lỗi chính."""
    assert "ERR_DEPTH" in FALLBACK_ADVICE
    assert "ERR_VALGUS" in FALLBACK_ADVICE
    assert "hạ hông" in FALLBACK_ADVICE["ERR_DEPTH"].lower()
    assert "đầu gối" in FALLBACK_ADVICE["ERR_VALGUS"].lower()


def test_llm_agent_error_duration_threshold():
    """Lỗi xuất hiện dưới 1.0 giây sẽ KHÔNG kích hoạt sinh lời khuyên."""
    agent = LLMAgent(api_key=None, min_error_duration_sec=1.0, cooldown_sec=5.0)

    t0 = 1000.0
    # Frame 1: Lỗi vừa xuất hiện tại t0
    adv1 = agent.process_error("ERR_DEPTH", "Hạ hông chưa đủ sâu", 0.95, rep_count=1, timestamp=t0)
    assert adv1 is None

    # Frame 2: Sau 0.5s (chưa đủ 1.0s) -> Vẫn chưa kích hoạt
    adv2 = agent.process_error("ERR_DEPTH", "Hạ hông chưa đủ sâu", 0.95, rep_count=1, timestamp=t0 + 0.5)
    assert adv2 is None

    # Frame 3: Sau 1.2s (đủ >= 1.0s) -> Kích hoạt!
    agent.process_error("ERR_DEPTH", "Hạ hông chưa đủ sâu", 0.95, rep_count=1, timestamp=t0 + 1.2)
    # Chờ luồng ngầm thực thi xong (rất nhanh vì dùng fallback)
    time.sleep(0.1)

    # Frame 4: Lấy kết quả lời khuyên
    adv4 = agent.process_error("ERR_DEPTH", "Hạ hông chưa đủ sâu", 0.95, rep_count=1, timestamp=t0 + 1.3)
    assert adv4 is not None
    assert "hạ hông" in adv4.lower()


def test_llm_agent_cooldown_mechanism():
    """Sau khi vừa sinh lời khuyên, cooldown ngăn chặn gọi liên tục."""
    agent = LLMAgent(api_key=None, min_error_duration_sec=1.0, cooldown_sec=10.0)

    t0 = 2000.0
    # Kích hoạt lần 1
    agent.process_error("ERR_VALGUS", "Chụm gối", 0.95, rep_count=2, timestamp=t0)
    agent.process_error("ERR_VALGUS", "Chụm gối", 0.95, rep_count=2, timestamp=t0 + 1.2)
    time.sleep(0.1)

    assert agent.latest_advice is not None
    first_advice_time = agent.last_advice_timestamps.get("ERR_VALGUS", 0)
    assert first_advice_time == pytest.approx(t0 + 1.2, abs=0.1)

    # Cố tình gọi lại tại t0 + 5.0s (chưa đủ 10s cooldown)
    agent.process_error("ERR_VALGUS", "Chụm gối", 0.95, rep_count=2, timestamp=t0 + 5.0)
    # Timestamp kích hoạt vẫn giữ nguyên lần 1
    assert agent.last_advice_timestamps["ERR_VALGUS"] == first_advice_time


def test_tts_service_hashing_and_path_safety(tmp_path):
    """Kiểm tra hàm băm SHA256 và lưu cache an toàn."""
    tts = TTSService(cache_dir=str(tmp_path))

    text1 = "Tư thế squat chuẩn xác"
    text2 = "Tư thế squat chuẩn xác "  # Thừa khoảng trắng vẫn cùng hash
    hash1 = tts._hash_text(text1)
    hash2 = tts._hash_text(text2)

    assert hash1 == hash2
    assert len(hash1) == 16
    assert "/" not in hash1
    assert "\\" not in hash1


def test_tts_audio_synthesis(tmp_path):
    """Kiểm tra tạo file mp3 tiếng Việt thành công vào thư mục cache."""
    tts = TTSService(cache_dir=str(tmp_path))
    sample_text = "Thử nghiệm âm thanh"

    url = tts.synthesize_blocking_for_test(sample_text)
    assert url is not None
    assert url.startswith("/static/audios/")
    assert url.endswith(".mp3")

    # Kiểm tra file vật lý tồn tại trên đĩa
    file_name = url.replace("/static/audios/", "")
    actual_path = os.path.join(str(tmp_path), file_name)
    assert os.path.exists(actual_path)
    assert os.path.getsize(actual_path) > 0

    # Lần 2 kiểm tra lấy trực tiếp từ cache mà không cần tạo lại
    cached_url = tts.get_cached_audio_url(sample_text)
    assert cached_url == url

    tts.shutdown()
