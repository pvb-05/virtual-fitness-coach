"""
Virtual Fitness Coach - Main Flask Application
Hệ thống hỗ trợ tập luyện thể hình sử dụng Computer Vision & Google Gemini API.
Giai đoạn 8: Hoàn thiện toàn diện hệ thống Virtual Fitness Coach.
"""

import os
import sys
import time
import json
import base64
import logging
import numpy as np
import cv2
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv

# Import các module cốt lõi
from core.pose_detector import PoseDetector
from core.biomechanics import extract_squat_features
from core.rep_counter import RepCounter, RepCounterConfig, SquatState
from core.pose_classifier import PoseClassifier
from services.llm_agent import LLMAgent
from services.tts_service import TTSService
from database.db_manager import DatabaseManager

# Đảm bảo console Windows hỗ trợ tiếng Việt có dấu
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("VirtualFitnessCoach")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "virtual-fitness-coach-secret-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

sock = Sock(app)

# Khởi tạo các dịch vụ dùng chung
global_detector = PoseDetector(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    visibility_threshold=0.5,
    model_complexity=1
)
global_classifier = PoseClassifier()
global_tts = TTSService()
global_db = DatabaseManager()


# =============================================================================
# ROUTERS GIAO DIỆN
# =============================================================================

@app.route("/")
def index():
    """Trang chủ: Giới thiệu hệ thống, hướng dẫn đặt camera và kiểm tra webcam."""
    return render_template("index.html")


@app.route("/workout")
def workout():
    """Trang phòng tập: Giao diện theo dõi camera, đếm rep và nhận phản hồi AI."""
    return render_template("workout.html")


@app.route("/history")
def history():
    """Trang lịch sử: Dashboard thống kê tổng quan và nhật ký các buổi tập từ SQLite."""
    stats = global_db.get_workout_statistics()
    sessions = global_db.get_all_sessions(limit=50)
    return render_template("history.html", stats=stats, sessions=sessions)


# =============================================================================
# REST APIS QUẢN LÝ DỮ LIỆU BUỔI TẬP (SQLITE CRUD)
# =============================================================================

@app.route("/api/workout/save", methods=["POST"])
def save_workout():
    """API lưu trữ buổi tập cùng danh sách lỗi tư thế vào SQLite."""
    try:
        data = request.get_json() or {}
        session_id = global_db.save_full_session(data)
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Đã lưu buổi tập thành công vào SQLite"
        }), 201
    except Exception as e:
        logger.error(f"Lỗi lưu buổi tập: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """API lấy danh sách các buổi tập gần đây."""
    limit = request.args.get("limit", default=50, type=int)
    sessions = global_db.get_all_sessions(limit=limit)
    return jsonify({"success": True, "sessions": sessions})


@app.route("/api/sessions/<int:session_id>", methods=["GET"])
def get_session_details(session_id):
    """API lấy chi tiết một buổi tập và toàn bộ lỗi liên quan."""
    details = global_db.get_session_details(session_id)
    if not details:
        return jsonify({"success": False, "error": "Không tìm thấy buổi tập"}), 404
    return jsonify({"success": True, "session": details})


@app.route("/api/sessions/<int:session_id>", methods=["DELETE"])
def delete_session(session_id):
    """API xóa một buổi tập khỏi SQLite."""
    deleted = global_db.delete_session(session_id)
    if not deleted:
        return jsonify({"success": False, "error": "Không tìm thấy buổi tập để xóa"}), 404
    return jsonify({"success": True, "message": f"Đã xóa buổi tập #{session_id}"})


@app.route("/api/statistics", methods=["GET"])
def get_statistics():
    """API lấy dữ liệu thống kê tổng hợp cho Dashboard."""
    stats = global_db.get_workout_statistics()
    return jsonify({"success": True, "statistics": stats})


@app.route("/api/health")
def health_check():
    """API kiểm tra trạng thái sức khỏe của Backend và các module AI."""
    import mediapipe as mp
    import sklearn

    gemini_key_configured = bool(
        os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here"
    )

    return jsonify({
        "status": "healthy",
        "stage": "Giai đoạn 8: Hoàn thiện toàn diện hệ thống Virtual Fitness Coach",
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "mediapipe_version": mp.__version__,
        "sklearn_version": sklearn.__version__,
        "detector_ready": global_detector is not None,
        "classifier_ready": global_classifier.is_loaded,
        "classes": global_classifier.classes,
        "database_connected": os.path.exists(global_db.db_path),
        "gemini_api_key_set": gemini_key_configured,
        "websocket_support": True
    })


# =============================================================================
# REALTIME WEBSOCKET STREAM
# =============================================================================

@sock.route("/ws/stream")
def stream_socket(ws):
    """
    WebSocket endpoint phục vụ xử lý video frame thời gian thực:
    Nhận JPEG base64 -> MediaPipe -> Rep Counter -> KNN -> LLM Agent -> TTS
    """
    logger.info("Client đã kết nối tới WebSocket phòng tập (/ws/stream)")

    session_rep_counter = RepCounter(
        RepCounterConfig(
            standing_angle=160.0,
            descending_angle=140.0,
            bottom_angle=95.0,
            ascending_angle=115.0,
            standing_complete_angle=155.0,
            smoothing_factor=0.6
        )
    )
    session_llm = LLMAgent(
        min_error_duration_sec=1.0,
        cooldown_sec=10.0
    )

    try:
        while True:
            message = ws.receive()
            if message is None:
                break

            try:
                payload = json.loads(message)
            except Exception:
                continue

            msg_type = payload.get("type", "")

            # 1. Lệnh điều khiển
            if msg_type == "reset":
                session_rep_counter.reset()
                session_llm.reset()
                ws.send(json.dumps({"type": "reset_ack", "message": "Đã reset bộ đếm rep"}))
                continue

            # 2. Xử lý video frame
            if msg_type == "frame":
                start_proc = time.time()
                image_data = payload.get("image", "")

                if not image_data:
                    continue

                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                try:
                    img_bytes = base64.b64decode(image_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                except Exception as dec_err:
                    logger.warning(f"Lỗi giải mã ảnh frame: {dec_err}")
                    continue

                if frame is None or frame.size == 0:
                    continue

                result = global_detector.process_frame(frame)
                now_ts = time.time()

                if result.detected and result.is_valid:
                    features = extract_squat_features(result.landmarks)

                    rep_res = session_rep_counter.update(
                        features=features,
                        is_valid_pose=True,
                        timestamp=now_ts
                    )

                    clf_res = global_classifier.classify(features)

                    advice = session_llm.process_error(
                        error_code=clf_res.error_code,
                        error_description=clf_res.description_vi,
                        confidence=clf_res.confidence,
                        rep_count=rep_res.rep_count,
                        timestamp=now_ts,
                        on_advice_ready_callback=lambda adv, code: global_tts.request_audio(adv)
                    )

                    audio_url = None
                    feedback_text = clf_res.description_vi
                    if advice:
                        feedback_text = advice
                        audio_url = global_tts.get_cached_audio_url(advice)
                        if not audio_url:
                            global_tts.request_audio(advice)

                    landmarks_summary = [
                        {
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "visibility": round(lm.visibility, 2)
                        }
                        for lm in result.landmarks
                    ]

                    proc_time = round((time.time() - start_proc) * 1000, 1)

                    response = {
                        "type": "pose_result",
                        "pose_detected": True,
                        "is_valid": True,
                        "error_message": "",
                        "rep_count": rep_res.rep_count,
                        "state": rep_res.state.value,
                        "progress_pct": rep_res.progress_pct,
                        "current_knee_angle": rep_res.current_knee_angle,
                        "min_depth_angle": rep_res.min_depth_angle,
                        "rep_just_completed": rep_res.rep_just_completed,
                        "rep_duration": rep_res.rep_duration,
                        "pose_label": clf_res.label,
                        "confidence": clf_res.confidence,
                        "error_code": clf_res.error_code,
                        "feedback_text": feedback_text,
                        "audio_url": audio_url,
                        "landmarks": landmarks_summary,
                        "processing_time_ms": proc_time
                    }

                elif result.detected and not result.is_valid:
                    rep_res = session_rep_counter.update(
                        features=None,
                        is_valid_pose=False,
                        timestamp=now_ts
                    )

                    landmarks_summary = [
                        {
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "visibility": round(lm.visibility, 2)
                        }
                        for lm in result.landmarks
                    ]

                    proc_time = round((time.time() - start_proc) * 1000, 1)

                    response = {
                        "type": "pose_result",
                        "pose_detected": True,
                        "is_valid": False,
                        "error_message": result.error_message,
                        "rep_count": rep_res.rep_count,
                        "state": rep_res.state.value,
                        "progress_pct": rep_res.progress_pct,
                        "current_knee_angle": rep_res.current_knee_angle,
                        "min_depth_angle": rep_res.min_depth_angle,
                        "rep_just_completed": False,
                        "rep_duration": rep_res.rep_duration,
                        "pose_label": "unknown",
                        "confidence": 0.0,
                        "error_code": "ERR_OCCLUDED",
                        "feedback_text": result.error_message,
                        "audio_url": None,
                        "landmarks": landmarks_summary,
                        "processing_time_ms": proc_time
                    }

                else:
                    rep_res = session_rep_counter.update(
                        features=None,
                        is_valid_pose=False,
                        timestamp=now_ts
                    )

                    proc_time = round((time.time() - start_proc) * 1000, 1)

                    response = {
                        "type": "pose_result",
                        "pose_detected": False,
                        "is_valid": False,
                        "error_message": "Chưa phát hiện người",
                        "rep_count": rep_res.rep_count,
                        "state": rep_res.state.value,
                        "progress_pct": 0.0,
                        "current_knee_angle": 180.0,
                        "min_depth_angle": 180.0,
                        "rep_just_completed": False,
                        "rep_duration": 0.0,
                        "pose_label": "unknown",
                        "confidence": 0.0,
                        "error_code": "ERR_NO_PERSON",
                        "feedback_text": "Chưa phát hiện người trong khung hình",
                        "audio_url": None,
                        "landmarks": [],
                        "processing_time_ms": proc_time
                    }

                ws.send(json.dumps(response))

    except Exception as err:
        logger.info(f"WebSocket ngắt kết nối: {err}")
    finally:
        session_llm.shutdown()
        logger.info("Phiên WebSocket kết thúc.")


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Không tìm thấy trang yêu cầu", "status": 404}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Lỗi hệ thống 500: {error}")
    return jsonify({"error": "Lỗi máy chủ nội bộ", "status": 500}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ["true", "1", "yes"]

    logger.info(f"Đang khởi động Virtual Fitness Coach trên http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
