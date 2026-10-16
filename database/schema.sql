-- Virtual Fitness Coach - Database Schema (SQLite)

-- Bảng 1: Người dùng (Tùy chọn, mặc định hỗ trợ chế độ khách)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bảng 2: Buổi tập luyện (Workout Sessions)
CREATE TABLE IF NOT EXISTS workout_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER DEFAULT 1,
    exercise_type TEXT NOT NULL DEFAULT 'squat',
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER DEFAULT 0,
    total_reps INTEGER DEFAULT 0,
    correct_reps INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Bảng 3: Lịch sử lỗi tư thế phát hiện được (Form Errors)
CREATE TABLE IF NOT EXISTS form_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    error_code TEXT NOT NULL,
    error_description TEXT,
    timestamp_occurred TIMESTAMP NOT NULL,
    duration_seconds REAL DEFAULT 0.0,
    feedback_text TEXT,
    audio_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES workout_sessions(id) ON DELETE CASCADE
);

-- Chỉ mục để tối ưu tốc độ truy vấn lịch sử
CREATE INDEX IF NOT EXISTS idx_sessions_user ON workout_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_errors_session ON form_errors(session_id);
