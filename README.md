# Virtual Fitness Coach 🏋️‍♂️

> **Đồ án Sinh viên Khoa Công nghệ Thông tin**  
> **Hệ thống hỗ trợ tập luyện thể hình thông minh sử dụng Computer Vision và Mô hình Ngôn ngữ Lớn (LLM)**

Hệ thống cho phép người dùng theo dõi và chỉnh sửa tư thế tập luyện (trọng tâm là bài tập **Squat**) thời gian thực thông qua webcam máy tính. Dự án tích hợp:
- **MediaPipe Pose**: Trích xuất 33 điểm mốc giải phẫu học cơ thể (pose landmarks).
- **Bộ tính toán cơ sinh học (Biomechanics)**: Đo đạc chính xác góc gối, góc hông, độ nghiêng thân trên và độ thẳng hàng của đầu gối.
- **State Machine Rep Counter**: Đếm số lần squat hoàn chỉnh độc lập, chống đếm sai do rung lắc hoặc dừng ở đáy squat.
- **K-Nearest Neighbors (KNN)**: Phân loại tư thế Chuẩn (`correct`) và các lỗi sai điển hình (`insufficient_depth`, `knee_valgus`, ...).
- **Google Gemini API (`google-genai`)**: Phân tích lỗi kéo dài và đưa ra lời khuyên sửa tư thế súc tích bằng tiếng Việt.
- **Text-to-Speech (TTS)**: Phát âm thanh nhắc nhở qua trình duyệt mà không làm nghẽn tiến trình xử lý hình ảnh.
- **SQLite Database**: Lưu trữ nhật ký buổi tập, số rep và lịch sử lỗi tư thế.

---

## 📁 Cấu trúc Thư mục Dự án

```text
virtual_fitness_coach/
├── app.py                     # Entry point khởi chạy Flask Server & WebSocket
├── requirements.txt           # Danh sách các thư viện phụ thuộc đã kiểm thử
├── README.md                  # Hướng dẫn chi tiết dự án
├── .env.example               # Mẫu biến môi trường (API Key, Port, DB)
├── .gitignore                 # Cấu hình bỏ qua file rác, venv, sqlite db
├── data/                      # Chứa dataset và script huấn luyện KNN
│   ├── dataset_squat.csv
│   ├── train_knn.py
│   └── knn_model.pkl
├── core/                      # Lõi Computer Vision & Xử lý cơ sinh học
│   ├── __init__.py
│   ├── pose_detector.py       # Nhận diện landmarks qua MediaPipe Pose
│   ├── biomechanics.py        # Tính góc khớp và trích xuất vector đặc trưng
│   ├── rep_counter.py         # State machine đếm rep độc lập
│   └── pose_classifier.py     # Phân loại tư thế bằng KNN pipeline
├── services/                  # Các dịch vụ bên ngoài
│   ├── __init__.py
│   ├── llm_agent.py           # Gọi Gemini API khi lỗi kéo dài
│   └── tts_service.py         # Tạo audio hướng dẫn tiếng Việt
├── database/                  # Quản lý cơ sở dữ liệu SQLite
│   ├── schema.sql             # Cấu trúc bảng users, workout_sessions, form_errors
│   └── db_manager.py          # CRUD thao tác với SQLite
├── static/                    # Tài nguyên tĩnh Frontend
│   ├── css/
│   │   └── tailwind.css       # File CSS tĩnh dự phòng không cần Node.js
│   ├── js/
│   │   └── stream.js          # Quản lý webcam và truyền nhận WebSocket
│   └── audios/                # Bộ nhớ đệm âm thanh TTS được sinh ra
├── templates/                 # Giao diện HTML
│   ├── index.html             # Trang chủ, hướng dẫn camera & chẩn đoán webcam
│   ├── workout.html           # Phòng tập với canvas overlay thời gian thực
│   └── history.html           # Thống kê và lịch sử buổi tập
└── tests/                     # Bộ kiểm thử tự động (pytest)
    ├── test_skeleton.py
    ├── test_biomechanics.py
    ├── test_rep_counter.py
    ├── test_pose_classifier.py
    └── test_database.py
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Trên Windows

### Yêu cầu tiên quyết:
- **Hệ điều hành**: Windows 10 hoặc Windows 11.
- **Python**: Khuyến nghị **Python 3.12** (do MediaPipe 0.10.x tương thích hoàn hảo và ổn định nhất với Python 3.12).
- **Webcam**: Camera tích hợp trên laptop hoặc webcam USB ngoài.

### Bước 1: Khởi tạo môi trường ảo

Mở terminal PowerShell tại thư mục dự án và chạy:

```powershell
# Sử dụng Python 3.12 để tạo venv
py -3.12 -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1
```

> *Mẹo xử lý nếu gặp lỗi Execution Policy trên PowerShell:*  
> Chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi chạy lại lệnh kích hoạt.

### Bước 2: Cài đặt các thư viện phụ thuộc

```powershell
pip install -r requirements.txt
```

### Bước 3: Cấu hình biến môi trường

Sao chép file `.env.example` thành `.env`:

```powershell
Copy-Item .env.example .env
```

Nếu bạn có **Google Gemini API Key**, hãy mở file `.env` và điền vào:
```env
GEMINI_API_KEY=AIzaSy...your_real_key_here
```
*(Nếu chưa có key, hệ thống vẫn hoạt động bình thường với cơ chế phản hồi dự phòng offline).*

### Bước 4: Chạy kiểm thử tự động

```powershell
pytest -v
```

### Bước 5: Khởi động máy chủ Flask

```powershell
python app.py
```

Mở trình duyệt web truy cập vào: **[http://localhost:5000](http://localhost:5000)**.

## Liên hệ

Tác giả: Bac Phan — email: phanvanbac15112005@gmail.com
