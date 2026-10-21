# Virtual Fitness Coach 🏋️‍♂️

> **Đồ án Sinh viên Khoa Công nghệ Thông tin**  
> **Hệ thống hỗ trợ tập luyện thể hình thông minh sử dụng Computer Vision và Mô hình Ngôn ngữ Lớn (LLM)**  
> *Tác giả: Nhóm Sinh viên CNTT | Pair Programming with AI*

---

## 📌 1. Giới Thiệu Tổng Quan

Trong tập luyện thể hình (Fitness / Calisthenics), việc thực hiện sai kỹ thuật (sai form) không chỉ làm giảm hiệu quả bài tập mà còn tiềm ẩn nguy cơ chấn thương nghiêm trọng cho cột sống thắt lưng và khớp gối (thoát vị đĩa đệm, tổn thương dây chằng chéo). Tuy nhiên, không phải người tập nào cũng có điều kiện thuê Huấn luyện viên cá nhân (PT) kèm riêng.

**Virtual Fitness Coach** là hệ thống trợ lý ảo hỗ trợ tập luyện thể hình theo thời gian thực được thiết kế để chạy trực tiếp trên máy tính cá nhân có webcam thông thường, với trọng tâm hoàn thiện bài tập **Squat**. Hệ thống kết hợp sức mạnh của **Computer Vision** (thị giác máy tính), **Machine Learning** (máy học truyền thống) và **Large Language Models** (mô hình ngôn ngữ lớn) để:
1. Nhận diện chuyển động cơ thể qua webcam trình duyệt thời gian thực mà không cần gắn cảm biến vật lý.
2. Đo đạc các góc cơ sinh học và đếm số rep tự động bằng máy trạng thái hữu hạn (FSM).
3. Phân loại tư thế đúng/sai bằng mô hình K-Nearest Neighbors (KNN).
4. Phân tích lỗi sai kéo dài và tạo lời khuyên chỉnh sửa bằng Google Gemini API (kèm âm thanh nhắc nhở tiếng Việt TTS).
5. Lưu trữ nhật ký buổi tập, lịch sử lỗi và thống kê tiến độ vào cơ sở dữ liệu SQLite.

---

## 🏛️ 2. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo kiến trúc luồng dữ liệu thời gian thực (Realtime Event-Driven Pipeline) kết hợp mô hình Client - Server:

```text
+-----------------------------------------------------------------------------------+
|                              TRÌNH DUYỆT (FRONTEND)                                |
|  Webcam API (getUserMedia)  -->  Canvas Render  -->  HUD & HUD Telemetry          |
|  WebSocket Stream Client    <--  JSON Results   <--  HTML5 Audio Player (TTS)     |
+----------------------------------------|------------------------------------------+
                                         | (JPEG Base64 /ws/stream)
                                         v
+-----------------------------------------------------------------------------------+
|                             MÁY CHỦ FLASK (BACKEND)                               |
|                                                                                   |
|  1. Pose Detection (core/pose_detector.py)                                        |
|     └── MediaPipe Pose (33 Landmarks, kiểm tra visibility, lọc nhiễu tọa độ)     |
|                                                                                   |
|  2. Biomechanical Extraction (core/biomechanics.py)                               |
|     └── 7 Đặc trưng: Góc gối L/R, Góc hông L/R, Độ nghiêng thân, Độ lệch gối      |
|                                                                                   |
|  3. State Machine Rep Counter (core/rep_counter.py)                               |
|     └── [STANDING] <-> [DESCENDING] <-> [BOTTOM] <-> [ASCENDING] (Hysteresis)     |
|                                                                                   |
|  4. Pose Classification (core/pose_classifier.py)                                 |
|     └── KNN Classifier + StandardScaler Pipeline (correct / errors / unknown)     |
|                                                                                   |
|  5. AI Reasoning & Voice (services/llm_agent.py & services/tts_service.py)        |
|     └── Bắt lỗi kéo dài (>= 1.0s) -> Google Gemini API -> gTTS Tiếng Việt         |
|     └── Non-blocking ThreadPoolExecutor + SHA-256 Audio Cache                     |
|                                                                                   |
|  6. Persistence & Analytics (database/db_manager.py)                              |
|     └── SQLite Database (workout_sessions, form_errors)                           |
+-----------------------------------------------------------------------------------+
```

---

## ⚙️ 3. Chi Tiết Các Module Cốt Lõi

### 3.1. Tính Toán Cơ Sinh Học (Biomechanics Module)
- **Công thức tính góc khớp 3 điểm**:
  Góc giữa ba điểm $A$, $B$ (khớp đỉnh), $C$ được tính thông qua tích vô hướng giữa hai vector $\vec{u} = \vec{A} - \vec{B}$ và $\vec{v} = \vec{C} - \vec{B}$:
  $$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}, \quad \theta = \arccos(\text{clip}(\cos(\theta), -1.0, 1.0)) \times \frac{180^\circ}{\pi}$$
  Mã nguồn xử lý triệt để trường hợp vector độ dài 0 và giới hạn giá trị trong khoảng $[-1, 1]$ chống lỗi số học `NaN`.
- **Vector 7 đặc trưng thống nhất**:
  1. `knee_angle_left`: Góc gối trái (hông trái - gối trái - cổ chân trái).
  2. `knee_angle_right`: Góc gối phải.
  3. `hip_angle_left`: Góc hông trái (vai trái - hông trái - gối trái).
  4. `hip_angle_right`: Góc hông phải.
  5. `torso_angle`: Góc nghiêng thân trên so với phương thẳng đứng.
  6. `knee_alignment_left`: Độ lệch gối trái vào trong (Knee Valgus Index).
  7. `knee_alignment_right`: Độ lệch gối phải vào trong.
- **Chỉ số lệch gối bất biến theo tỉ lệ (Scale-Invariant Valgus Index)**:
  $$\text{valgus\_index} = \frac{\Delta x_{\text{knee\_inward}}}{\text{hip\_width}}$$
  Đảm bảo chỉ số không phụ thuộc vào khoảng cách người tập đứng gần hay xa camera.

### 3.2. Máy Trạng Thái Hữu Hạn Đếm Rep (FSM Rep Counter)
Đếm rep hoàn toàn độc lập với mô hình KNN, tuân thủ chu trình chuyển trạng thái nghiêm ngặt:
- `STANDING` (Đứng thẳng, góc gối $\ge 160^\circ$).
- `DESCENDING` (Bắt đầu hạ người, góc gối $< 140^\circ$).
- `BOTTOM` (Đạt độ sâu squat, góc gối $\le 95^\circ$ - đùi song song hoặc sâu hơn mặt sàn).
- `ASCENDING` (Đang đứng lên, vượt ngưỡng hysteresis $\ge 115^\circ$).
- Quay về `STANDING` ($\ge 155^\circ$): **Cộng 1 Rep thành công**.

**Cơ chế chống nhiễu**:
- **EMA Smoothing ($\alpha = 0.6$)**: Làm mượt góc gối, loại bỏ rung giật khung hình.
- **Hysteresis Buffer ($20^\circ$)**: Ngăn nhảy trạng thái liên tục giữa `BOTTOM` và `ASCENDING`.
- **Chống cộng dồn**: Dừng lại lâu ở đáy squat không bị đếm nhiều lần.
- **An toàn khi mất landmarks**: Khi người tập bước ra ngoài hoặc bị che khuất quá 2 giây, FSM tự động chuyển về trạng thái an toàn mà không cộng rep ảo.

### 3.3. Phân Loại Tư Thế Bằng KNN (Pose Classifier)
- **Kiến trúc Pipeline**: `StandardScaler` + `KNeighborsClassifier(n_neighbors=5, weights='distance')`.
- **Nhãn phân loại**:
  - `correct`: Squat đúng kỹ thuật, đủ độ sâu, lưng thẳng, gối thẳng hàng.
  - `insufficient_depth`: Hạ người chưa đủ độ sâu (chưa đạt đùi song song).
  - `knee_valgus`: Đầu gối chụm/lệch vào trong (nguy cơ chấn thương dây chằng chéo trước ACL).
  - `unknown`: Độ tin cậy $< 55\%$ hoặc dữ liệu ngoài biên.
- **Quy tắc khoa học**: Hệ thống không chuyển đổi khoảng cách láng giềng thành "tỷ lệ phần trăm tư thế tuyệt đối", mà sử dụng xác suất dự đoán $P(\text{class})$ và trả về `unknown` khi không đủ cơ sở chắc chắn.

### 3.4. Module AI & Giọng Nói Trợ Lý (LLM & TTS)
- **Tích hợp Google Gemini**: Sử dụng SDK chính thức mới nhất `google-genai` với mô hình mặc định `gemini-2.0-flash`.
- **Cơ chế kích hoạt thông minh**:
  - Chỉ gửi yêu cầu phân tích khi lỗi duy trì liên tục $\ge 1.0\text{s}$ (đo bằng timestamp thực tế của hệ điều hành).
  - Áp dụng thời gian chờ làm nguội (Cooldown = 10s) chống spam API và tránh làm người tập phân tâm.
  - Xử lý bất đồng bộ trong `ThreadPoolExecutor` để chu kỳ WebSocket luôn duy trì tốc độ $\ge 25\text{ FPS}$.
- **Chế độ Offline Fallback**: Khi không có `GEMINI_API_KEY`, mất mạng hoặc hết quota, hệ thống tự động sử dụng từ điển lời khuyên dự phòng chuyên môn được biên soạn sẵn bằng tiếng Việt.
- **Text-to-Speech (TTS)**: Sử dụng `gTTS` tiếng Việt kết hợp cơ chế băm `SHA-256` lưu đệm (cache) trong `static/audios/` giúp phát âm thanh ngay lập tức mà không cần gọi lại mạng.

### 3.5. Cơ Sở Dữ Liệu SQLite (Database & Dashboard)
- Quản lý qua file `database/fitness_coach.db` sử dụng chuẩn `schema.sql`:
  - Bảng `users`: Thông tin người dùng.
  - Bảng `workout_sessions`: Nhật ký buổi tập (thời gian, tổng rep, rep đúng, thời lượng).
  - Bảng `form_errors`: Chi tiết từng lỗi kỹ thuật được phát hiện, độ tự tin, lời khuyên và file âm thanh liên quan.
  - Khóa ngoại `FOREIGN KEY` với `ON DELETE CASCADE` đảm bảo toàn vẹn dữ liệu khi xóa buổi tập.
- Dashboard giao diện đẹp mắt cung cấp các KPI: Tổng số buổi tập, tổng số rep, tỷ lệ chuẩn xác (\%) và biểu đồ phân bố lỗi.

---

## 📁 4. Cấu Trúc Thư Mục Dự Án

```text
virtual-fitness-coach/
├── app.py                      # Máy chủ Flask chính & WebSocket Router (/ws/stream)
├── demo_cv_webcam.py           # Công cụ kiểm tra thị giác máy tính OpenCV độc lập
├── requirements.txt            # Danh sách thư viện phụ thuộc (Python 3.12 compatible)
├── README.md                   # Tài liệu hướng dẫn đồ án chi tiết
├── .env.example                # File mẫu cấu hình biến môi trường
├── .env                        # Biến môi trường thực tế (Gemini Key, Secret Key, Port)
├── .gitignore                  # Cấu hình bỏ qua môi trường ảo, cache, file database
│
├── core/                       # Lõi Computer Vision & Xử lý cơ sinh học
│   ├── __init__.py
│   ├── pose_detector.py        # MediaPipe Pose Detector & Skeleton Visualizer
│   ├── biomechanics.py         # Đo góc khớp 3 điểm, vector tích vô hướng, 7 đặc trưng
│   ├── rep_counter.py          # State Machine đếm rep squat độc lập (EMA + Hysteresis)
│   └── pose_classifier.py      # KNN Pipeline Classifier (StandardScaler + KNeighbors)
│
├── data/                       # Dữ liệu và kịch bản huấn luyện Machine Learning
│   ├── dataset_squat.csv       # Bộ dữ liệu 120 mẫu chuẩn hóa (3 nhãn kỹ thuật)
│   ├── collect_data.py         # Kịch bản thu thập mẫu tư thế thực tế từ webcam
│   ├── train_knn.py            # Huấn luyện KNN, đánh giá metrics & xuất pipeline
│   └── knn_model.pkl           # Mô hình đã huấn luyện kèm siêu dữ liệu
│
├── services/                   # Các dịch vụ bên ngoài & Trí tuệ nhân tạo
│   ├── __init__.py
│   ├── llm_agent.py            # Google Gemini Agent (bắt lỗi kéo dài, cooldown, fallback)
│   └── tts_service.py          # Trình tổng hợp giọng nói tiếng Việt (gTTS + SHA256 Cache)
│
├── database/                   # Quản lý cơ sở dữ liệu SQLite
│   ├── schema.sql              # Lược đồ bảng SQLite (users, sessions, form_errors)
│   ├── db_manager.py           # DAO xử lý CRUD, an toàn giao dịch & phân tích KPI
│   └── fitness_coach.db        # File cơ sở dữ liệu SQLite thực tế
│
├── static/                     # Tài nguyên tĩnh Frontend
│   ├── css/
│   │   └── tailwind.css        # Tailwind CSS dự phòng offline
│   ├── js/
│   │   └── stream.js           # Quản lý Webcam, nén JPEG, WebSocket Ping-Pong HUD
│   └── audios/                 # Bộ nhớ đệm các file âm thanh nhắc nhở (.mp3)
│
├── templates/                  # Giao diện người dùng HTML (Tailwind & FontAwesome)
│   ├── index.html              # Trang chủ: Hướng dẫn đặt camera & chẩn đoán webcam
│   ├── workout.html            # Phòng tập trực tiếp: Video canvas, HUD số liệu, đếm rep
│   └── history.html            # Dashboard lịch sử tập luyện & biểu đồ phân tích lỗi
│
└── tests/                      # Bộ kiểm thử tự động toàn diện (55 tests)
    ├── __init__.py
    ├── test_skeleton.py        # Kiểm thử định tuyến và HTTP endpoints
    ├── test_biomechanics.py    # Kiểm thử toán học vector, góc khớp & tính bất biến tỉ lệ
    ├── test_pose_detector.py   # Kiểm thử MediaPipe wrapper, khung hình rỗng & vẽ xương
    ├── test_rep_counter.py     # Kiểm thử State Machine (chu trình chuẩn, lỗi độ sâu, timeout)
    ├── test_pose_classifier.py # Kiểm thử KNN pipeline, độ tin cậy và phân loại tư thế
    ├── test_stream_pipeline.py # Kiểm thử mã hóa base64 & hợp đồng WebSocket
    ├── test_services.py        # Kiểm thử LLM Agent, Cooldown & TTS Caching
    ├── test_database.py        # Kiểm thử SQLite CRUD, Cascade delete & KPI calculation
    └── test_integration.py     # Kiểm thử End-to-End toàn diện vòng đời buổi tập
```

---

## 🚀 5. Hướng Dẫn Cài Đặt & Triển Khai Trên Windows

### 5.1. Yêu Cầu Tiên Quyết
- **Hệ điều hành**: Windows 10 hoặc Windows 11 (64-bit).
- **Python**: Khuyến nghị **Python 3.12** (bản 3.12.x có thư viện MediaPipe `0.10.14` biên dịch C++ Protobuf ổn định nhất).
- **Phần cứng**: Máy tính có Webcam (tích hợp hoặc USB), CPU đa nhân $\ge 2.0\text{ GHz}$, RAM $\ge 4\text{ GB}$.

### 5.2. Bước 1: Khởi Tạo Môi Trường Ảo
Mở **PowerShell** tại thư mục dự án:

```powershell
# Tạo môi trường ảo với Python 3.12
py -3.12 -m venv venv

# Kích hoạt môi trường ảo
.\venv\Scripts\Activate.ps1
```

> [!TIP]
> *Xử lý lỗi Execution Policy*: Nếu gặp thông báo `cannot be loaded because running scripts is disabled on this system`, hãy chạy lệnh sau một lần:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

### 5.3. Bước 2: Cài Đặt Thư Viện Phụ Thuộc
Chạy lệnh cài đặt các phiên bản đã được khóa tương thích trong `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### 5.4. Bước 3: Cấu Hình Biến Môi Trường (.env)
Sao chép file mẫu `.env.example` thành `.env`:

```powershell
Copy-Item .env.example .env
```

Mở file `.env` bằng Notepad hoặc VS Code:
```env
# Google Gemini API Key (Lấy miễn phí tại: https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here

# Mô hình Gemini khuyên dùng
GEMINI_MODEL=gemini-2.0-flash

# Cấu hình Flask
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=virtual-fitness-coach-secret-key-2026
PORT=5000
HOST=0.0.0.0
```
> [!NOTE]
> Nếu chưa có API key của Gemini, bạn vẫn có thể chạy hệ thống hoàn toàn bình thường nhờ cơ chế **Offline Fallback** tích hợp sẵn.

### 5.5. Bước 4: Chạy Kiểm Thử Tự Động Toàn Diện
Kiểm tra tính toàn vẹn của toàn bộ 55 bài kiểm thử đơn vị và tích hợp:

```powershell
pytest -v
```
*Kết quả mong đợi*: **55 passed** trong khoảng 3-4 giây.

### 5.6. Bước 5: (Tùy chọn) Kiểm Tra Webcam & OpenCV Độc Lập
Trước khi chạy web, bạn có thể kiểm tra trực tiếp camera và bộ khung xương MediaPipe trên một cửa sổ desktop độc lập:

```powershell
python demo_cv_webcam.py
```
*(Bấm phím `q` trên cửa sổ OpenCV để thoát).*

### 5.7. Bước 6: Khởi Động Máy Chủ Web Flask
Khởi chạy ứng dụng:

```powershell
python app.py
```

Sau khi màn hình console hiển thị:
```text
 * Running on http://127.0.0.1:5000
 * WebSocket route ready at ws://localhost:5000/ws/stream
```

Mở trình duyệt (khuyến nghị Google Chrome hoặc Microsoft Edge) và truy cập vào:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🎯 6. Hướng Dẫn Sử Dụng Giao Diện Người Dùng

Hệ thống cung cấp 3 phân hệ trực quan trên thanh điều hướng:

1. **Trang Chủ & Chẩn Đoán (`/`)**:
   - Cung cấp các tiêu chuẩn đặt góc camera (đứng nghiêng $45^\circ - 90^\circ$, cách camera 2-3 mét, thấy rõ từ đầu đến chân).
   - Tích hợp khung **Camera Diagnostic**: Bấm *"Bật Camera Kiểm Tra"* để thử nghiệm độ mượt của webcam và kiểm tra tình trạng kết nối backend ngay lập tức.
2. **Phòng Tập Thời Gian Thực (`/workout`)**:
   - Khung hình Video trực tiếp từ webcam với canvas đồ họa đè (Skeleton Overlay) nối liền 33 điểm khớp cơ thể.
   - **HUD Số Liệu Thời Gian Thực**:
     - Đồng hồ đếm rep khổng lồ nổi bật.
     - Thanh đo độ sâu Squat (Depth Progress Gauge) đổi màu theo thời gian thực (Cam: Đang hạ $\to$ Xanh lá: Đạt đáy $\to$ Xanh dương: Hoàn thành).
     - Góc gối hiện tại và góc sâu nhất đạt được trong rep.
     - Nhãn tư thế KNN (`Chuẩn Xác`, `Chưa Đủ Sâu`, `Chụm Đầu Gối`).
   - Hộp thoại **Phản hồi từ Gemini AI**: Tự động hiển thị lời khuyên và phát giọng nói tiếng Việt khi mắc lỗi kéo dài.
   - Nút bấm *"Kết Thúc & Lưu Buổi Tập"*: Tự động tính toán số rep, thời gian tập và đồng bộ lên cơ sở dữ liệu SQLite.
3. **Dashboard Lịch Sử Tập Luyện (`/history`)**:
   - 4 Thẻ KPI tổng hợp: Tổng số buổi tập, Tổng số rep, Tỷ lệ chuẩn xác (\%), Tổng thời gian tập.
   - Biểu đồ phân bố các lỗi thường gặp trong quá trình tập.
   - Bảng nhật ký chi tiết các buổi tập (ngày giờ, số rep, số lỗi) với nút xem chi tiết từng lỗi kèm lời khuyên và nút xóa buổi tập.

---

## 🔬 7. Tái Huấn Luyện Mô Hình Máy Học (KNN Retraining)

Nếu muốn thu thập thêm dữ liệu của chính mình để huấn luyện lại mô hình KNN:

1. **Thu thập dữ liệu từ webcam**:
   ```powershell
   python data/collect_data.py --samples 50 --label correct
   python data/collect_data.py --samples 50 --label insufficient_depth
   python data/collect_data.py --samples 50 --label knee_valgus
   ```
2. **Huấn luyện lại và xuất mô hình mới**:
   ```powershell
   python data/train_knn.py
   ```
   Kịch bản sẽ tự động chia tập Train/Test (80/20), chuẩn hóa `StandardScaler`, huấn luyện `KNeighborsClassifier`, in ma trận nhầm lẫn (Confusion Matrix) và lưu vào `data/knn_model.pkl`.

---

## 🛠️ 8. Bảng Xử Lý Sự Cố Thường Gặp (Troubleshooting)

| Sự Cố / Hiện Tượng | Nguyên Nhân Thường Gặp | Cách Khắc Phục Nhanh |
| :--- | :--- | :--- |
| **Không bật được webcam trên trình duyệt** | Trình duyệt chưa được cấp quyền Camera hoặc ứng dụng khác (Zoom, Teams) đang chiếm giữ webcam. | 1. Bấm vào biểu tượng ổ khóa bên cạnh thanh địa chỉ URL $\to$ Cho phép Camera.<br>2. Đóng các ứng dụng họp trực tuyến đang mở. |
| **Lỗi `ExecutionPolicy` khi kích hoạt venv** | PowerShell mặc định chặn chạy script cục bộ. | Chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` trước khi gọi script `Activate.ps1`. |
| **Không đếm rep dù đã squat** | 1. Người đứng quá gần khiến camera không thấy chân/cổ chân.<br>2. Chưa hạ người đủ sâu ($\le 95^\circ$). | Lùi ra xa 2-3 mét sao cho toàn bộ cơ thể từ đầu đến gót chân nằm trọn trong khung hình; hạ hông sâu đến khi đùi song song mặt sàn. |
| **Lỗi `404 Not Found` từ Gemini API** | Tên model bị thay đổi hoặc API key không hợp lệ. | File `.env` cấu hình `GEMINI_MODEL=gemini-2.0-flash`. Nếu không có key, hệ thống tự kích hoạt Offline Fallback không gây lỗi. |
| **Lỗi font chữ tiếng Việt trên PowerShell** | Bảng mã console Windows mặc định là `cp1252` hoặc `cp936`. | Các file Python trong dự án đều đã cấu hình `sys.stdout.reconfigure(encoding="utf-8")`. Bạn cũng có thể gõ `chcp 65001` trên PowerShell. |

---

## ⚖️ 9. Tuyên Bố Giới Hạn & Hướng Phát Triển

### Giới Hạn Kỹ Thuật (Known Limitations)
- **Độ sâu 2D vs 3D**: Tọa độ trích xuất từ webcam 2D thông thường thông qua MediaPipe là phép ước lượng thị giác máy tính, không thể thay thế cho các hệ thống phòng lab đo lường quang học 3D chuyên nghiệp (Vicon) hay cảm biến lực đa trục.
- **Góc đặt camera**: Hệ thống hoạt động tối ưu nhất khi camera đặt ngang tầm hông, nghiêng góc khoảng $45^\circ - 70^\circ$ so với mặt trước người tập.
- **Ánh sáng**: Môi trường tập thiếu sáng hoặc người tập mặc quần áo trùng màu với hậu cảnh có thể làm giảm độ tự tin của landmark gối và cổ chân.

### Hướng Phát Triển Tương Lai (Roadmap)
- [ ] Bổ sung các bài tập thể hình phổ biến khác: **Push-up (Chống đẩy)**, **Plank**, **Lunge**, **Bicep Curl**.
- [ ] Tích hợp mô hình học sâu chuỗi thời gian (LSTM / ST-GCN / Pose-Transformer) để đánh giá nhịp điệu và tốc độ của toàn bộ rep (Tempo).
- [ ] Thêm tính năng đăng nhập nhiều tài khoản người dùng và biểu đồ tiến bộ cá nhân theo tuần/tháng.
- [ ] Hỗ trợ camera điện thoại di động thông qua WebRTC hoặc giao thức PWA.

---

## 📜 10. Giấy Phép & Lời Cảm Ơn
- Dự án được xây dựng phục vụ mục đích học tập và nghiên cứu trong khuôn khổ đồ án sinh viên ngành Công nghệ Thông tin.
- Thư viện mã nguồn mở sử dụng: OpenCV, Google MediaPipe, Scikit-learn, Flask, Flask-Sock, Tailwind CSS, FontAwesome.
