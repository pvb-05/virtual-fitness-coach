"""
Core Computer Vision & Biomechanics Module for Virtual Fitness Coach.
Bao gồm:
- pose_detector: Nhận diện landmarks bằng MediaPipe Pose
- biomechanics: Tính toán góc khớp & trích xuất vector đặc trưng
- rep_counter: State machine đếm rep squat độc lập
- pose_classifier: Phân loại tư thế đúng/sai bằng mô hình KNN
"""

__version__ = "1.0.0"
