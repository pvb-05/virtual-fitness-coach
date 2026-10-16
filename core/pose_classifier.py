"""
Core Module: Pose Classifier
Nạp mô hình Machine Learning KNN đã huấn luyện và thực hiện phân loại tư thế Squat theo thời gian thực.
Đảm bảo:
- Xác thực số lượng và thứ tự 7 đặc trưng nghiêm ngặt theo FEATURE_NAMES
- Dự đoán nhãn tư thế: correct | insufficient_depth | knee_valgus | unknown
- Tính toán độ tin cậy ước lượng (confidence score)
- Trả về 'unknown' / 'uncertain' khi độ tự tin dưới ngưỡng an toàn, không khẳng định sai lỗi
- Xử lý ngoại lệ an toàn khi thiếu file model hoặc đặc trưng bất thường
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import joblib

from core.biomechanics import FEATURE_NAMES, features_to_vector, validate_feature_vector

logger = logging.getLogger("PoseClassifier")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knn_model.pkl")

# Bản đồ ánh xạ nhãn dự đoán sang mã lỗi hệ thống và thông báo tiếng Việt
LABEL_METADATA = {
    "correct": {
        "error_code": None,
        "is_error": False,
        "description_vi": "Tư thế squat chuẩn xác",
        "severity": "normal"
    },
    "insufficient_depth": {
        "error_code": "ERR_DEPTH",
        "is_error": True,
        "description_vi": "Hạ hông chưa đủ độ sâu (chưa đạt đùi song song)",
        "severity": "warning"
    },
    "knee_valgus": {
        "error_code": "ERR_VALGUS",
        "is_error": True,
        "description_vi": "Đầu gối đang bị lệch/chụm vào trong (Knee Valgus)",
        "severity": "danger"
    },
    "unknown": {
        "error_code": "ERR_UNKNOWN",
        "is_error": False,
        "description_vi": "Tư thế chưa xác định hoặc độ tự tin thấp",
        "severity": "info"
    }
}


class ClassificationResult:
    """Kết quả phân loại tư thế trả về cho từng mẫu đặc trưng."""
    def __init__(
        self,
        label: str,
        confidence: float,
        is_reliable: bool,
        error_code: Optional[str] = None,
        description_vi: str = "",
        probabilities: Optional[Dict[str, float]] = None
    ):
        self.label = label
        self.confidence = float(round(confidence, 4))
        self.is_reliable = is_reliable
        self.error_code = error_code
        self.description_vi = description_vi
        self.probabilities = probabilities or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "is_reliable": self.is_reliable,
            "error_code": self.error_code,
            "description_vi": self.description_vi,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()}
        }


class PoseClassifier:
    """Bộ phân loại tư thế sử dụng Pipeline KNN đã huấn luyện."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_confidence_threshold: float = 0.55
    ):
        """
        Khởi tạo bộ phân loại.
        :param model_path: Đường dẫn tới file artifact knn_model.pkl
        :param min_confidence_threshold: Ngưỡng tự tin tối thiểu (nếu thấp hơn sẽ gán nhãn 'unknown')
        """
        self.model_path = os.path.abspath(model_path)
        self.min_confidence_threshold = min_confidence_threshold
        self.pipeline = None
        self.classes: List[str] = []
        self.is_loaded = False
        self.model_metadata: Dict[str, Any] = {}

        self.load_model()

    def load_model(self) -> bool:
        """Nạp model từ đĩa và kiểm tra tính toàn vẹn metadata."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Chưa tìm thấy file model tại: {self.model_path}. Hệ thống sẽ chạy ở chế độ dự phòng.")
            self.is_loaded = False
            return False

        try:
            artifact = joblib.load(self.model_path)
            if not isinstance(artifact, dict) or "pipeline" not in artifact:
                logger.error(f"File model '{self.model_path}' không đúng cấu trúc artifact mong đợi.")
                self.is_loaded = False
                return False

            # Kiểm tra khớp danh sách đặc trưng
            saved_features = artifact.get("feature_names", [])
            if saved_features != FEATURE_NAMES:
                logger.error(
                    f"Không khớp danh sách đặc trưng! Model mong đợi: {saved_features}, "
                    f"Hệ thống hiện tại: {FEATURE_NAMES}"
                )
                self.is_loaded = False
                return False

            self.pipeline = artifact["pipeline"]
            self.classes = artifact.get("classes", list(self.pipeline.classes_))
            self.model_metadata = artifact
            self.is_loaded = True
            logger.info(
                f"Đã nạp thành công mô hình KNN ({len(self.classes)} lớp: {self.classes}) "
                f"được huấn luyện lúc {artifact.get('created_at', 'N/A')}"
            )
            return True

        except Exception as e:
            logger.error(f"Lỗi khi nạp mô hình KNN: {e}")
            self.is_loaded = False
            return False

    def classify(
        self,
        features: Union[Dict[str, float], np.ndarray, List[float]]
    ) -> ClassificationResult:
        """
        Phân loại tư thế từ từ điển hoặc vector đặc trưng.
        :param features: Dict chứa 7 đặc trưng hoặc np.ndarray shape (7,)
        :return: ClassificationResult
        """
        # 1. Kiểm tra trạng thái nạp model
        if not self.is_loaded or self.pipeline is None:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                is_reliable=False,
                error_code="ERR_MODEL_UNAVAILABLE",
                description_vi="Mô hình phân loại chưa được nạp hoặc chưa huấn luyện",
                probabilities={}
            )

        # 2. Chuẩn hóa đầu vào thành numpy array (7,)
        try:
            if isinstance(features, dict):
                vector = features_to_vector(features)
            elif isinstance(features, (list, tuple)):
                vector = np.array(features, dtype=np.float64)
            elif isinstance(features, np.ndarray):
                vector = features.astype(np.float64)
            else:
                raise TypeError(f"Kiểu dữ liệu đặc trưng không hỗ trợ: {type(features)}")

            if not validate_feature_vector(vector):
                raise ValueError(f"Vector đặc trưng không hợp lệ: shape={vector.shape}, có NaN/Inf")

        except Exception as err:
            logger.warning(f"Lỗi tiền xử lý đặc trưng phân loại: {err}")
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                is_reliable=False,
                error_code="ERR_INVALID_FEATURES",
                description_vi="Dữ liệu đặc trưng góc không hợp lệ",
                probabilities={}
            )

        # 3. Dự đoán phân phối xác suất từ KNN Pipeline
        try:
            proba = self.pipeline.predict_proba(vector.reshape(1, -1))[0]
            prob_dict = {cls_name: float(proba[i]) for i, cls_name in enumerate(self.classes)}

            max_idx = int(np.argmax(proba))
            predicted_class = self.classes[max_idx]
            max_prob = float(proba[max_idx])

            # 4. Kiểm tra độ tự tin: Nếu không vượt ngưỡng, trả về unknown
            if max_prob < self.min_confidence_threshold:
                return ClassificationResult(
                    label="unknown",
                    confidence=max_prob,
                    is_reliable=False,
                    error_code="ERR_UNCERTAIN",
                    description_vi="Tư thế chưa đủ căn cứ phân loại chắc chắn",
                    probabilities=prob_dict
                )

            # Lấy metadata nhãn
            meta = LABEL_METADATA.get(predicted_class, LABEL_METADATA["unknown"])

            return ClassificationResult(
                label=predicted_class,
                confidence=max_prob,
                is_reliable=True,
                error_code=meta["error_code"],
                description_vi=meta["description_vi"],
                probabilities=prob_dict
            )

        except Exception as pred_err:
            logger.error(f"Lỗi dự đoán pipeline: {pred_err}")
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                is_reliable=False,
                error_code="ERR_PREDICTION_FAILED",
                description_vi="Lỗi trong quá trình tính toán dự đoán",
                probabilities={}
            )
