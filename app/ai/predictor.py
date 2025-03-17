import time
from ultralytics import YOLO
import torch

class CarBodyTypePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_path)
        print(f"model loaded. Device : {self.device}")
    def predict(self, image):
        start_time = time.time()
        
        try:
            results = self.model.predict(image)
            
            if results[0].probs is not None and len(results[0].probs) > 0:
                probs = results[0].probs.data.tolist()
                class_id = probs.index(max(probs))
                class_name = self.model.names[class_id]
                confidence = max(probs)
                predicted_label = class_name
                confidence_score = confidence
            else:
                predicted_label = None
                confidence_score = None
                
            processing_time = time.time() - start_time
            print(f"Time : {processing_time:.4f} сек")
            
            return {
                'car_type_body': predicted_label,
                'car_type_body_score': confidence_score
            }
            
        except Exception as ex:
            print(f"error : {ex}")
            return {
                'car_type_body': None,
                'car_type_body_score': None
            }
