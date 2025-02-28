from ultralytics import YOLO
import time
from decouple import config
MODEL_PATH = config("MODEL_PATH")
model = YOLO(MODEL_PATH)

CONF_THRESHOLD = 0.3

while True:
    file_path = input('file path: ')

    start_time = time.time()

    results = model.predict(file_path)

    if results[0].probs is not None and len(results[0].probs) > 0:
        probs = results[0].probs.data.tolist()
        class_id = probs.index(max(probs))
        class_name = model.names[class_id]
        confidence = max(probs)
        if confidence >= CONF_THRESHOLD:
            print(f"Class: {class_name}, Confidence: {confidence:.2f}")
        else:
            print("No confident classification found.")
    print("time: ", time.time() -  start_time)