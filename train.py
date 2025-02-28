from ultralytics import YOLO
from decouple import config
DATASET_PATH = config("DATASET_PATH")
model = YOLO('yolo11l-cls.pt')

results = model.train(data=DATASET_PATH, epochs=20)