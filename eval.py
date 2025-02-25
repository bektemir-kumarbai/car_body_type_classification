import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F
from decouple import config
import os
import time

# 🔹 Параметры
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = config("MODEL_PATH")

# 🔹 Загружаем feature extractor
feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 🔹 Преобразования (такие же, как при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 🔹 Получаем классы
class_names = ['Bus', 'CarCarrier', 'Convertible', 'Coupe', 'Hatchback', 'Limousine', 'Minivan', 'PassengerVan', 'Pick-Up', 'SUV', 'Sedan', 'SpecialTransport', 'Tank', 'Truck', 'TruckVan', 'Wagon']


# 🔹 Загружаем модель
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()  # Переключаем в режим оценки
print(f"✅ Модель загружена: {MODEL_PATH}")
print(f"🚗 Классы модели: {class_names}")

# 🔥 Функция для предсказания одного изображения
# 🔥 Функция для предсказания одного изображения
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        probabilities = F.softmax(outputs, dim=1)  # Применяем softmax для вероятностей
        confidence, predicted_class = probabilities.max(1)  # Находим наиболее вероятный класс

    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100  # Переводим в проценты

    print(f"🔮 Предсказанный класс: {predicted_label} (уверенность: {confidence_score:.2f}%)")
while True:
    file_path = input('file path: ')
    start_time = time.time()
    if os.path.exists(file_path):
        predict(file_path)
    else:
        print(f"⚠️ Файл {file_path} не найден! Добавьте изображение для теста.")
    print(f"Время выполнения: {time.time() - start_time} секунд")
