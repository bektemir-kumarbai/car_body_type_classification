import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from decouple import config as env_reader
from PIL import Image
import os
import time

# 🔹 Параметры
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = env_reader("DATASET_PATH") # 📂 Путь к датасету
MODEL_PATH = "models/vit_epoch_3.pth"  # 🔥 Файл обученной модели

# 🔹 Загружаем feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# 🔹 Преобразования (такие же, как при обучении)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 🔹 Загружаем тестовый датасет
test_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 🔹 Получаем классы
class_names = test_dataset.classes


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

# 🔥 Функция для оценки на тестовом датасете
def evaluate():
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images).logits
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\n🎯 Точность на тестовом датасете: {accuracy:.2f}%\n")

# 🔥 Функция для предсказания одного изображения
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor).logits
        _, predicted_class = outputs.max(1)

    predicted_label = class_names[predicted_class.item()]
    print(f"🔮 Предсказанный класс: {predicted_label}")
    return predicted_label

# 📌 Запуск оценки
# evaluate()

while True:
    file_path = input('file path: ')
    start_time = time.time()
    if os.path.exists(file_path):
        predict(file_path)
    else:
        print(f"⚠️ Файл {file_path} не найден! Добавьте изображение для теста.")
    print(f"Время выполнения: {time.time() - start_time} секунд")
