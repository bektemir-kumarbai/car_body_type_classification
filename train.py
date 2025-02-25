import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from decouple import config as env_reader
import os
from tqdm import tqdm  # 📌 Добавляем прогресс-бар

# Параметры
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Путь к датасету
DATASET_PATH = env_reader("DATASET_PATH")
SAVE_PATH = "models"  # Папка для сохранения моделей

# Загружаем feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Загружаем датасет
train_dataset = ImageFolder(root=DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Получаем список классов
class_names = train_dataset.classes
print(f"\n🚗 Найденные классы ({len(class_names)}): {class_names}\n")

# Загружаем предобученную ViT-модель
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(class_names)
)
model.to(DEVICE)

# Оптимизатор и функция потерь
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Создаём папку для сохранения моделей, если её нет
os.makedirs(SAVE_PATH, exist_ok=True)

# 🔥 Обучение модели с прогресс-баром
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 🔄 Прогресс-бар для батчей
    loop = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{EPOCHS}", unit="batch")

    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 📌 Обновляем описание в прогресс-баре
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    accuracy = 100 * correct / total
    print(f"\n✅ Эпоха [{epoch+1}/{EPOCHS}], Средняя потеря: {total_loss/len(train_loader):.4f}, Точность: {accuracy:.2f}%\n")

    # ✅ Сохраняем модель после каждой эпохи
    model_save_path = os.path.join(SAVE_PATH, f"vit_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Модель сохранена: {model_save_path}\n")

print("🎉 Обучение завершено!")
