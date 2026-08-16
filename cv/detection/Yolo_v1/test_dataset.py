from dataset import VOCDataset # файл сгенерированным LLM
from train import transform

dataset = VOCDataset(
    csv_file="data/8examples.csv",
    img_dir="data/images",
    label_dir="data/labels",
    transform=transform,
)

print(f"Размер датасета: {len(dataset)}")

for i in range(len(dataset)):
    try:
        img, label = dataset[i]
        print(f"✅ Элемент {i}: img shape={img.shape}, label shape={label.shape}")
    except Exception as e:
        print(f"❌ Ошибка в элементе {i}: {e}")
