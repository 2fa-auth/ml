"""
код сгененирован с помощью LLM (Gemini)

1. скачивание выборки (25 тысяч фотографий двух классов)
2. разложение классов 'cats' и 'dogs' в 
   обучающую (80%) и тестовую (20%) выборки 
""" 

import os
import shutil
import random
import subprocess
import sys

DATA_URL = "https://github.com/alexeygrigorev/large-datasets/releases/download/dogs-cats/train.zip"
SOURCE_DIR = 'train'          # папка, куда распакуется архив
OUTPUT_DIR = 'dataset'        # папка с готовой структурой train/test
TEST_SIZE = 0.2               # 20% данных для теста
RANDOM_SEED = 42

def download_data():
    if os.path.exists(SOURCE_DIR) and len(os.listdir(SOURCE_DIR)) > 0:
        return
    
    zip_file = "train.zip"
    
    print(f"Скачиваем данные с {DATA_URL}...")
    try:
        subprocess.run(
            ["wget", "-c", DATA_URL, "-O", zip_file],
            check=True
        )
        print("Скачивание завершено!")
    except subprocess.CalledProcessError:
        print("Ошибка при скачивании. Проверьте интернет-соединение.")
        sys.exit(1)
    
    # Распаковываем
    print(f"Распаковываем {zip_file}...")
    try:
        subprocess.run(["unzip", "-q", zip_file], check=True)
        print("Распаковка завершена!")
    except subprocess.CalledProcessError:
        print("Ошибка при распаковке. Установите unzip: sudo apt install unzip")
        sys.exit(1)
    
    os.remove(zip_file)
    print(f"Архив {zip_file} удалён.")

def organize_data():
    print("\nРаскладываем файлы по папкам cats/ и dogs/...")
    
    os.makedirs(os.path.join(SOURCE_DIR, 'cats'), exist_ok=True)
    os.makedirs(os.path.join(SOURCE_DIR, 'dogs'), exist_ok=True)
    
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith('.jpg'):
            continue
            
        src_path = os.path.join(SOURCE_DIR, filename)
        
        if filename.startswith('cat.'):
            shutil.move(src_path, os.path.join(SOURCE_DIR, 'cats', filename))
        elif filename.startswith('dog.'):
            shutil.move(src_path, os.path.join(SOURCE_DIR, 'dogs', filename))
    
    cat_count = len(os.listdir(os.path.join(SOURCE_DIR, 'cats')))
    dog_count = len(os.listdir(os.path.join(SOURCE_DIR, 'dogs')))
    print(f"Готово! Котов: {cat_count}, Собак: {dog_count}")

def split_data():
    for split in ['train', 'test']:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    
    for cls in ['cats', 'dogs']:
        src_dir = os.path.join(SOURCE_DIR, cls)
        files = os.listdir(src_dir)
        
        random.shuffle(files)
        
        split_idx = int(len(files) * (1 - TEST_SIZE))
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        for f in train_files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, 'train', cls, f))
        for f in test_files:
            shutil.copy2(os.path.join(src_dir, f), os.path.join(OUTPUT_DIR, 'test', cls, f))
def cleanup():
    if os.path.exists(SOURCE_DIR):
        shutil.rmtree(SOURCE_DIR)
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    
    download_data()
    organize_data()
    split_data()
    cleanup() 