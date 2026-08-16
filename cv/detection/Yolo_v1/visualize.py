import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from model import Yolov1
from dataset import VOCDataset
from train import transform, DEVICE, IMG_DIR, LABEL_DIR
from utils import cellboxes_to_boxes, non_max_suppression

def load_model(model_path="checkpoint.pth.tar"):
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def predict_image(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor.unsqueeze(0).to(DEVICE))
    return predictions

def get_boxes(predictions, S=7, B=2, C=20, threshold=0.4):
    boxes = cellboxes_to_boxes(predictions.cpu(), S, B, C)
    boxes = boxes[0]
    boxes = non_max_suppression(boxes, iou_threshold=0.5, threshold=threshold)
    return boxes

def visualize(image_path, boxes):
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in boxes:
        class_pred, confidence, x, y, w, h = box
        x1 = (x - w / 2) * 448
        y1 = (y - h / 2) * 448
        width = w * 448
        height = h * 448

        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"Class: {int(class_pred)}, Conf: {confidence:.2f}",
            color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7)
        )

    plt.axis('off')
    plt.show()

def main():
    model = load_model("checkpoint.pth.tar")

    dataset = VOCDataset(
        csv_file="data/8examples.csv",
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
    )

    image, label = dataset[0]
    img_path = dataset.annotations.iloc[0, 0]
    full_img_path = f"{IMG_DIR}/{img_path}"

    predictions = predict_image(model, image)
    boxes = get_boxes(predictions, threshold=0.3)

    print(f"Найдено рамок: {len(boxes)}")
    for box in boxes:
        print(f"  Класс: {int(box[0])}, Уверенность: {box[1]:.2f}, x: {box[2]:.2f}, y: {box[3]:.2f}")

    visualize(full_img_path, boxes)

if __name__ == "__main__":
    main()