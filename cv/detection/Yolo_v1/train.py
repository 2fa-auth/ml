import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss
from utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
    save_checkpoint,
)
import tqdm
import os

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"
CHECKPOINT_FILE = "checkpoint.pth.tar"


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            if isinstance(t, transforms.ToTensor):
                img = t(img)
            else:
                img, bboxes = t(img), bboxes
        return img, bboxes


transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm.tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    if len(mean_loss) > 0:
        avg_loss = sum(mean_loss) / len(mean_loss)
        print(f"Mean loss was {avg_loss}")
        return avg_loss
    else:
        print("No data in loader!")
        return 0.0


def main():
    print(f"Using device: {DEVICE}")

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL and os.path.exists(LOAD_MODEL_FILE):
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        print(f"Model loaded from {LOAD_MODEL_FILE}")

    train_dataset = VOCDataset(
        csv_file="data/8examples.csv",
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
    )

    test_dataset = VOCDataset(
        csv_file="data/test.csv",
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        transform=transform,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")

        try:
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
            )

            mean_avg = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP: {mean_avg:.4f}")
        except Exception as e:
            print(f"mAP calculation failed: {e}")
            mean_avg = 0.0

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mAP': mean_avg,
            }, CHECKPOINT_FILE)
            print(f"model saved. loss: {best_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'mAP': mean_avg,
        }, "last_checkpoint.pth.tar")

    print(f"\n{'='*50}")
    print(f"training is end!")
    print(f"best loss: {best_loss:.4f}")
    print(f"best model saved to: {CHECKPOINT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()