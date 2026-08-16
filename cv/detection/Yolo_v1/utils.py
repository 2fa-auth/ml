import torch
from collections import Counter

# ========== IoU ==========
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    else:  # corners
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# ========== mAP ==========
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0
            best_idx = -1

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0


# ========== Преобразование клеток в рамки ==========
def cell_to_bbox(predictions, S=7, B=2, C=20):
    predictions = predictions.reshape(S, S, 30)
    bboxes = []

    for i in range(S):
        for j in range(S):
            # Первая рамка
            conf1 = predictions[i, j, 20]
            if conf1 > 0.4:
                x = (j + predictions[i, j, 21]) / S
                y = (i + predictions[i, j, 22]) / S
                w = predictions[i, j, 23]
                h = predictions[i, j, 24]
                class_probs = predictions[i, j, :20]
                class_pred = torch.argmax(class_probs)
                bboxes.append([class_pred.item(), conf1.item(), x, y, w, h])

            # Вторая рамка
            conf2 = predictions[i, j, 25]
            if conf2 > 0.4:
                x = (j + predictions[i, j, 26]) / S
                y = (i + predictions[i, j, 27]) / S
                w = predictions[i, j, 28]
                h = predictions[i, j, 29]
                class_probs = predictions[i, j, :20]
                class_pred = torch.argmax(class_probs)
                bboxes.append([class_pred.item(), conf2.item(), x, y, w, h])

    return bboxes


def cellboxes_to_boxes(predictions, S=7, B=2, C=20):
    batch_size = predictions.shape[0]
    boxes = []
    for i in range(batch_size):
        boxes.append(cell_to_bbox(predictions[i], S, B, C))
    return boxes


# ========== NMS ==========
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


# ========== get_bboxes ==========
def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for box in nms_boxes:
                all_pred_boxes.append([train_idx] + box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


# ========== load/save checkpoint ==========
def load_checkpoint(filepath, model, optimizer):
    print(f"=> Loading checkpoint {filepath}")
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"=> loaded checkpoint {filepath}")


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint")
    torch.save(state, filename)