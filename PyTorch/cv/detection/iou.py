import torch 

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
  """
  вычисление IoU:
    boxes_preds - координаты 1-й ограничивающей рамки (bounding box)
    boxes_labels - координаты 2-й ограничивающей рамки
    box_format - формат описания огранич. рамки
  """
  if box_format == "midpoint":
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2 
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

  if box_format == "cornerns":
    # boxes_preds.shape - (N, 4) где N это кол-во рамок 
    # boxes_labels.shape - (N, 4)
    box1_x1 = boxes_preds[..., 0:1] # чтобы получить (N, 1)
    box1_y1 = boxes_preds[..., 1:2] 
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

  x1 = torch.max(box1_x1, box2_x1)
  y1 = torch.max(box1_y1, box2_y1)
  x2 = torch.max(box1_x2, box2_x2)
  y2 = torch.max(box1_y2, box2_y2)

  # .clamp(0) если рамки не пересекаются то пересечение должно быть нулем, а не отриц. знач.
  intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0) # вычисление площади пересечения (общей части двух рамок) 

  box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))  
  box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

  return intersection / (box1_area + box2_area - intersection + 1e-6)
  # IoU = Пересечение / Объединение
  # ?* где объединение это площадь двух прямоугольников не считая общую часть (поэтому минус пересечение)

