#!/home/client/Documents/fun/py/venv/bin/python3 
import torch 
import torch.nn as nn

# вычисление iou
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

  if box_format == "cornerns":
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
  x2 = torch.max(box1_x2, box2_x2)
  y2 = torch.max(box1_y2, box2_y2)

  intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  

  box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))  
  box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

  return intersection / (box1_area + box2_area - intersection + 1e-6)


class YoloLoss(nn.Module):
  def __init__(self, S=7, B=2, C=20):
    super(YoloLoss, self).__init__()
    self.mse = nn.MSELoss(reduction='sum')
    self.S = S
    self.B = B
    self.C = C
    self.lambda_noobj = 0.5
    self.lambda_coord = 5

  def farward(self, predictions, target):
    predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

    iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
    iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
    iou_maxes, bestbox = torch.max(ious, dim=0)
    exists_box = target[..., 20].unsqueeze(3)

    # ======================== #
    #    FOR BOX COORDINATES   #
    # ======================== #
    box_predictions = exists_box * (
      (
        bestbox * predictions[..., 26:30]
        + (1 - bestbox) * predictions[..., 21:25]
      )
    )

    box_targets = exists_box * target[..., 21:25]

    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
      torch.abs(box_predictions[..., 2:4] + 1e-6)
    )

    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

    # (N, S, S, 4) - > (N*S*S, 4)
    box_loss = self.mse(
      torch.flatten(box_predictions, end_dim=-2),
      torch.flatten(box_targets, end_dim=-2),
    )

    # =================== #
    #   FOR OBJECT LOSS   #
    # =================== #
    pred_box = (
      bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
    )    

    # (N*S*S, 1)
    object_loss = self.mse(
      torch.flatten(exists_box * pred_box),
      torch.flatten(exists_box * target[..., 20:21])
    )

    # ======================== # 
    #   FOR NO OBJECT LLOSS    #
    # ======================== #
    # (N, S, S, 1) - > (N, S*S)
    no_object_loss = self.mse(
      torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
      torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
    )

    no_object_loss += self.mse(
      torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
      torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
    )

    # =================== #
    #   FOR CLASS LOSS    #
    # =================== #

    # (N, S, S, 20) - > (N*S*S, 20)
    class_loss = self.mse(
      torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
      torch.flatten(exists_box * target[..., 20], end_dim=-2),
    ) 

    loss = (
      self.lambda_coord * box_loss 
      + object_loss
      + self.lambda_noobj * no_object_loss
      + class_loss
    )

    return loss
  







