import copy
import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchstat import stat
from thop import profile
from imutils import perspective
import datetime
from pathlib import Path
import torchvision.transforms as transforms
from dataset import DataAugment, convert
from train import xywhangle_to_rext
import math
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pycocotools.coco import COCO

torch.set_float32_matmul_precision('high')

def draw_mask(image, bboxes, pred_mask, save_path, draw_bboxes, draw_edges):
    pred_mask = (pred_mask >= 0.5).float()
    h, w = pred_mask.shape[1], pred_mask.shape[2]
    pred_mask = pred_mask.cpu().numpy()

    # fuse
    pred_mask_fuse = np.zeros((h, w))
    for i in range(len(pred_mask)):
        pred_mask_fuse += pred_mask[i]
    pred_mask_fuse[pred_mask_fuse > 0] = 1

    show_image = np.zeros((h, w, 3))

    # draw masks
    pred_mask_fuse[pred_mask_fuse > 0] = 255
    show_image[:, :, 0] = show_image[:, :, 0] + pred_mask_fuse  # R

    # draw boxes
    'OBB'
    edge_points_inside_rects = []
    if draw_bboxes:
        for box in bboxes[0]:
            box = box.cpu().numpy().reshape(-1)  # (N,)  x1, y1(0-1), x2, y2(0-1), angle(0-180)
            x_min, y_min, x_max, y_max, angle = box[0], box[1], box[2], box[3], box[4]
            p1 = np.array([x_min, y_min]) * image.shape[2]
            p2 = np.array([x_max, y_min]) * image.shape[2]
            p3 = np.array([x_max, y_max]) * image.shape[2]
            p4 = np.array([x_min, y_max]) * image.shape[2]
            center = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]) * image.shape[2]

            angle = angle / 180 * math.pi
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])

            p1 = np.dot(rotation_matrix, p1 - center) + center
            p2 = np.dot(rotation_matrix, p2 - center) + center
            p3 = np.dot(rotation_matrix, p3 - center) + center
            p4 = np.dot(rotation_matrix, p4 - center) + center

            cv2.line(show_image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)  # G
            cv2.line(show_image, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (0, 255, 0), 2)
            cv2.line(show_image, (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1])), (0, 255, 0), 2)
            cv2.line(show_image, (int(p4[0]), int(p4[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

            'edges'
            if draw_edges:
                edges = cv2.Canny(pred_mask_fuse.astype(np.uint8), threshold1=50, threshold2=150)

                mask = np.zeros_like(edges)

                box_points = np.array([p1, p2, p3, p4]).astype(np.int32)
                cv2.fillPoly(mask, [box_points], 255)

                edge_points_inside_rect = cv2.bitwise_and(edges, mask)
                edge_points_inside_rects.append(edge_points_inside_rect)

    show_image = cv2.cvtColor(show_image.astype("uint8"), cv2.COLOR_RGB2BGR)

    'overlap'
    image = torch.squeeze(image, dim=0).cpu()
    numpy_image = image.permute(1, 2, 0).numpy()
    restored_image = (numpy_image * 255).astype(np.uint8)
    restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("restored_debug.png", restored_image)

    # 2. overlap
    alpha = 0.5
    show_image = cv2.addWeighted(restored_image, 1 - alpha, show_image, alpha, 0)

    'edges'
    if draw_edges:
        for edge_points_inside_rect in edge_points_inside_rects:
            indices = np.where(edge_points_inside_rect > 0)
            row_indices, col_indices = indices
            for ii in range(len(row_indices)):
                cv2.circle(show_image, (int(col_indices[ii]), int(row_indices[ii])), 2, (255, 0, 0), thickness=-1)  # B

    cv2.imwrite(save_path, show_image)

def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert bytes to megabytes

def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    # with fabric.device:
    model = Model(cfg)
    model.setup(fabric.device)

    model.eval()

    'pred'
    image_path = 'datasets/train_images/24.png'
    rbox_path = image_path.replace('train_images', 'train_box_labels').replace('png', 'txt')

    'ge_mask'
    # gt_masks = [np.zeros((1024, 1024), dtype=np.uint8)]  # 随便给一个

    annotation_file=  "datasets/annotations/train_sam.json"
    image_id = int(image_path.split('images/')[1].split('.')[0])
    coco = COCO(annotation_file)
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    gt_masks = []
    for ann in anns:
        mask = coco.annToMask(ann)
        gt_masks.append(mask)


    'image'
    images = cv2.imread(image_path)
    # cam_image = copy.deepcopy(images)

    HEIGHT, WIDTH = images.shape[0], images.shape[1]
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    'OBB'
    bboxes = []
    with open(rbox_path, 'r') as rb:
        for line_rb in rb:
            words = line_rb.split()

            bbox_temp = np.zeros((8))
            for word_i in range(8):
                bbox_temp[word_i] = int(words[word_i])

            #  x1, y1, x2, y2, x3, y3, x4, y4 to x1, y1(0-1), x2, y2(0-1), angle(0-180)
            bbox_temp = convert(bbox_temp, images.shape[0])
            bboxes.append(bbox_temp)

    'transform'
    transform = DataAugment()
    images, gt_masks, bboxes = transform(images, gt_masks, np.array(bboxes))

    # 格式调整
    images = torch.unsqueeze(images, dim=0)
    images = images.to(fabric.device)

    bboxes = np.stack(bboxes, axis=0)
    bboxes = torch.tensor(bboxes).to(fabric.device)
    bboxes = tuple([bboxes])

    gt_masks = np.stack(gt_masks, axis=0)
    gt_masks = torch.tensor(gt_masks).float().to(fabric.device)
    gt_masks = tuple([gt_masks])

    torch.cuda.reset_max_memory_allocated()  # 重置最大显存占用记录

    t1 = time.time()
    print(model)
    pred_masks, _ = model(images, bboxes)  # images: (1, 3, 1024, 1024)  bboxes: tuple(Tensor(1, 5))
    t2 = time.time()
    print('sam inference time: ', (t2 - t1) * 1000, 'ms.')

    max_memory = get_max_memory_allocated()

    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        pred_mask = F.sigmoid(pred_mask)
        pred_mask = torch.clamp(pred_mask, min=0, max=1)

        draw_mask(images, bboxes, pred_mask, save_path='demo_pred_mask.png', draw_bboxes=True, draw_edges=True)
        draw_mask(images, bboxes, gt_mask, save_path='demo_gt_mask.png', draw_bboxes=False, draw_edges=False)


    '*********************************************  grad-cam library only for image  **********************************'
    #
    # target_mask = pred_masks[0].detach().cpu().numpy()
    # target_mask = np.float32(target_mask >= 0.5)
    # target_mask = np.sum(target_mask, axis=0)
    # target_mask[target_mask > 0] = 1
    # target_mask = 255 * np.float32(target_mask)
    #
    # cv2.imwrite('demo_mask.png', target_mask)
    #
    # target_layers = [model.model.image_encoder.blocks[-1].norm1]
    # targets = [SemanticSegmentationTarget(target_mask)]
    #
    # model = SegmentationModelOutputWrapper(model.to(fabric.device))
    # input_tensor = [images, bboxes]
    # with GradCAM(model=model,
    #              target_layers=target_layers,
    #              use_cuda=False) as cam:
    #
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    #
    #     visualization = show_cam_on_image(cam_image, grayscale_cam, use_rgb=True)
    #
    # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    # cv2.imwrite('demo_cam.png', visualization)


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, input_tensor):
        images, bboxes = input_tensor[0], input_tensor[1]
        return self.model(images, bboxes)


class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask)

    def __call__(self, model_output):
        model_output = torch.sum(model_output[0], dim=0)
        self.mask = self.mask.cuda()

        out = (model_output * self.mask).sum()

        return out


if __name__ == "__main__":
    main(cfg)
























