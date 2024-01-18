import os
import time

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from student_config import student_cfg
from teacher_config import teacher_cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
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
from scipy.ndimage import convolve1d
from pathlib import Path
import math

from losses import DiceLoss, FocalLoss, KLDLoss, CELoss, calc_iou_t, MSELoss

torch.set_float32_matmul_precision('high')

def xywhangle_to_rext(x, y, w, h, angle):
    rect_xy = []
    rect_wh = []
    rect = []
    rect_xy.append(x)
    rect_xy.append(y)
    rect_wh.append(w)
    rect_wh.append(h)
    rect_xy = tuple(rect_xy)
    rect_wh = tuple(rect_wh)
    rect.append(rect_xy)
    rect.append(rect_wh)
    rect.append(angle)
    rect = tuple(rect)

    return rect

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    mask = mask.cpu().numpy()
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    box = box[0].cpu().numpy().reshape(-1) / 3
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def show_prediction(image, masks, boxes):
    plt.figure()
    image = image.cpu().numpy()
    plt.imshow(image)
    show_box(boxes, plt.gca())
    show_mask(masks, plt.gca())
    plt.axis('off')
    plt.show()

def draw_mask(image, bboxes, pred_mask, gt_mask, epoch, iter, save_path='runs/val/masks', show_image=None):
    pred_mask = (pred_mask >= 0.5).float()
    h, w = pred_mask.shape[1], pred_mask.shape[2]
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()

    # fuse
    # a, b, c, d = np.max(pred_mask), np.min(pred_mask), np.max(gt_mask), np.min(gt_mask)  # 1, 0, 1, 0
    pred_mask_fuse, gt_mask_fuse = np.zeros((h, w)), np.zeros((h, w))
    for i in range(len(pred_mask)):
        pred_mask_fuse += pred_mask[i]
    for i in range(len(gt_mask)):
        gt_mask_fuse += gt_mask[i]
    pred_mask_fuse[pred_mask_fuse > 0] = 1
    gt_mask_fuse[gt_mask_fuse > 0] = 1

    if show_image == None:
        show_image = np.zeros((h, w, 3))

    # draw masks
    gt_mask_fuse[gt_mask_fuse > 0] = 255
    show_image[:, :, 2] = show_image[:, :, 2] + gt_mask_fuse  # B
    pred_mask_fuse[pred_mask_fuse > 0] = 255
    show_image[:, :, 0] = show_image[:, :, 0] + pred_mask_fuse  # R  (B + R = Pure)

    # draw boxes
    'OBB'
    for box in bboxes[0]:
        box = box.cpu().numpy().reshape(-1) 
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

    save_path = save_path + '_epoch' + str(epoch) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + str(iter) + '.png'
    show_image = cv2.cvtColor(show_image.astype("uint8"), cv2.COLOR_RGB2BGR)

    'overlap'
    # 1. restore from tensor to image
    image = torch.squeeze(image, dim=0).cpu()
    numpy_image = image.permute(1, 2, 0).numpy()  
    restored_image = (numpy_image * 255).astype(np.uint8) 
    restored_image = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("restored_debug.png", restored_image)

    # 2. overlap
    alpha = 0.6  
    show_image = cv2.addWeighted(restored_image, 1 - alpha, show_image, alpha, 0)

    cv2.imwrite(save_path, show_image)
    # print('')

def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, bboxes, gt_masks = data

            num_images = images.size(0)

            t1 = time.time()
            pred_masks, _ = model(images, bboxes)
            t2 = time.time()

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = F.sigmoid(pred_mask)
                pred_mask = torch.clamp(pred_mask, min=0, max=1)

                # draw masks, boxes
                draw_mask(images, bboxes, pred_mask, gt_mask, epoch, iter)

                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")

                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)

    inference_time = (t2 - t1) * 1000  # ms
    # print('model inference time: ', inference_time, 'ms.')
    with open("runs/val/log.txt", 'a') as f: 
        f.write('Val: [' + str(epoch) + '] - ')  
        f.write('Mean IoU: [' + str(torch.round(ious.avg, decimals=4)) + '] ')
        f.write('Inference Time: [' + str(round(inference_time, 3)) + 'ms]')
        f.write('\n') 
        f.flush()  
    f.close()

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {student_cfg.out_dir}")
    state_dict = model.model.state_dict()
    if fabric.global_rank == 0:
        torch.save(state_dict, os.path.join(student_cfg.out_dir, f"epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    student_model: Model,
    teacher_model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    # mask_loss = KLDLoss()
    mask_loss = CELoss()
    embedding_loss = MSELoss()

    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()

        mask_t_losses = AverageMeter()  # mask decoder from teacher

        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            if epoch > 1 and epoch % cfg.eval_interval == 0 and not validated: 
            # if epoch % cfg.eval_interval == 0 and not validated:
                validate(fabric, student_model, val_dataloader, epoch)
                validated = True

            data_time.update(time.time() - end)
            images, bboxes, gt_masks = data
            batch_size = images.size(0)

            'inference'
            student_pred_masks, student_iou_predictions, student_sparse_embeddings = student_model(images, bboxes, KD_train_mode=True)
            teacher_pred_masks, teacher_iou_predictions, teacher_sparse_embeddings = teacher_model(images, bboxes, KD_train_mode=True)

            num_masks = sum(len(pred_mask) for pred_mask in student_pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)

            loss_mask_t = torch.tensor(0., device=fabric.device)  # mask decoder from teacher
            loss_iou_t = torch.tensor(0., device=fabric.device)  # mask decoder from teacher
            loss_embedding_t = torch.tensor(0., device=fabric.device)  # prompt decoder from teacher

            'student vs ground truth'
            # gt_kernel_size = 5
            # for pred_mask, gt_mask in zip(student_pred_masks, gt_masks):
            #
            #     gt_mask = edge_smooth(gt_mask, kernel_size=gt_kernel_size)
            #
            #     loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
            #     loss_dice += dice_loss(pred_mask, gt_mask, num_masks)

            loss_total_gt = 20. * loss_focal + loss_dice + loss_iou

            'student vs teacher'
            # mask decoder
            t_kernel_size = 5
            sigma = 1.0

            for student_pred_mask, teacher_pred_mask in zip(student_pred_masks, teacher_pred_masks):
                teacher_pred_mask = F.sigmoid(teacher_pred_mask)
                student_pred_mask = F.sigmoid(student_pred_mask)

                teacher_pred_mask = gaussian_smooth(teacher_pred_mask, kernel_size=t_kernel_size, sigma=sigma)  # teacher label smoothing
                # student_pred_mask = gaussian_smooth(student_pred_mask, kernel_size=t_kernel_size, sigma=sigma)  # student label smoothing

                loss_mask_t += mask_loss(student_pred_mask, teacher_pred_mask)  # decoder

            # OBB prompt encoder
            delta = 0.3
            teacher_sparse_embeddings = gaussian_smooth_1d(teacher_sparse_embeddings, kernel_size=t_kernel_size, delta=delta)

            loss_embedding_t = embedding_loss(student_sparse_embeddings, teacher_sparse_embeddings)  # OBB prompt encoder


            'sum loss'
            loss_total_t = 0.9 * loss_mask_t + 0.1 * loss_embedding_t

            gamma = 0.005
            loss_total = gamma * loss_total_gt + (1 - gamma) * (sigma**2 + ((t_kernel_size - 1) / 2)**2) * loss_total_t

            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            mask_t_losses.update(loss_mask_t.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter+1}/{len(train_dataloader)}]'
                         f' | Time [({batch_time.avg:.3f}s)]'
                         f' | Data [({data_time.avg:.3f}s)]'
                         f' | Focal Loss [({focal_losses.avg:.4f})]'
                         f' | Dice Loss [({dice_losses.avg:.4f})]'
                         f' | Mask_t Loss [({mask_t_losses.avg:.4f})]'
                         f' | Total Loss [({total_losses.avg:.4f})]')


def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps * 2 / cfg.batch_size:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0] * 2 / cfg.batch_size:
            return 1.0
        elif step < cfg.opt.steps[1] * 2 / cfg.batch_size:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated() / 1024 ** 2  # Convert bytes to megabytes


def main(student_cfg: Box, teacher_cfg: Box) -> None:
    ''

    "******************************************************teacher*****************************************************"
    teacher_fabric = L.Fabric(accelerator="auto",
                      devices=teacher_cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(teacher_cfg.out_dir, name="lightning-sam")])
    teacher_fabric.launch()
    teacher_fabric.seed_everything(1337 + teacher_fabric.global_rank)

    if teacher_fabric.global_rank == 0:
        os.makedirs(teacher_cfg.out_dir, exist_ok=True)

    # with fabric.device:
    teacher_model = Model(teacher_cfg)
    teacher_model.setup(teacher_fabric.device)

    print("******************************teacher_model***************************")
    print(teacher_model)
    # image = torch.randn(1, 3, 1024, 1024).to(teacher_fabric.device)
    # boxes = torch.randn(1, 5).to(teacher_fabric.device)
    # boxes = [boxes]
    # boxes = tuple(boxes)
    # flops, params = profile(teacher_model, inputs=(image, boxes))  # tiny: 9.79M  b: 90M   l:300M   h:600M
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))



    "******************************************************student*****************************************************"
    student_fabric = L.Fabric(accelerator="auto",
                              devices=student_cfg.num_devices,
                              strategy="auto",
                              loggers=[TensorBoardLogger(student_cfg.out_dir, name="lightning-sam")])
    student_fabric.launch()
    student_fabric.seed_everything(1337 + student_fabric.global_rank)

    if student_fabric.global_rank == 0:
        os.makedirs(student_cfg.out_dir, exist_ok=True)

    # with fabric.device:
    student_model = Model(student_cfg)
    student_model.setup(student_fabric.device)

    print("****************************student_model******************************")
    print(student_model)
    # image = torch.randn(1, 3, 1024, 1024).to(student_fabric.device)
    # boxes = torch.randn(1, 5).to(student_fabric.device)
    # boxes = [boxes]
    # boxes = tuple(boxes)
    # flops, params = profile(student_model, inputs=(image, boxes))  # tiny: 9.79M  b: 90M   l:300M   h:600M
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))

    # total_num_params = 0

    # target_layer = student_model.model.image_encoder
    # num_params = sum(p.numel() for p in target_layer.parameters())
    # total_num_params += num_params
    # print(f"Number of parameters in {target_layer.__class__.__name__}: {num_params / 1e6}")
    #
    # target_layer = student_model.model.prompt_encoder
    # num_params = sum(p.numel() for p in target_layer.parameters())
    # total_num_params += num_params
    # print(f"Number of parameters in {target_layer.__class__.__name__}: {num_params / 1e6}")
    #
    # target_layer = student_model.model.mask_decoder
    # num_params = sum(p.numel() for p in target_layer.parameters())
    # total_num_params += num_params
    # print(f"Number of parameters in {target_layer.__class__.__name__}: {num_params / 1e6}")
    #
    # print(f"Total Number of parameters: {total_num_params / 1e6}")

    student_optimizer, student_scheduler = configure_opt(student_cfg, student_model)
    student_model, student_optimizer = student_fabric.setup(student_model, student_optimizer)



    "******************************************************train*******************************************************"
    train_data, val_data = load_datasets(student_cfg, student_model.model.image_encoder.img_size)
    train_data = student_fabric._setup_dataloader(train_data)
    val_data = student_fabric._setup_dataloader(val_data)

    if not os.path.exists('runs/val/log.txt'):
        path = Path('runs/val/log.txt')
        path.touch()
    else:
        with open('runs/val/log.txt', 'w') as file:
            file.truncate(0)

    torch.cuda.reset_max_memory_allocated()  

    train_sam(student_cfg, student_fabric, student_model, teacher_model, student_optimizer, student_scheduler, train_data, val_data)
    validate(student_fabric, student_model, val_data, epoch=0)

    max_memory = get_max_memory_allocated() 
    print(f"max_memory: {max_memory:.2f} MB")  # vit_h: 5822 MB; vit_l: 4485 MB; vit_b: 2854 MB; vit_tiny: 375 MB


def  gaussian_smooth(teacher_mask, kernel_size=3, sigma=1.0):

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    kernel_center = kernel_size // 2
    total = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_center, j - kernel_center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            total += kernel[i, j]
    kernel /= total

    device = teacher_mask.device
    teacher_mask = teacher_mask.cpu().numpy()

    for k in range(len(teacher_mask)):
        teacher_mask[k] = cv2.filter2D(teacher_mask[k], -1, kernel)  
    teacher_mask = torch.from_numpy(teacher_mask).to(device)

    return teacher_mask


def  gaussian_smooth_1d(sparse_embeddings, kernel_size=3, delta=0.3):
    kernel = np.zeros(kernel_size, dtype=np.float32)

    kernel_center = kernel_size // 2
    total = 0
    for i in range(kernel_size):
        x = i - kernel_center
        kernel[i] = np.exp(-(x ** 2) / (2 * delta ** 2))
        total += kernel[i]
    kernel /= total

    device = sparse_embeddings.device
    teacher_embeddings = sparse_embeddings.cpu().numpy()

    teacher_embeddings = np.squeeze(teacher_embeddings)
    for t in range(len(teacher_embeddings)):
        teacher_embeddings[t] = convolve1d(teacher_embeddings[t], kernel)
    teacher_embeddings = teacher_embeddings[None, :]

    teacher_embeddings = torch.from_numpy(teacher_embeddings).to(device)

    return teacher_embeddings


def edge_smooth(gt_mask, kernel_size=3):
    device = gt_mask.device
    for k in range(len(gt_mask)): 
        temp_mask = torch.zeros(gt_mask[k].shape[0]-(kernel_size-1), gt_mask[k].shape[1]-(kernel_size-1)).to(device)

        for i in range(kernel_size):
            for j in range(kernel_size):
                temp_mask += gt_mask[k][i : (i + temp_mask.shape[0]), j : (j + temp_mask.shape[1])]

        # transform
        temp_mask =  temp_mask / (kernel_size ** 2) * math.pi  # (0, pi)
        temp_mask = torch.cos(temp_mask)  # (1, -1)
        temp_mask = (temp_mask * (-1) + 1) / 2  # (0, 1)

        gt_mask[k][int((kernel_size-1)/2) : int(gt_mask[k].shape[0]-(kernel_size-1) / 2), int((kernel_size-1)/2) : int(gt_mask[k].shape[1]-(kernel_size-1) / 2)] = temp_mask

    return gt_mask


if __name__ == "__main__":
    main(student_cfg, teacher_cfg)
