import os
import cv2
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from sam.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# x1, y1, x2, y2, x3, y3, x4, y4 to x1, y1(0-1), x2, y2(0-1), angle(0-180)
def convert(points, size):
    points = points.reshape((4, 2))
    points = points.astype(np.float32)

    rect = cv2.minAreaRect(points)
    center, (width, height), angle = rect

    if width >= height:
        w = width
        h = height
        a = angle
    elif width < height:
        w = height
        h = width
        a = angle + 90

    if a == 180:
        a = 0

    x_min = (center[0] - w / 2) / size
    y_min = (center[1] - h / 2) / size
    x_max = (center[0] + w / 2) / size
    y_max = (center[1] + h / 2) / size

    return [x_min, y_min, x_max, y_max, a]


class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, data_augment, is_data_augment):
        self.root_dir = root_dir
        self.data_augment = data_augment
        self.is_data_augment = is_data_augment

        'OBB'
        if self.root_dir.split('/')[-1] == 'train_images':
            self.rbox_path = os.path.join(self.root_dir.split("train_images")[0], 'train_box_labels/')
        elif self.root_dir.split('/')[-1] == 'val_images':
            self.rbox_path = os.path.join(self.root_dir.split("val_images")[0], 'val_box_labels/')
        else:
            self.rbox_path = None

        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        # HEIGHT, WIDTH = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        'OBB'
        bboxes = []
        rbox_path = self.rbox_path + str(image_id) + '.txt'
        with open(rbox_path, 'r') as rb:
            for line_rb in rb:
                words = line_rb.split()

                bbox_temp = np.zeros((8))
                for word_i in range(8):
                    bbox_temp[word_i] = int(words[word_i])

                #  x1, y1, x2, y2, x3, y3, x4, y4 to x1, y1(0-1), x2, y2(0-1), angle(0-180)
                bbox_temp = convert(bbox_temp, image.shape[0])
                bboxes.append(bbox_temp)

        masks = []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        # transform
        image, masks, bboxes = self.data_augment(image, masks, np.array(bboxes), self.is_data_augment)

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)

        return image, torch.tensor(bboxes), torch.tensor(masks).float()


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


class DataAugment:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes, is_data_augment=False):
        ''

        if is_data_augment:

            'Random Boxes'
            # bboxes:(n, 5)   x1, y1(0 - 1), x2, y2(0 - 1), angle(0 - 180)
            for k in range(len(bboxes)):
                scale_length = 0.02
                scale_angle = 2

                d_w_a = random.uniform(-scale_length, scale_length) * (bboxes[k][2] - bboxes[k][0])  # scale_length% * w
                d_w_b = random.uniform(-scale_length, scale_length) * (bboxes[k][2] - bboxes[k][0])  # scale_length% * w
                d_h_a = random.uniform(-scale_length, scale_length) * (bboxes[k][3] - bboxes[k][1])  # scale_length% * h
                d_h_b = random.uniform(-scale_length, scale_length) * (bboxes[k][3] - bboxes[k][1])  # scale_length% * h
                d_angle = random.randint(-scale_angle, scale_angle)  # -scale_angle～scale_angle deg

                bboxes[k][0] += d_w_a
                bboxes[k][2] += d_w_b
                bboxes[k][1] += d_h_a
                bboxes[k][3] += d_h_b
                bboxes[k][4] += d_angle

                # clamp
                bboxes[k][:4][bboxes[k][:4] < 0] = 0
                bboxes[k][:4][bboxes[k][:4] >= 1] = 0.999999

                if bboxes[k][4] < 0:
                    bboxes[k][4] += 180
                elif bboxes[k][4] >= 180:
                    bboxes[k][4] -= 180


            'Random Flip'
            prob_flip_horizontal = 0.5
            prob_flip_vertical = 0.5
            radom_horizontal = random.random()  # 0 -1
            radom_vertical = random.random()  # 0 -1

            if radom_horizontal > prob_flip_horizontal:
                # image:(1024, 1024, 3)
                image = cv2.flip(image, 1)

                # bboxes:(n, 5)   x1, y1(0 - 1), x2, y2(0 - 1), angle(0 - 180)
                for i in range(len(bboxes)):
                    bboxes[i][0] = 1 - bboxes[i][0]  # x
                    bboxes[i][2] = 1 - bboxes[i][2]  # x
                    bboxes[i][4] = 180 - bboxes[i][4]  # angle

                    if bboxes[i][0] >= 1:
                        bboxes[i][0] = 0.999999
                    if bboxes[i][2] >= 1:
                        bboxes[i][2] = 0.999999
                    if bboxes[i][4] == 180:
                        bboxes[i][4] = 0

                # masks: list:n(1024, 1024)
                for j in range(len(masks)):
                    masks[j] = cv2.flip(masks[j], 1)

            if radom_vertical > prob_flip_vertical:
                image = cv2.flip(image, 0)

                for i in range(len(bboxes)):
                    bboxes[i][1] = 1 - bboxes[i][1]  # y
                    bboxes[i][3] = 1 - bboxes[i][3]  # y
                    bboxes[i][4] = 180 - bboxes[i][4]  # angle

                    if bboxes[i][1] >= 1:
                        bboxes[i][1] = 0.999999
                    if bboxes[i][3] >= 1:
                        bboxes[i][3] = 0.999999
                    if bboxes[i][4] == 180:
                        bboxes[i][4] = 0

                for j in range(len(masks)):
                    masks[j] = cv2.flip(masks[j], 0)


        image = self.to_tensor(image)

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    data_augment = DataAugment()  # 1024
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        data_augment=data_augment,
                        is_data_augment=True)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      data_augment=data_augment,
                      is_data_augment=False)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=1,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
