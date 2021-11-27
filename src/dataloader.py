import os
import json
import numpy as np
import torch
import cv2
import config
from torch.utils.data import Dataset, DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        annotations = json.load(open(os.path.join(root,'annotations','annotations-cleaned.json'),'r'))
        self.annots = {}
        for obj in annotations['annotations']:
            if obj['image_id'] in self.annots:
                self.annots[obj['image_id']].append(obj)
            else:
                self.annots[obj['image_id']] = [obj]

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        img_id = int(self.imgs[idx].split('.')[0].split('_')[-1]) - 1 # Zero indexed
        objects = self.annots[img_id]

        # get bounding box coordinates for each mask
        boxes = []
        labels = []
        for obj in objects:
            if obj['bbox'][2]>0 and obj['bbox'][3]>0: # Removing very small boxes, else albumentations throw exceptions.
                boxes.append([obj['bbox'][0],obj['bbox'][1],obj['bbox'][0]+obj['bbox'][2],obj['bbox'][1]+obj['bbox'][3]]) # xywh to xyxy
                labels.append(obj['category_id'])
        
        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']      

        
        # Removing bounding boxes with 0 width or height. (Generated after augmentations/ resizing.)
        # indices = [i for i in range(len(boxes)) if (boxes[i][2]-boxes[i][0])>=1 and (boxes[i][3]-boxes[i][1])>=1]
        # boxes = np.array(boxes)[indices] 
        # labels = np.array(labels)[indices]

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float) 
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        # print(target)
        # indices = [i for i in range(len(boxes)) if (boxes[i][2]-boxes[i][0])>=1 and (boxes[i][3]-boxes[i][1])>=1]
        # target["boxes"] = target["boxes"][indices]        
        # target["labels"] = target["labels"][indices]

        return img, target

    def __len__(self):
        return len(self.imgs)


dataset = Dataset(config.ROOT,config.TRANFORMS)
train_size = int(config.TRAIN_RATIO * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

def collate_fn(x):
    '''
    Create batch with varying size of tensors.
    Tried using lambda function for this but that's not supported.
    '''
    return tuple(zip(*x))

train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.TRAIN_WORKERS,
                            pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=config.VAL_WORKERS,
                            pin_memory=True, collate_fn=collate_fn, persistent_workers=True)