import random
import glob
import cv2
import torchvision
import torch
import numpy as np

from src.model import model

NUM_IMAGES_TO_SHOW = 10
THRESHOLD = 0.5
IOU_THRESHOLD = 0.1
SAVE_IMG = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

images = random.sample(glob.glob('trainval/images/*.jpg'),NUM_IMAGES_TO_SHOW)

model = model.load_from_checkpoint('trainval/models/epoch=4-step=279.ckpt').to(DEVICE)

label_dict = {1:'Person', 2:'Car'}
color_dict = {1: (255,0,0), 2: (0,0,255)}

for img_path in images:
    orig_img = cv2.imread(img_path)
    img = cv2.cvtColor(orig_img.copy(),cv2.COLOR_BGR2RGB)
    img = img/255.
    img = torchvision.transforms.ToTensor()(img).type(torch.float).to(DEVICE)

    # Run image through the model
    results = model(torch.unsqueeze(img,0))

    # Extract the result
    boxes = results[0]['boxes'].detach().cpu()
    labels = results[0]['labels'].detach().cpu()
    scores = results[0]['scores'].detach().cpu()

    # Filter result using Threshold
    boxes = boxes[np.where(scores>THRESHOLD)]
    labels = labels[np.where(scores>THRESHOLD)]
    scores = scores[np.where(scores>THRESHOLD)]

    # Non-max Suppression
    indices = torchvision.ops.nms(boxes,scores,iou_threshold=IOU_THRESHOLD)
    boxes = boxes[indices].numpy().tolist()
    labels = labels[indices].numpy().tolist()
    scores = scores[indices].numpy().tolist()

    for i,(x,y,x1,y1) in enumerate(boxes):
        orig_img = cv2.rectangle(orig_img, (int(x),int(y)),(int(x1),int(y1)),(255,255,255),1)
        orig_img = cv2.putText(orig_img, label_dict[labels[i]], (int(x)-5,int(y)-5), 0, 0.75, color_dict[labels[i]], 2)
    
    cv2.imshow('Object Detection Test', orig_img)
    cv2.waitKey(0)

    if SAVE_IMG:
        cv2.imwrite('Outputs/'+str(img_path.split("\\")[-1]), orig_img)

cv2.destroyAllWindows()