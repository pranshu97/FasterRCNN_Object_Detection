import cv2
import torch
import torchvision
import numpy as np

from src.model import model

INPUT_VID = 'test.mp4'
NUM_IMAGES_TO_SHOW = 5
THRESHOLD = 0.5
IOU_THRESHOLD = 0.1
SAVE_VID = True
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.load_from_checkpoint('trainval/models/epoch=4-step=279.ckpt').to(DEVICE)

label_dict = {1:'Person', 2:'Car'}
color_dict = {1: (255,0,0), 2: (0,0,255)}

cap = cv2.VideoCapture(INPUT_VID)

if SAVE_VID:
    writer = cv2.VideoWriter('Outputs/output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (640,640))

while (cap.isOpened()):
 
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))

    img = cv2.cvtColor(frame.copy(),cv2.COLOR_BGR2RGB)
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
    
    # plot boxes and classes
    for i,(x,y,x1,y1) in enumerate(boxes):
        frame = cv2.rectangle(frame, (int(x),int(y)),(int(x1),int(y1)),(255,255,255),1)
        frame = cv2.putText(frame, label_dict[labels[i]], (int(x)-5,int(y)-5), 0, 0.75, color_dict[labels[i]], 2)

    cv2.imshow('Object Detection Test', frame)
    if cv2.waitKey(10)==ord('q'):
        break

    if SAVE_VID:
        writer.write(frame)

cv2.destroyAllWindows()
cap.release()
writer.release()

