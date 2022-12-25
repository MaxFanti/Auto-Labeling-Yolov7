import cv2
import numpy as np
import torch
import cv2 as cv
from models.yolo import Model
from utils.torch_utils import select_device
import os

vidcap = cv2.VideoCapture('csgo.mkv')
success,image = vidcap.read()
count = 0

image_width = 1280
image_height = 720

DT_IMG_SAVE_PATH = "./data/images"
DT_LABEL_SAVE_PATH = "./data/labels"
DT_LABEL_FORMAT = "{id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"

if not os.path.exists(DT_IMG_SAVE_PATH):
    os.makedirs(DT_IMG_SAVE_PATH)
    
if not os.path.exists(DT_LABEL_SAVE_PATH):
    os.makedirs(DT_LABEL_SAVE_PATH)   
def get_dt_label_content(label, xmin, ymin, xmax, ymax, image_width, image_height):
    global DT_LABEL_FORMAT
    data = DT_LABEL_FORMAT

    x_center = (xmin + xmax) /2
    y_center = (ymin + ymax) /2

    x_center_norm = abs(x_center) / image_width
    y_center_norm = abs(y_center) / image_height

    width_norm = abs(xmax - xmin) / image_width
    height_norm = abs(ymax - ymin) / image_height

    data = data.replace("{id}", str(label))
    data = data.replace("{x_center_norm}", "{:.4f}".format(x_center_norm))
    data = data.replace("{y_center_norm}", "{:.4f}".format(y_center_norm))
    data = data.replace("{width_norm}", "{:.4f}".format(width_norm))
    data = data.replace("{height_norm}", "{:.4f}".format(height_norm))

    return str(data)

def LoadModel():
    path_or_model = './best.pt'  # path to model
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(
        path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model
    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    hub_model = hub_model.autoshape()
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    model = hub_model.to(device)
    print("Model ready!")
    return model

model = LoadModel()

def Yolov7_render(image):
    result = np.array(image)
    result = cv.cvtColor(result, cv.COLOR_RGB2BGR)
    result = model(result)

    info = result.pandas().xyxy[0].to_dict(orient="records")
    if len(info) != 0:
        for obj in info:
                conf, cls, name = obj['confidence'], int(
                    obj['class']), str(obj['name'])
                xf, yf, xi, yi = int(obj['xmax']), int(
                    obj['ymax']), int(obj['xmin']), int(obj['ymin'])

                data = get_dt_label_content(cls,xi,yi,xf,yf, image_width, image_height)
                if(data != ''):
                    with open(DT_LABEL_SAVE_PATH+"/frame%d.txt" % count, 'w') as f:
                            f.write(data + "\n")

while success:
  if(count % 25 == 0):
    cv2.imwrite(DT_IMG_SAVE_PATH+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
    Yolov7_render(image)
    print('Save a new image: ', success)
  success,image = vidcap.read()
  count += 1

print("Finally!")

