import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-main\runs\train\exp56\weights\best.pt')
    model.val(data=r'D:\ultralytics-main\150016\data.yaml',
              split='val',
              imgsz=640,
              batch=24,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )