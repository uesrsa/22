import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-main\ultralytics\cfg\models\v8\yolov8-GFPN1.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:\ultralytics-main\150016\data.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=24,
                close_mosaic=0,
                workers=6,
                device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                #resume=r'D:\ultralytics-main\runs\train\exp13\weights\last.pt', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                conf=0.25,
                )