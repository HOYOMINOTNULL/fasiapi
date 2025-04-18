from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    results = model.train(data=r"/root/.jupyter/project/Smart_Construction/data/custom_data.yaml",
                          epochs=500,
                          imgsz=640,
                          patience=30,
                          device=0,
                          batch=96,
                          save=True,
                          save_period=15,
                          cache="disk",
                          workers=6,
                          lr0=0.001,
                          pretrained=True,
                          optimizer='Adam',
                          amp=True,
                          val=True,
                          )