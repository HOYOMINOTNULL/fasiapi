from ultralytics import YOLO
import cv2 as cv
import threading
import  numpy as np
import  traceback
from config import YOLO_MODEL_PATH,cameras
model=YOLO(YOLO_MODEL_PATH)



def acquire_frame_deal(frame):

    return model(frame)[0].plot()



class Mycam:
    def __init__(self):
        self.model = model
        self.cameras=[]
        self.frames=[]
        self.successes=[]
        self.dealed_frames=[]
        print(f"相机个数:{len(cameras)}")
        for i,cameraid in enumerate(cameras):
            self.cameras.append(cv.VideoCapture(cameraid))
            if self.cameras[i].isOpened() == False:
                print(f"open camera{i} failure")
                exit(0)
            #self.cameras[i].set(cv.CAP_PROP_FRAME_WIDTH, 1280)  设置分辨率
            #self.cameras[i].set(cv.CAP_PROP_FRAME_HEIGHT, 640)
            # 获取相机
            success, frame = self.cameras[i].read()
            if success == 0:
                print("read camera{i} failure")
                exit(0)
            self.successes.append(success)
            self.frames.append(frame)
            self.dealed_frames.append(self.model(self.frames[i])[0].plot())
            # 尝试读取

        thread1 = threading.Thread(target=self.read)  # 多线程会执行以下函数
        thread1.start()
        thread2 = threading.Thread(target=self.deal)
        thread2.start()


    def read(self):
        while 1:
            for idx,camera in enumerate(self.cameras):
                self.successes[idx], tem = camera.read()
                if self.successes[idx]:
                    self.frames[idx] = tem
                else:
                    traceback.print_exc("read camera failure")
                    exit(0)


    def deal(self):
        while 1:
            for idx,frame in enumerate(self.frames):
                tem=self.model(frame)[0].plot()
                self.dealed_frames[idx]=tem


    def get(self):
        return self.dealed_frames


if __name__ =="__main__":
    mycam=Mycam()
    while 1:
        frames=mycam.get()
        for i in range(0,len(frames)):
            print(type(frames[i]))
            cv.imshow(f"camera{i}",frames[i])
        cv.waitKey(20)

