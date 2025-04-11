import cv2 as cv
from ultralytics import YOLO
import numpy as np
import face_recognition as fr
import io
import json
import sqlite3 as sq
import datetime
import config

def EncodingToStr(encoding: np.ndarray) -> str:
    encoding_list = encoding.tolist()
    ret = json.dumps(encoding_list)
    return ret

def  StrToEncoding(s: str) -> np.ndarray:
    temp = json.loads(s)
    encoding = np.array(temp)
    return encoding

class Detector():
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH, data_path: str = config.FACE_DB_PATH):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.25  # 进一步降低置信度阈值
        self.iou_threshold = 0.45  # 调整NMS阈值
        self.input_size = 640
        self.invalid_cls = 1 #非法标签
        self.database_path = data_path
        # 模型标签 {0: 'person', 1: 'head', 2: 'helmet'}



    def examine(self, image: np.ndarray) -> np.ndarray:
        trans = cv.cvtColor(image, cv.COLOR_BGR2RGB) #预处理
        res = self.model.predict(trans, imgsz=self.input_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    augment=True,
                    verbose=False)
        raw_image = image.copy()
        invalid_record = []
        for r in res:
            image = r.plot(img=image)
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id == self.invalid_cls:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cropped = raw_image[y1:y2, x1:x2]
                    invalid_record.append(cropped)
        invalid_faces = self.face_recognize(invalid_record)
        for face in invalid_faces:
            self.record(face)
        return image
    
    def face_recognize(self, captures: list[np.ndarray]) -> list[np.ndarray]:
        '''
        从输入的图片中识别人脸并裁剪，返回
        '''
        ret = []
        for capture in captures:
            locations = fr.face_locations(capture)
            for location in locations:
                face_cropped = capture[location[0]:location[2], location[3]:location[1]]
                ret.append(face_cropped)
        return ret
    
    def record(self, img: np.ndarray):
        '''
        将违规人脸img的相关信息存入违规记录数据库，一人一天只记一次违规
        如果该人脸第一次出现，则将他以UNKONW的type存入人脸数据库中
        '''
        img = img.astype(np.uint8)
        target = fr.face_encodings(img)
        database = sq.connect(self.database_path)
        cursor = database.cursor()
        cursor.execute('SELECT * FROM face_data')
        res = cursor.fetchall()
        flag = False
        for r in res:
            encode = StrToEncoding(r[-1])
            comp_res = fr.compare_faces(target, encode)
            if comp_res:
                flag = comp_res[0]
            if flag:
                break
        current = datetime.datetime.now()
        if flag:
            face_id = r[0]
            type = r[1]
            cursor.execute("SELECT time FROM examination_res WHERE face_id=? ORDER BY time DESC", (face_id,))
            res = cursor.fetchone()
            if res:
                time = res[-1]
                time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                if time.year != current.year and time.month != current.month and time.day != current.day:
                    cursor.execute("INSERT INTO examination_res (type, face_id, time) VALUES (?, ?, ?)", 
                                   (type, face_id, current.strftime("%Y-%m-%d %H:%M:%S")))
                    database.commit()
        else:
            success, bin_img = cv.imencode('.jpg', img)
            if success and target:
                binary_data = bin_img.tobytes()
                bin_img =  io.BytesIO(binary_data)
                image_data = bin_img.getvalue()
                target_encoding = EncodingToStr(target[0])
                cursor.execute('INSERT INTO face_data (face_type, filename, size, type, data, encoding) VALUES (?, ?, ?, ?, ?, ?)',
                               ('UNKNOWN', 'UNKNOWN.jpg', len(image_data), 'jpg', image_data, target_encoding))
                database.commit()
                cursor.execute('SELECT id FROM face_data ORDER BY id DESC')
                res = cursor.fetchone()
                face_id = res[0]
                cursor.execute("INSERT INTO examination_res (type, face_id, time) VALUES (?, ?, ?)", 
                                   ('UNKNOWN', face_id, current.strftime("%Y-%m-%d %H:%M:%S")))
                database.commit()
        database.close()

if __name__ == '__main__':
    detr = Detector('best.pt', 'database.db')
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('error')
    while True:
        ret, image = cap.read()
        if not ret:
            print('error2222')
            break
        res = detr.examine(image)
        cv.imshow('pre', image)
        cv.imshow('after', res)
        if cv.waitKey(10) == ord(' '):
            break
    cap.release()