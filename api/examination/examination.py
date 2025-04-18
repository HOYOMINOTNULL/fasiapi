from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
import config
from fastapi.responses import PlainTextResponse
import api.examination.function as myfunc
import cv2 as cv

app = APIRouter()

conf_threshold = 0.5 #置信度阈值
invalid_flag = False #是否存在违规现象

async def frame(detr: myfunc.Detector, cap: cv.VideoCapture, request: Request):
    '''
    帧生成函数
    '''
    global invalid_flag
    while True:
        if await request.is_disconnected(): #前端断开请求则停止帧生成
            break
        ret, image = cap.read()
        if not ret:
            print('ERROR: Can\'t get image from the camera')
            break
        res, invalid_flag = detr.examine(image, conf_threshold)
        success, bin_img = cv.imencode('.jpg', res)
        if success:
            binary_data = bin_img.tobytes()
            yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + binary_data + b'\r\n'
            )
        else:
            print('ERROR: Can\'t change the form')
    invalid_flag = False
    cap.release()

@app.get('/examination/')
async def main_examination(request: Request, index: int = Query(default=0, ge=0, lt=len(config.cameras))):
    cap = cv.VideoCapture(config.cameras[index])
    if not cap.isOpened():
        print('ERROR: Can\'t open the camera')
        cap.release()
        return
    detr = myfunc.Detector(config.YOLO_MODEL_PATH, config.FACE_DB_PATH)
    return StreamingResponse(frame(detr, cap, request), media_type='multipart/x-mixed-replace; boundary=frame') #以流的形式返回帧

@app.get('/examination/confidence/')
async def change_conf(v: float = Query(ge=0, le=1)):
    '''
    更改置信度阈值
    '''
    global conf_threshold
    conf_threshold = v

@app.get('/examination/alert',response_class=PlainTextResponse)
async def alert():
    '''
    是否发出警报，'1'为报警，'0'为否
    '''
    if invalid_flag:
        return "1"
    else:
        return "0"
