from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
import config
import api.examination.function as myfunc
import cv2 as cv

app = APIRouter()

async def frame(detr: myfunc.Detector, cap: cv.VideoCapture, request: Request):
    if not cap.isOpened():
        print('ERROR: Can\'t open the camera')
        cap.release()
        return
    while True:
        if await request.is_disconnected():
            break
        ret, image = cap.read()
        if not ret:
            print('ERROR: Can\'t get image from the camera')
            continue
        res = detr.examine(image)
        success, bin_img = cv.imencode('.jpg', res)
        if success:
            binary_data = bin_img.tobytes()
            yield(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + binary_data + b'\r\n'
            )
        else:
            print('ERROR: Can\'t change the form')
    cap.release()

@app.get('/examination/')
async def main_examination(request: Request, index: int = Query(default=0, ge=0, lt=len(config.cameras))):
    cap = cv.VideoCapture(config.cameras[index])
    detr = myfunc.Detector(config.YOLO_MODEL_PATH, config.FACE_DB_PATH)
    return StreamingResponse(frame(detr, cap, request), media_type='multipart/x-mixed-replace; boundary=frame')