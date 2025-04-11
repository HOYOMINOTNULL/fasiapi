from typing import Optional

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
import sqlite3 as sq
import io
import api.examination.function as myfunc
import cv2 as cv
import numpy as np
import face_recognition as fr
import config

app = APIRouter()

class FaceQuery(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None

class FaceData(BaseModel):
    code: str
    name: str
    data: UploadFile

'''
face_data表结构:
    id: int,
    face_type: text,  该人脸信息类型，已知KNOWN，未知UNKNOWN
    code: text, 工号
    name: text, 姓名
    filename: text, 图像文件名
    size: int, 图像大小
    type: text, 图像格式
    data: BLOB  二进制图像数据
    encoding: text, 人脸编码
'''

@app.get('/face/all')
async def all_face_data() -> list[FaceData]:
    '''
    返回数据库中所有人脸数据
    '''
    ret = []
    database = sq.connect(config.FACE_DB_PATH)
    cursor = database.cursor()
    cursor.execute('SELECT * FROM face_data')
    res = cursor.fetchone()
    while res:
        type = res[1]
        if type == 'KNOWN':
            code = res[2]
            name = res[3]
            filename = res[4]
            size = res[5]
            data = io.BytesIO(res[7])
            ret.append(FaceData(code, name, UploadFile(data, size=size, filename=filename)))
        res = cursor.fetchone()
    database.close()
    return ret

@app.post('/face/upload')
async def upload_face(data: FaceData):
    '''
    向数据库上传人脸数据，工号code，姓名name为必须
    '''
    filename = data.data.filename
    index = filename.find('.')
    type = filename[index+1:]
    dtr = myfunc.Detector()
    file = io.BytesIO(data.data.file)
    file = file.read()
    np_arr = np.frombuffer(file, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    cropped_img = dtr.face_recognize([img])
    if cropped_img:
        img = cropped_img[0]
        target = fr.face_encodings(img)
        database = sq.connect(config.FACE_DB_PATH)
        cursor = database.cursor()
        cursor.execute("SELECT id, data FROM face_data WHERE type='UNKNOWN'")
        res = cursor.fetchall()
        flag = False
        for r in res:
            encode = myfunc.StrToEncoding(r[-1])
            comp_res = fr.compare_faces(target, encode)
            if comp_res:
                flag = comp_res[0]
            if flag:
                break
        target_encoding = myfunc.EncodingToStr(target[0])
        save_data = ('KNOWN', data.code, data.name, filename, len(data.data.file), type, data.data.file, target_encoding)
        if flag:
            cursor.execute(
                f"UPDATE face_data (face_type, code, name, filename, size, type, data, encoding) VALUES (?, ?, ?, ?, ?, ?, ?) WHERE id={r[0]}", 
                save_data)
        else:
            cursor.execute(
                'INSERT INTO face_data (face_type, code, name, filename, size, type, data, encoding) VALUES (?, ?, ?, ?, ?, ?, ?)',
                save_data)
        database.commit()
        database.close()

@app.get('/face/query/')
async def query_face_data(query: FaceQuery) -> Optional[FaceData]:
    '''
    查询数据库中的人脸数据，以工号code和姓名name为索引，code不为None则优先以code为索引进行查询，
    否则以name为索引进行查询，若code和name都为None则无法查询
    '''
    if query.code == None and query.name == None:
        return 
    elif query.code != None:
        database = sq.connect(config.FACE_DB_PATH)
        cursor = database.cursor()
        cursor.execute('SELECT * FROM face_data WHERE code=?', (query.code,))
        res = cursor.fetchone()
        database.close()
        if not res:
            return None
        name = res[3]
        filename = res[4]
        size = res[5]
        data = res[7]
        ret = FaceData(query.code, name, UploadFile(data, size=size, filename=filename))
        database.close()
        return ret
    else:
        database = sq.connect(config.FACE_DB_PATH)
        cursor = database.cursor()
        cursor.execute('SELECT * FROM face_data WHERE name=?', (query.name,))
        res = cursor.fetchone()
        database.close()
        if not res:
            return None
        code = res[2]
        filename = res[4]
        size = res[5]
        data = res[7]
        ret = FaceData(code, query.name, UploadFile(data, size=size, filename=filename))
        database.close()
        return ret