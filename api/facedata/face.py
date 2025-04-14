from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
import sqlite3 as sq
import io
import base64
import api.examination.function as myfunc
import cv2 as cv
import numpy as np
import face_recognition as fr
import config

app = APIRouter()


class FaceData(BaseModel):
    code: str
    name: str
    image: str

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
            data = res[7]
            image = base64.b64encode(data).decode()
            ret.append(FaceData(code=code, name=name, image=image))
        res = cursor.fetchone()
    database.close()
    return ret

@app.post('/face/upload')
async def upload_face(code: str = Form(), name: str = Form(), data: UploadFile = File()):
    '''
    向数据库上传人脸数据，工号code，姓名name为必须
    '''
    filename = data.filename
    index = filename.find('.')
    type = filename[index+1:]
    dtr = myfunc.Detector()
    bytes_file = await data.read()
    np_arr = np.frombuffer(bytes_file, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    location = dtr.face_recognize([img])
    if location:
        img = img.astype(np.uint8)
        target = fr.face_encodings(img, known_face_locations=location[0])
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
        save_data = ('KNOWN', code, name, filename, len(bytes_file), type, bytes_file, target_encoding)
        if flag:

            cursor.execute(
                "UPDATE face_data SET face_type = ?, code = ?, name = ?, filename = ?, size = ?, type = ?, data = ?, encoding = ? WHERE id = ?", 
                (*save_data, r[0])
            )
        else:
            cursor.execute(
                'INSERT INTO face_data (face_type, code, name, filename, size, type, data, encoding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                save_data)
        database.commit()
        database.close()
    else:
        return {'message': 'There\'s no face'}

@app.get('/face/query/')
async def query_face_data(code: Optional[str] = None, name: Optional[str] = None) -> list[FaceData]:
    '''
    查询数据库中的人脸数据，以工号code和姓名name为索引，code不为None则优先以code为索引进行查询，
    否则以name为索引进行查询，若code和name都为None则无法查询
    '''
    if code == None and name == None:
        return

    database = sq.connect(config.FACE_DB_PATH)
    cursor = database.cursor()
    if code !=None and name !=None:
        cursor.execute('SELECT * FROM face_data WHERE code=? and name=?', (code,name))
    elif code != None:
        cursor.execute('SELECT * FROM face_data WHERE code=? ', (code,))
    else :
        cursor.execute('SELECT * FROM face_data WHERE name=? ', (name,))

    results = cursor.fetchall()
    database.close()
    ret=[]
    if len(results) ==0:
        return
    for r in results:
        code = r[2]
        name = r[3]
        data = r[7]
        image = base64.b64encode(data).decode()
        ret.append( FaceData(code=code, name=name, image=image))

    #print(f"code{ret[0].code},name{ret[0].name}")
    database.close()
    return ret
