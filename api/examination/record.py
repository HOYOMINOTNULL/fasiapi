from typing import Optional

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel
import sqlite3 as sq
import config
import base64

app = APIRouter()

'''
examination_res表结构
    id: int
    type: text 数据库中是否有该人脸的个人信息，即该人脸是否已知，值为KNOWN或UNKNOWN
    face_id: int 该人脸在face_data表中的编号
    time: int 违规时间
'''

class RecordQuery(BaseModel):
    type: str
    code: Optional[str] = None
    name: Optional[str] = None
    time: str
    image: str

@app.get('/record/')
async def record_query() -> list[RecordQuery]:
    '''
    以时间降序获得所有违规记录
    '''
    ret = []
    database = sq.connect(config.FACE_DB_PATH)
    print("接收到record")
    cursor = database.cursor()
    cursor.execute('SELECT * FROM examination_res ORDER BY time DESC')
    res = cursor.fetchall()
    for item in res:
        type = item[1]
        time = item[3]
        data = item[-1]
        image = base64.b64encode(data).decode()
        if type == 'UNKNOWN':
            ret.append(RecordQuery(type=type, time=time, image=image))
        else:
            id = item[2]
            cursor.execute('SELECT code, name FROM face_data WHERE id=?', (id,))
            get = cursor.fetchone()
            code = get[0]
            name = get[1]
            ret.append(RecordQuery(type=type, code=code, name=name, time=time, image=image))
    database.close()
    return ret