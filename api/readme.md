##### 项目基本文件结构:
SAFETY_HELMET_PROJECT 【项目根目录】
    |------api 【路由列表】
    |       |------examination
    |       |           |------__init__.py
    |       |           |------examination.py 【路由脚本1 负责检测和传输相机获取的图像】
    |       |           |------record.py 【路由脚本2 负责违规记录的查询】
    |       |           |------function.py 【主要存放功能函数，包括使用模型预测、人脸识别等等】
    |       |
    |       |------facedata
    |       |           |------__init__.py
    |       |           |------face.py 【路由脚本 主要负责人脸信息数据库的查询与添加】
    |       |
    |       |------static 【静态文件】
    |       |       |------index.html 【前端文件】
    |
    |----Smart_Construction 【主要存放模型相关文件】
    |
    |----database.db 【数据库文件】
    |----config.py 【项目配置文件】
    |----main.py 【服务器启动脚本】


##### 数据库基本结构：
    数据库包含两个表:
        face_data   【用于存放人脸信息】
        examination_res 【用于存放检测到的违规记录】【一人一天最多会被记一次】

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

    examination_res表结构
    id: int
    type: text 数据库中是否有该人脸的个人信息，即该人脸是否已知，值为KNOWN或UNKNOWN
    face_id: int 该人脸在face_data表中的编号
    time: int 违规时间

##### 主要实现：
```python

    ##face.py

    #类定义；
    class FaceQuery(BaseModel):
        code: str | None = None
        name: str | None = None

    class FaceData(BaseModel):
        code: str
        name: str
        data: UploadFile

    #函数定义
    all_face_data() -> list[FaceData]:
        #返回数据库中所有已知的人脸数据

    upload_face(data: FaceData):
        #从前端接收一条人脸数据，并对其进行人脸识别与编码，然后存入数据库中

    query_face_data(query: FaceQuery) -> FaceData | None:
        #人脸信息查询，从前端接收一个人脸查询类，优先按工号code进行查询；返回一条人脸数据，无效查询和未查到则返回None


    ##examination.py

    #函数定义
    frame(detr: myfunc.Detector, cap: cv.VideoCapture, request: Request):
        #从相机cap中获取图像，并通过模型Detr检测，产生帧
        #request为网页请求，当请求中断时函数停止运行
    
    main_examination(request: Request, index: int = Query(default=0, ge=0, lt=len(config.cameras))):
        #实现视频流的传输
        #接收一个查询参数index用作索引，选定使用哪个摄像头


    ###record.py

    #类定义
    class RecordQuery(BaseModel):
        type: str
        code: str | None = None
        name: str | None = None
        time: str

    #函数定义
    record_query() -> list[RecordQuery]:
        #返回数据库中所有违规记录


    ##function.py

    EncodingToStr(encoding: np.ndarray) -> str:
        #将人脸编码转化为数据库可以存储的字符串类

    StrToEncoding(s: str) -> np.ndarray:
        #将从数据库中读出的字符串形式的人脸编码转化为ndarray类型

    class Detector():
        model: YOLO
        conf_threshold: float = 0.25  # 进一步降低置信度阈值
        iou_threshold: float = 0.45  # 调整NMS阈值
        input_size: int = 640
        invalid_cls: int = 1
        database_path: str
        # 模型标签 {0: 'person', 1: 'head', 2: 'helmet'}

        examine(self, image: np.ndarray) -> np.ndarray:
            #使用YOLO模型对输入图像image进行检测，记录违规信息，最终返回画好框的图像

        face_recognize(self, captures: list[np.ndarray]) -> list[np.ndarray]:
            #从输入的多个图片中寻找人脸，如果找到就对该人脸进行裁剪并加入列表，最后一并返回

        record(self, img: np.ndarray):
            #记录违规信息，img为违规人员人脸图像
            #函数先去数据库中比对该人脸，如果没有找到则以UNKNOWN的形式存储进数据库。
            # 接着查找此人的违规记录，如果当天已经被记过违规则不再记，否则记入违规信息