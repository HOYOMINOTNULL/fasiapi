# 工地安全帽检测系统

## 上手指南
### 开发前配置要求
你的电脑应有英伟达显卡，并且安装了cuda，否则运行起来将会非常缓慢。  
python版本最好为3.9，需要的各种库在requirements.txt里面。  

### 安装步骤  

#### 数据集的处理
首先我的数据集用的是 "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset" 它的数据集结构如下：  

---VOC2028
* ---Annotations    
* ---ImageSets    
* ---JPEGImages     
  
所以要使用 "Smart_Construction\data\gen_data"中的gen_head_helmet.py将原格式转换成yolov8的格式，如下：

---VOC2028
* ---images
  * ---train
  * ---test
* ---lables
  * ---train
  * ---test

调整一下路径，即可运行该程序
```python
FILE_ROOT = Path(r"E:\AI_Project\AI_Learning\Dataset")# 原始数据集
IMAGE_SET_ROOT = FILE_ROOT.joinpath(r"VOC2028\ImageSets\Main")  # 图片区分文件的路径
IMAGE_PATH = FILE_ROOT.joinpath(r"VOC2028\JPEGImages")  # 图片的位置
ANNOTATIONS_PATH = FILE_ROOT.joinpath(r"VOC2028\Annotations")  # 数据集标签文件的位置
LABELS_ROOT = FILE_ROOT.joinpath(r"VOC2028\Labels")  # 进行归一化之后的标签位置
# YOLO 需要的数据集形式的新数据集
DEST_IMAGES_PATH = Path(r"Safety_Helmet_Train_dataset\score\images")  # 区分训练集、测试集、验证集的图片目标路径
DEST_LABELS_PATH = Path(r"Safety_Helmet_Train_dataset\score\labels")  # 区分训练集、测试集、验证集的标签文件目标路径
```
然后增加安全帽子数据集的分类（这步可以略过）因为原有的数据集的分类只有两个，即"head"和"helmet"两个分类
，要增加person分类的话，要先用一遍yolo官方模型，来识别数据集中的每个图片，它会自动给出图片中每个person
的方框坐标，并保存在txt文件内，这个时候我们使用"Smart_Construction\data\gen_data"中的“merge_data.py”，
将输出的txt文件与原有数据集的labels进行合并，

```python
YOLOV5_LABEL_ROOT = f"E:\AI_Project\Smart_Construction_Project\inference\output\\"  # yolov5 导出的推理图片的 txt
DATASET_LABEL_ROOT = f"E:\AI_Project\AI_Learning\Dataset\VOC2028\Labels\\"  # 数据集的路径
```
#### 模型训练

首先我们需要指定训练的".yaml"文件，这个文件在训练时不可缺失，它指定了你要识别的类别以及数据集的位置
```yaml
# 训练集和验证集的 labels 和 image 文件的位置
train: ./score/images/train
val: ./score/images/val

# number of classes
nc: 3

# class names
names: ['person', 'head', 'helmet']
```
训练文件位于"Smart_Construction/new_train.py"上
```python
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8n.pt")#这里使用预训练的模型，可以大大减小模型训练的时间
    results = model.train(data=r"/root/.jupyter/project/Smart_Construction/data/custom_data.yaml",#data就是我们刚才写的yaml文件
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
```
训练后的结果会自动保存在"Smart_Construction\runs\"文件夹里面，可以查看训练的一些参数。

#### 模型推理及使用
采用opencv读取摄像头和，并交给模型推理图片，文件位于"Smart_Construction/acquire_frame__deal.py"上
请注意里面有几处引用，项目目录下的config.py存放着一些全局变量，如yolo模型位置，相机id号列表，
通过修改yolo模型位置，可以使用不同模型，通过修改相机id号列表，可以同时读取并处理多个摄像头传来的画面
```python
#acquire_frame__deal.py
from config import YOLO_MODEL_PATH,cameras

#config.py
PROJECT_PATH=r"E:\safety_helmet\safety_helmet_project"
YOLO_MODEL_PATH=os.path.join(PROJECT_PATH,r"Smart_Construction\models\ultimate_model\weights\best.pt")
FACE_MODEL_PATH=os.path.join(PROJECT_PATH,r"Smart_Construction/weights/helmet_head_person_l.pt")
FACE_DB_PATH=os.path.join(PROJECT_PATH,r"Smart_Construction/faces.db")

```


config.py
全局变量保存位置 



                                 YOLO V8 安全帽识别模型
模型位置:   模型及相关参数文件保存在"Smart_Construction/models/ultimate_model"下面


训练:   训练文件位于"Smart_Construction/new_train.py"上，有关参数调整请参考"https://blog.csdn.net/qq_37553692/article/details/130910432"

推理:   推理文件在位于"Smart_Construction/detect.ipynb"上，第一个cell是多线程，第二个cell是单线程处理


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