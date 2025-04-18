# 工地安全帽检测系统

## 上手指南
### 开发前配置要求
你的电脑应有英伟达显卡，并且安装了cuda，否则运行起来将会非常缓慢。  
python版本最好为3.9，需要的各种库在requirements.txt里面。  

### 安装步骤  


##### 项目基本文件结构:
```python
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
    |       |       |------alert.mp3 【警报语音文件】
    |
    |----Smart_Construction 【主要存放模型训练相关文件】
    |----models 【存放要使用的模型】
    |----database 【存放数据库文件】
    |
    |----config.py 【项目配置文件】
    |----main.py 【服务器启动脚本】
```

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
      data: BLOB,  二进制图像数据
      encoding: text 人脸编码

    examination_res表结构
      id: int,
      type: text, 数据库中是否有该人脸的个人信息，即该人脸是否已知，值为KNOWN或UNKNOWN
      face_id: int, 该人脸在face_data表中的编号
      time: int, 违规时间
      image: BLOB 二进制图像数据，存储违规时的图像

    数据库部署:
```sql
    CREATE TABLE "face_data" (
        "id"	INTEGER NOT NULL,
        "face_type"	TEXT NOT NULL,
        "code"	TEXT,
        "name"	TEXT,
        "filename"	TEXT,
        "size"	INTEGER,
        "type"	TEXT,
        "data"	BLOB NOT NULL,
        "encoding"	TEXT NOT NULL,
        PRIMARY KEY("id" AUTOINCREMENT)
    );

    CREATE TABLE "examination_res" (
        "id"	INTEGER NOT NULL,
        "type"	TEXT NOT NULL,
        "face_id"	INTEGER NOT NULL,
        "time"	DATETIME NOT NULL,
        "image"	BLOB,
        PRIMARY KEY("id" AUTOINCREMENT)
    );
```

##### 主要实现：
```python

    ##face.py

    #类定义；
    class FaceData(BaseModel):
        code: str
        name: str
        image: str #将图像转为base64编码的字符串进行存储和传输


    #函数定义
    all_face_data() -> list[FaceData]:
        #返回数据库中所有已知的人脸数据

    upload_face(code: str = Form(), name: str = Form(), data: UploadFile = File()):
        #从前端接收一条人脸数据，并对其进行人脸识别与编码，然后存入数据库中

    query_face_data(query: FaceQuery) -> list[FaceData]:
        #人脸信息查询，从前端接收一个人脸查询类，优先按工号code进行查询；返回一条人脸数据，无效查询和未查到则返回None


    ##examination.py

    #全局变量
    conf_threshold = 0.5 #置信度阈值
    invalid_flag = False #是否检测到违规行为

    #函数定义
    frame(detr: myfunc.Detector, cap: cv.VideoCapture, request: Request):
        #从相机cap中获取图像，并通过模型Detr检测，产生帧
        #request为网页请求，当请求中断时函数停止运行
    
    main_examination(request: Request, index: int = Query(default=0, ge=0, lt=len(config.cameras))):
        #实现视频流的传输
        #接收一个查询参数index用作索引，选定使用哪个摄像头

    change_conf(v: float = Query(ge=0, le=1)):
        #接收前端传来的一个查询参数，修改全局变量conf_threshold，进而修改模型的置信度阈值

    alert() -> str:
        #读取全局变量invalid_flag，存在违规行为则返回'1'，否则返回'0'

    ##record.py

    #类定义
    class RecordQuery(BaseModel):
        type: str
        code: str | None = None
        name: str | None = None
        time: str
        image: str

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
        iou_threshold: float = 0.45  # 调整NMS阈值
        input_size: int = 640
        invalid_cls: int = 1
        database_path: str
        # 模型标签 {0: 'person', 1: 'head', 2: 'helmet'}

        def examine(self, image: np.ndarray, conf_threshold: float) -> tuple[np.ndarray, bool]:
            #使用YOLO模型对输入图像image进行检测，置信度阈值为conf_threshold，记录违规信息，最终返回画好框的图像与一个表示是否存违规现象的bool变量

        face_recognize(self, captures: list[np.ndarray]) -> list[list[tuple[int, int, int, int]]]:
            #从输入的多个图片中寻找人脸，如果找到就记录人脸在图片中位置，并加入列表，最后一并返回

        recordrecord(self, img: np.ndarray, locations: list[tuple[int, int, int, int]], res: list[Any]):
            #记录违规信息，img为违规记录图像，locations为违规人员脸部位置
            #函数先去数据库中比对该人脸，如果没有找到则以UNKNOWN的形式存储进数据库。
            # 接着查找此人的违规记录，如果当天已经被记过违规则不再记，否则记入违规信息

        multi_process(self, invalid_record: list[np.ndarray], invalid_faces: list[list[tuple[int, int, int, int]]], res: list[Any]):
            #多线程处理器，对存在的每一条违规记录创建一个线程调用record函数单独运行，以期提高运行速度
            #res中保存了数据库中已有的所有人脸的部分信息，供人脸比对使用，避免频繁查询数据库
```


#### YOLOV8数据集的处理







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

#### 模型训练

首先我们需要指定训练的".yaml"文件，这个文件在训练时不可缺失，它指定了你要识别的类别以及数据集的位置
```yaml
# 训练集和验证集的 labels 和 image 文件的位置
train: ./score/images/train
val: ./score/images/val

# number of classes
nc: 2

# class names
names: [ 'head', 'helmet']
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

#### YOLOV8模型推理及使用
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







#### 智能问答机器人模型数据集准备

本次微调使⽤的数据集是  **jnmil/safety**，该数据集为本小组制作的纯文本，包含问题、复杂推理链和回答三个部分，将数据保存为 JSON 文件上传到 Hugging Face Hub，总共约500条记录。部分数据见下表。

| Questions                    | Complex_CoT                                                  |                           Response                           |
| :--------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------: |
| 安全帽的有效使用期限是多少？ | 材料老化速率 → 使用频率 → 环境暴露（紫外线/化学品） → 制造商建议 | 一般为3至5年，具体视材料和使用环境而定（参考GB 2811-2019标准） |
| 安全帽是否可以进行二次使用？ | 受过冲击的帽体结构完整性 → 内衬吸能性能 → 安全性能评估       |        受过重大冲击的安全帽应立即更换，不建议二次使用        |
| 安全帽的正确佩戴方法是什么？ | 调整帽衬至适合头型 → 保持帽体水平 → 系紧下颌带               |            确保帽体稳固不晃动，下颌带贴合但不压迫            |
| 安全帽的清洁方法有哪些？     | 使用中性清洁剂 → 避免强酸碱 → 自然晾干                       |             用温水和中性洗涤剂清洗，避免使用溶剂             |
| 安全帽的存储条件有哪些要求？ | 避免阳光直射 → 避免高温潮湿 → 避免化学品接触                 |          存放在干燥、阴凉、通风的环境中，远离化学品          |
| 安全帽的颜色是否有特定含义？ | 行业惯例 → 职位区分 → 可视性要求                             |            颜色可用于区分职位或工种，但无强制标准            |



本次微调的基座模型是  **DeepSeek-R1-Distill-Qwen-1.5B**

训练得到的模型地址为  **jnmil/DeepSeek-R1-Construction-Safety-f16**

上传的gguf模型地址为  jnmil/DeepSeek-R1-Construction-Safety-f16-GGUF

```python
# === 步骤 1：加载模型和分词器 ===
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # 基座模型，1.5B 参数，适合微调
    max_seq_length=2048,  # 最长序列长度，支持长上下文输入
    load_in_4bit=True,  # 4 位量化，降低内存占用（约 2-3 GB），适合 Colab T4
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,  # 使用 bfloat16（若支持）以提高精度
    trust_remote_code=True,  # 允许加载 DeepSeek 的自定义代码
)
```

完善提示词模版

```python
# 定义提示模板，使用中文注释以清晰表达意图
train_prompt_style = """以下是一个任务描述和输入上下文的指令。请撰写一个合适的回答。在回答之前，仔细思考并构建逐步推理链，确保回答逻辑清晰、准确。

### Instruction：
您是建筑安全领域的专家，特别是在安全帽使用方面。请回答以下问题。

### Question：
{}
### Response：
<think>
{}
</think>
{}"""
```

格式化数据集，提高批处理效率

```python
# 格式化数据集函数，将问题、推理链和回答组合为完整文本
def formatting_prompts_func(examples):
    inputs = examples["Question"]  # 问题字段
    cots = examples["Complex_CoT"]  # 推理链字段
    outputs = examples["Response"]  # 回答字段
    texts = []
    for input, cot, output in zip(inputs, cots, outputs):
        # 按模板格式化，添加 EOS 标记以结束序列
        text = train_prompt_style.format(input, cot, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}
# 应用格式化，批处理提高效率
dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
```



#### 大模型训练与微调

对模型进行调参，使用unsloth加速

```python
# === 步骤 3：配置 LoRA 微调 ===
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA 秩，增加到 32 增强适配能力，权衡内存（约增加 0.5 GB）
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力模块
        "gate_proj", "up_proj", "down_proj"  # MLP 模块
    ],  # 针对 Transformer 核心层，全面优化
    lora_alpha=32,  # 缩放因子，与 r 匹配，增强 LoRA 效果
    lora_dropout=0,  # 无 dropout，微调无需正则化
    bias="none",  # 无偏置，减少参数量
    use_gradient_checkpointing="unsloth",  # Unsloth 优化检查点，降低 30% 内存
    random_state=3407,  # 固定随机种子，确保可重复性
    use_rslora=True,  # 使用 Rank-Stabilized LoRA，稳定训练
    loftq_config=None,  # 不使用 LoFTQ，保持标准 LoRA
)
```

定义训练参数

```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,        # 单条样本的最大token长度（根据GPU显存调整，典型值：512-4096）
    dataset_num_proc=4,			# 数据预处理的并行进程数
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,     # 每个GPU的批大小
        gradient_accumulation_steps=2,     # 梯度累积步数
        learning_rate=1e-4,     		   # 初始学习率
        max_steps=1000,					   # 最大训练步数（替代epoch的更灵活控制方式）
        warmup_steps=10,				   # 学习率预热步数
        lr_scheduler_type="cosine",        # 学习率调度策略
        optim="adamw_8bit",				   # 优化器
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),	 # 自动选择浮点精度
        bf16=torch.cuda.is_bf16_supported(),		 # Ampere架构以上GPU启用bf16
        logging_steps=10,					# 每隔多少步记录日志（监控loss/学习率变化）
        save_steps=200,
        output_dir="outputs",
        save_total_limit=2,  
        seed=3407,							# 随机种子
    ),
)

```

定义权重文件精度

```python
!python /content/llama.cpp/convert_hf_to_gguf.py /content/DeepSeek-R1-Construction-Safety-merged-f16 --outfile /content/DeepSeek-R1-Construction-Safety-f16.gguf --outtype f16
#指定输出guff的权重类型为半精度浮点数（f16）
```



#### 评估

损失值的下降趋势表明模型在训练过程中逐渐收敛，梯度范数在训练初期有较大的波动，随后逐渐趋于稳定。

最后采用cpolar结合ollama实现远程机器的模型调用


### 使用指南
#### 配置模型路径
在根目录里的config.py中保存着一些全局变量
```python
import  os
PROJECT_PATH=r"E:\safety_helmet\safety_helmet_project"#项目根路径
YOLO_MODEL_PATH=os.path.join(PROJECT_PATH,r"models/best.pt")#yolo模型路径
FACE_DB_PATH=os.path.join(PROJECT_PATH,r"datebase/database.db")#数据库路径
cameras=[0]#摄像头id

```
通过修改模型路径，数据库路径，摄像头id可以使用不同的模型，数据库，摄像头
#### 配置网络地址
```html
const BASE_URL = "http://127.0.0.1:8000";
const response = await fetch("http://192.168.175.195:11434/api/generate", {
```
在前端中,通过修改这两个ip地址，即可连接到不同的后端服务器和大模型服务器

#### 启动应用
配置完成后，即可通过命令行来启动服务器。
```bash
fastapi run main.py 
```


### 项目主要负责人
[HOYOMINOTNULL](https://github.com/HOYOMINOTNULL/)

### 贡献者
[falunwen123go](https://github.com/falunwen123go/)
[papase](https://github.com/papase/)
[For-Sakura](https://github.com/For-Sakura)

### 开源协议
MIT License

Copyright (c) [2025] [HOTOMINOTNULL]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.