                            config.py
全局变量保存位置


                                 YOLO V8 安全帽识别模型
模型位置:   模型及相关参数文件保存在"Smart_Construction/models/ultimate_model"下面


训练:   训练文件位于"Smart_Construction/new_train.py"上，有关参数调整请参考"https://blog.csdn.net/qq_37553692/article/details/130910432"

推理:   推理文件在位于"Smart_Construction/detect.ipynb"上，第一个cell是多线程，第二个cell是单线程处理
 


                                           人脸识别模块

database.py：将所有图片中未佩戴安全帽且可被提取人脸特征的人，将信息全部录入数据库face.db中（使用github训练好且显卡带得动的l型权重模型，识别效果较差，后续可用训练好的模型）

```python
if __name__ == "__main__":
    # 调整 PyTorch CUDA 内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    detector = HelmetFaceDetector(
        weights="weights/helmet_head_person_s.pt",
        db_path="Smart_Construction/faces.db"
    )
    detector.detect(
        "D:/PyCharm/Deep_learning/safetyhelmet/VOC2028/JPEGImages"
    )
```

在此处修改权重文件，数据库位置，以及侦测的图片位置

search：将指定图片中的未佩戴安全帽人的人脸特征放入数据库中比对，并打印该人信息。

```python
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    detector = HelmetFaceDetector(weights="weights/helmet_head_person_l.pt", db_path="faces.db")
    detector.detect("D:/PyCharm/Deep_learning/safetyhelmet/VOC2028/JPEGImages/000009.jpg")
    conn = sqlite3.connect("faces.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, employee_id, position, name, strftime('%Y-%m-%d %H:%M:%S', violation_time) FROM face_data")
    conn.close()
```

在此处修改权重文件，数据库位置，以及侦测的图片位置

修改文件位置，所有操作均在代码底部