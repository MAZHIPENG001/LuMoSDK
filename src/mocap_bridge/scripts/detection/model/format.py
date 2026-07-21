from ultralytics import YOLO

model_path = "/home/ma/GithubDoc/ultralytics/my_model/model/ball/yolo26l-seg/best.pt"
model = YOLO(model_path)

# 导出为 TensorRT engine 格式
# 参数说明:
# format="engine": 指定输出格式为 TensorRT
# half=True: 启用 FP16 以提升推理速度
# device=0: 指定使用第一张 GPU 进行导出计算
model.export(format="engine", half=True, device=0)