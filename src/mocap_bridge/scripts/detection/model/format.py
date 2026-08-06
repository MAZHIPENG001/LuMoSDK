from ultralytics import YOLO

model_path = "./pic_zed_ball_seg/yolo26n_seg_768_b16/weights/best.pt"
model = YOLO(model_path)

# 导出为 TensorRT engine 格式
# 参数说明:
# format="engine": 指定输出格式为 TensorRT
# half=True: 启用 FP16 以提升推理速度
# device=0: 指定使用第一张 GPU 进行导出计算
# model.export(format="engine", half=True, device=0)
model.export(
    format="engine",
    imgsz=640,
    batch=1,
    dynamic=False,
    quantize=16,
    device=0,
)