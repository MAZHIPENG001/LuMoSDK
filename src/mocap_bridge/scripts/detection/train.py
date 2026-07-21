from ultralytics import YOLO

# 加载预训练的分割模型
model_path="/home/ma/GithubDoc/ultralytics/my_model/pretrained_model/yolo26l-seg.pt"
model = YOLO(model_path)

# 开始训练
results = model.train(
    data='/home/ma/GithubDoc/ultralytics/my_model/BALL_data/data.yaml',
    epochs=1000,
    imgsz=640,
    batch=4,
    device=0,
    amp=False,
    save_period=100,
    project='ball_seg',
    name='seg_model_v1'
)

# from ultralytics import YOLO
# model_path = "/home/ma/GithubDoc/ultralytics/run/ball_seg/seg_model_v1/weights/last.pt"
# model = YOLO(model_path)
# # 开始续训
# results = model.train(
#     resume=True
# )