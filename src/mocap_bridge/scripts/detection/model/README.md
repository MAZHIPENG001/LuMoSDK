# 登录
```bash
modelscope login --token ms-***********************
```

# 模型
## 上传
```bash
modelscope upload \
  MaZp001/yolo_detection \
  "$HOME/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/detection/model" \
  --repo-type model
```
## 下载
```bash
modelscope download \
  ball/yolo26l_seg_768_b16_zed/best.pt \
  --model MaZp001/yolo_detection \
  --local_dir . \
  --repo-type model
```

# 数据
## 上传
```bash
modelscope upload \
  MaZp001/yolo_dataset \
  --local_dir "$HOME/GithubDoc/LuMoSDK/datasets/yolo_dataset" \
  --repo-type dataset

# 本地 train.zip 上传到仓库的 data/train.zip：
modelscope upload \
  MaZp001/yolo_dataset \
  "$HOME/datasets/train.zip" \
  "data/train.zip" \
  --repo-type dataset
  ```
## 下载
```bash
modelscope download \
  --dataset MaZp001/yolo_dataset \
  --local_dir "$HOME/datasets/yolo_dataset"
```