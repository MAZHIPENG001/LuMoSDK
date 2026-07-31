# 图像数据工具

本目录提供 RealSense 图像采集和图片批量重命名工具。两个脚本都支持在命令后直接传入目标目录；目录既可以是绝对路径，也可以是相对路径。

## 1. 采集图像

运行 `data_save.py`，并将图片保存目录作为第一个参数传入：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/data_save.py /path/to/save_dir
```

例如，将图片保存到当前目录下的 `pic_red_ball`：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/data_save.py ./pic_red_ball
```

如果不传入目录，图片默认保存到 `data/pic_red_ball`：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/data_save.py
```

运行后可使用以下按键：

- `c`：开始采集，每 0.1 秒保存一张彩色图片。
- `s`：停止采集。
- `q`：退出程序。

图片按 `pic001.jpg`、`pic002.jpg` 等格式保存。如果目录中已有同格式图片，新图片会从当前最大编号之后继续编号。

## 2. 批量重命名

运行 `rename.py`，并将待处理的图片目录作为第一个参数传入：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/rename.py /path/to/image_dir
```

例如：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/rename.py ./pic_red_ball
```

如果不传入目录，默认处理当前工作目录下的 `pic`：

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/rename.py
```

脚本只处理名称符合 `pic数字.jpg` 格式的文件，并按原编号顺序重命名为 `pic0001.jpg`、`pic0002.jpg` 等连续编号。

## 3. 查看帮助

```bash
python3 /home/ma/GithubDoc/LuMoSDK/data/data_save.py --help
python3 /home/ma/GithubDoc/LuMoSDK/data/rename.py --help
```
