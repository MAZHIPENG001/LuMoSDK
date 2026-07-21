# 下载
```bash
git clone git@github.com:MAZHIPENG001/LuMoSDK.git

cd LuMoSDK
source /opt/ros/humble/setup.bash
# pip install setuptools==59.6.0
colcon build --packages-select mocap_bridge --cmake-clean-cache
```

# 1. 步骤
## 1.1 动捕数据作为ros话题发布
```bash
cd ~/GithubDoc/LuMoSDK
colcon build --packages-select mocap_bridge
source install/setup.bash
```
```bash
cd ~/GithubDoc/LuMoSDK
source install/setup.bash
export LD_LIBRARY_PATH=src/mocap_bridge/sdk/lib:$LD_LIBRARY_PATH
```
```bash
./install/mocap_bridge/lib/mocap_bridge/mocap_publisher
```
## 1.2 相机数据作为ros话题发布
```bash
cd ~/GithubDoc/ultralytics/my_model
python3 ~/GithubDoc/ultralytics/my_model/eval_ros.py
```
## 1.3 消息订阅
```bash
# 数据查看
cd ~/GithubDoc/LuMoSDK
source install/setup.bash
export LD_LIBRARY_PATH=src/mocap_bridge/sdk/lib:$LD_LIBRARY_PATH
cd ~/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/
python3 mocap_subscriber.py
```
```bash
# 数据保存
cd ~/GithubDoc/LuMoSDK
source install/setup.bash
export LD_LIBRARY_PATH=src/mocap_bridge/sdk/lib:$LD_LIBRARY_PATH
cd ~/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/
python3 data_save.py
```
## 1.4 绘图
```bash
python3 ~/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/plot_auto_calib.py --dir ***
python3 ~/GithubDoc/LuMoSDK/src/mocap_bridge/scripts/plot_auto_calib.py
```