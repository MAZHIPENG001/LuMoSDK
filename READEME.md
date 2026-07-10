git clone git@github.com:MAZHIPENG001/LuMoSDK.git

pip install setuptools==59.6.0

colcon build --packages-select mocap_bridge --cmake-clean-cache
