cmake_minimum_required(VERSION 3.8)
project(mocap_bridge)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 自定义消息文件
set(msg_files
  "msg/Marker.msg"
  "msg/RigidBody.msg"
  "msg/MocapData.msg"
)

# 生成消息接口
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs
)

# 添加可执行文件
add_executable(mocap_publisher src/mocap_publisher.cpp)

# 指定 SDK 头文件路径
target_include_directories(mocap_publisher
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/sdk/include
)

# 链接 ROS2 依赖
ament_target_dependencies(mocap_publisher
  rclcpp
)

# 直接指定 LuMoSDK 库的完整路径
set(LuMoSDK_LIB ${CMAKE_CURRENT_SOURCE_DIR}/sdk/lib/libLuMoSDK.so)

# 链接库（包括 LuMoSDK 和系统库）
target_link_libraries(mocap_publisher
  ${LuMoSDK_LIB}
  pthread
  dl
)

# 链接生成的消息库（使用推荐方式，忽略警告）
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mocap_publisher ${cpp_typesupport_target})

# 安装
install(TARGETS mocap_publisher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()