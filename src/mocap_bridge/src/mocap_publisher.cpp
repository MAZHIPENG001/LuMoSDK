#include <rclcpp/rclcpp.hpp>
#include <mocap_bridge/msg/mocap_data.hpp>
#include <mocap_bridge/msg/marker.hpp>
#include <mocap_bridge/msg/rigid_body.hpp>
#include <thread>
#include <chrono>
#include <memory>
#include "LuMoSDKBase.hpp"   // 注意包含路径

class MocapPublisher : public rclcpp::Node {
public:
    MocapPublisher() : Node("mocap_publisher") {
        publisher_ = this->create_publisher<mocap_bridge::msg::MocapData>("mocap_data", 10);

        // 初始化 SDK 接收器（参考示例）
        receiver_ = lusternet::getFZReceive();
        receiver_->Init();
        std::string server_ip = "169.254.44.216";
        receiver_->Connect(server_ip);
        if (!receiver_->IsConnected()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to server at %s", server_ip.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Connected to mocap server.");

        // 启动一个单独的线程来接收数据（因为 ReceiveData 是阻塞的）
        running_ = true;
        receive_thread_ = std::thread(&MocapPublisher::receiveLoop, this);
    }

    ~MocapPublisher() {
        running_ = false;
        if (receive_thread_.joinable()) {
            receive_thread_.join();
        }
        if (receiver_->IsConnected()) {
            receiver_->Disconnect("169.254.44.216");
        }
        receiver_->Close();
    }

private:
    void receiveLoop() {
        lusternet::LusterMocapData mocap_data;
        while (rclcpp::ok() && running_) {
            try {
                receiver_->ReceiveData(mocap_data);  // 阻塞等待
                if (!rclcpp::ok()) break;

                auto msg = mocap_bridge::msg::MocapData();
                msg.frame_id = mocap_data.FrameID;
                msg.timestamp = mocap_data.TimeStamp;

                // 筛选 Marker ID == 1（足球）
                for (const auto& marker : mocap_data.Frame3DMarker) {
                    if (marker.MarkerID == 1) {
                        mocap_bridge::msg::Marker m;
                        m.marker_id = marker.MarkerID;
                        m.marker_name = marker.MarkerName;
                        m.x = marker.X;
                        m.y = marker.Y;
                        m.z = marker.Z;
                        msg.markers.push_back(m);
                        break;  // 只有一个
                    }
                }

                // 筛选 RigidBody ID == 4（相机）
                for (const auto& rigid : mocap_data.FrameRigidBody) {
                    if (rigid.RigidID == 4) {
                        mocap_bridge::msg::RigidBody rb;
                        rb.rigid_id = rigid.RigidID;
                        rb.rigid_name = rigid.RigidName;
                        rb.is_track = rigid.IsTrack;
                        rb.x = rigid.X;
                        rb.y = rigid.Y;
                        rb.z = rigid.Z;
                        rb.qx = rigid.qx;
                        rb.qy = rigid.qy;
                        rb.qz = rigid.qz;
                        rb.qw = rigid.qw;
                        msg.rigid_bodies.push_back(rb);
                        // break;
                    }
                    if (rigid.RigidID == 5) {
                        mocap_bridge::msg::RigidBody rb;
                        rb.rigid_id = rigid.RigidID;
                        rb.rigid_name = rigid.RigidName;
                        rb.is_track = rigid.IsTrack;
                        rb.x = rigid.X;
                        rb.y = rigid.Y;
                        rb.z = rigid.Z;
                        rb.qx = rigid.qx;
                        rb.qy = rigid.qy;
                        rb.qz = rigid.qz;
                        rb.qw = rigid.qw;
                        msg.rigid_bodies.push_back(rb);
                        break;
                    }
                }

                publisher_->publish(msg);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Receive error: %s", e.what());
            }
        }
    }

    rclcpp::Publisher<mocap_bridge::msg::MocapData>::SharedPtr publisher_;
    std::shared_ptr<lusternet::CReceiveBase> receiver_;
    std::thread receive_thread_;
    bool running_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MocapPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}