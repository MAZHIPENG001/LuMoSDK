/***********************************************************************
*
* Copyright (c) 2021-2022 Luster LightTech(Beijing) Co.Ltd.
* All Rights Reserved.
*
*
* FILE NAME		: FZMotion_Receive_Sample.cpp
* DESCRIPTION	: SDK API SAMPLE.
*
* VERDION	: 1.0.0
* DATE		: 2022/04/10
* AUTHOR	: jzy
*
***********************************************************************/

#include <iostream>
#include <string> 
#include <thread>
#include <time.h>
#include <mutex>
#include <queue>
#include <map>
#include <fstream>
#include <atomic>
#include <array>
#include <vector>
#include <chrono>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include "LuMoSDKBase.hpp"

int main()
{
    // print version info
    uint8_t ver[4];
    ver[0] = 1; ver[1] = 1; ver[2] = 0; ver[3] = 7;
    printf("LuMoSDK(LuMoSDK ver. %d.%d.%d.%d)\n", ver[0], ver[1], ver[2], ver[3]);

    // Do asynchronous server discovery.
    printf("查找本机服务端.\n");
    printf("请输入发送端IP以连接服务端接收数据，示例：169.254.205.202\n");

    std::string IP = "169.254.44.216";
    // std::cin >> IP;

    // establish receiver
    std::shared_ptr<lusternet::CReceiveBase> LusterMotionData = lusternet::getFZReceive();

    // init receiver
    LusterMotionData->Init();

    // input IP, connect server
    LusterMotionData->Connect(IP);
    if (LusterMotionData->IsConnected())
    {
        printf("连接成功.\n");
    }
    // Get Camera Info
    lusternet::LusterCameraList CameraList;
    LusterMotionData->GetCameraList(CameraList);
    for (int i = 0; i < CameraList.vCameraList.size(); i++)
    {
        printf("CameraIp = %s.\n", CameraList.vCameraList[i].Ip.c_str());
        printf("CameraSerial = %s.\n", CameraList.vCameraList[i].Serial.c_str());
        printf("CameraModel = %s.\n", CameraList.vCameraList[i].Model.c_str());
        printf("CameraExposure = %d.\n", CameraList.vCameraList[i].Exposure);
        printf("CameraGain = %d.\n", CameraList.vCameraList[i].Gain);
        printf("CameraFrameRate = %d.\n", CameraList.vCameraList[i].FrameRate);
        printf("CameraMTU = %f.\n", CameraList.vCameraList[i].MTU);
        printf("CameraId = %d.\n", CameraList.vCameraList[i].Id);
    }
    // receive skel data
    lusternet::LusterMocapData MocapData;
    std::vector<std::array<float, 3>> trajectory;
    std::atomic<bool> bExit{false};

    // --- 丢帧平滑参数（坐标单位：mm，时间：s）---
    // 参照 robot_controller::MotionCaptureDataUpdate() 中的三层平滑机制移植
    constexpr int    kMaxMissFrames      = 10;       // 最多外推帧数（对应 kMaxObjectMissFrames）
    constexpr float  kOutlierBaseMM      = 50.0f;   // 离群点基础阈值 mm（对应 kObjectOutlierBaseM * 1000）
    constexpr float  kOutlierVelGainSec  = 0.05f;   // 速度相关离群阈值增益 s（对应 kObjectOutlierVelGainSec）
    constexpr float  kMaxMarkerSpeedMmS  = 5000.0f; // 速度饱和上限 mm/s（对应 kMaxObjectMarkerSpeed * 1000）
    constexpr float  kVelEmaAlpha        = 0.8f;    // 速度 EMA 系数（对应 kObjectMarkerEmaAlpha）
    constexpr double kFiniteDiffMaxDtSec = 0.1;     // 差分有效最大帧间隔 s
    constexpr float  kPosEmaAlpha        = 0.4f;    // 位置 EMA 系数（越小越平滑，延迟越大）

    // --- 平滑状态 ---
    bool has_prev = false;
    std::array<float, 3> prev_pos = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> last_vel = {0.0f, 0.0f, 0.0f};
    std::chrono::steady_clock::time_point prev_time;
    int miss_frames = 0;
    bool smoothed_pos_initialized = false;
    std::array<float, 3> smoothed_pos = {0.0f, 0.0f, 0.0f};

    // keyboard listener thread: press 'q' to exit
    std::thread keyThread([&]() {
        struct termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        while (!bExit) {
            char c = getchar();
            if (c == 'q' || c == 'Q') { bExit = true; break; }
        }
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    });
    keyThread.detach();

    printf("按 'q' 键退出并保存 Marker1 轨迹。\n");

    while (!bExit)
    {
        if (LusterMotionData->IsConnected())
        {
            // if no data comes, it will wait all the time
            LusterMotionData->ReceiveData(MocapData);

            // Frame ID
            uint32_t FrameID = MocapData.FrameID;
            printf("FrameID = %d.\n", FrameID); //打印帧ID

            //TimeStamp
            unsigned long long TimeStamp = MocapData.TimeStamp;
            printf("TimeStamp = %llu.\n", TimeStamp); //打印当前帧时间戳

            // //CameraSyncTime
            // unsigned long long CameraSyncTime = MocapData.uCameraSyncTime;
            // printf("CameraSyncTime = %llu.\n", CameraSyncTime);  //打印相机同步时间

            //BroadcastTime
            unsigned long long uBroadcastTime = MocapData.uBroadcastTime;
            printf("BroadcastTime = %llu.\n", uBroadcastTime); //打印数据广播时间

            // Marker Data
            std::vector<lusternet::LST_MarkerINFO> Frame3DMarker = MocapData.Frame3DMarker;
            bool marker1Found = false;
            std::array<float, 3> raw_pos = {0.0f, 0.0f, 0.0f};
            for (int i = 0; i < Frame3DMarker.size(); ++i)
            {
                printf("MarkerID = %d.\n", Frame3DMarker[i].MarkerID);
                printf("MarkerName = %s.\n", Frame3DMarker[i].MarkerName.c_str());
                printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", Frame3DMarker[i].X, Frame3DMarker[i].Y, Frame3DMarker[i].Z);
                if (Frame3DMarker[i].MarkerID == 1 && Frame3DMarker[i].MarkerName == "Marker1")
                {
                    raw_pos = {Frame3DMarker[i].X, Frame3DMarker[i].Y, Frame3DMarker[i].Z};
                    marker1Found = true;
                }
            }

            // --- 三层丢帧平滑（移植自 robot_controller::MotionCaptureDataUpdate）---
            const auto now = std::chrono::steady_clock::now();

            // Step A: 离群点拒绝 —— 与基于上一帧位置+速度的预测对比，偏差过大则降级为丢帧
            if (marker1Found && has_prev)
            {
                const double dt = std::chrono::duration<double>(now - prev_time).count();
                if (dt > 0.0 && dt <= kFiniteDiffMaxDtSec)
                {
                    // 预测位置
                    float pred_x = prev_pos[0] + last_vel[0] * static_cast<float>(dt);
                    float pred_y = prev_pos[1] + last_vel[1] * static_cast<float>(dt);
                    float pred_z = prev_pos[2] + last_vel[2] * static_cast<float>(dt);
                    float vel_norm = std::sqrt(last_vel[0]*last_vel[0] + last_vel[1]*last_vel[1] + last_vel[2]*last_vel[2]);
                    float thr = kOutlierBaseMM + kOutlierVelGainSec * vel_norm;
                    float dx = raw_pos[0] - pred_x;
                    float dy = raw_pos[1] - pred_y;
                    float dz = raw_pos[2] - pred_z;
                    if (std::sqrt(dx*dx + dy*dy + dz*dz) > thr)
                    {
                        printf("[Smooth] Marker1 outlier rejected (deviation > %.1f mm)\n", thr);
                        marker1Found = false;  // 降级为软丢帧
                    }
                }
                // dt > kFiniteDiffMaxDtSec: 跳过离群检测，允许长间隔后重新锚定
            }

            bool accepted  = false;
            bool soft_miss = false;

            if (marker1Found)
            {
                // Step B: 接受帧 —— 有限差分估计速度，EMA 平滑后更新锚点
                std::array<float, 3> fresh_vel = last_vel;
                if (has_prev)
                {
                    const double dt = std::chrono::duration<double>(now - prev_time).count();
                    if (dt > 1e-4 && dt <= kFiniteDiffMaxDtSec)
                    {
                        float vx = (raw_pos[0] - prev_pos[0]) / static_cast<float>(dt);
                        float vy = (raw_pos[1] - prev_pos[1]) / static_cast<float>(dt);
                        float vz = (raw_pos[2] - prev_pos[2]) / static_cast<float>(dt);
                        float speed = std::sqrt(vx*vx + vy*vy + vz*vz);
                        if (speed <= kMaxMarkerSpeedMmS)
                        {
                            // EMA 平滑
                            fresh_vel[0] = (1.0f - kVelEmaAlpha) * last_vel[0] + kVelEmaAlpha * vx;
                            fresh_vel[1] = (1.0f - kVelEmaAlpha) * last_vel[1] + kVelEmaAlpha * vy;
                            fresh_vel[2] = (1.0f - kVelEmaAlpha) * last_vel[2] + kVelEmaAlpha * vz;
                        }
                        // 速度超限：保持 last_vel
                    }
                    // dt 超出范围：保持 last_vel，用当前帧重新锚定
                }
                last_vel  = fresh_vel;
                prev_pos  = raw_pos;
                prev_time = now;
                has_prev  = true;
                miss_frames = 0;
                accepted = true;
                // 位置 EMA 平滑，消除传感器高频噪声
                if (!smoothed_pos_initialized) {
                    smoothed_pos = raw_pos;
                    smoothed_pos_initialized = true;
                } else {
                    smoothed_pos[0] = kPosEmaAlpha * raw_pos[0] + (1.0f - kPosEmaAlpha) * smoothed_pos[0];
                    smoothed_pos[1] = kPosEmaAlpha * raw_pos[1] + (1.0f - kPosEmaAlpha) * smoothed_pos[1];
                    smoothed_pos[2] = kPosEmaAlpha * raw_pos[2] + (1.0f - kPosEmaAlpha) * smoothed_pos[2];
                }
                trajectory.push_back(smoothed_pos);
            }
            else
            {
                miss_frames++;
                soft_miss = (miss_frames <= kMaxMissFrames) && has_prev;
            }

            if (!accepted)
            {
                if (soft_miss)
                {
                    // Step C: 软丢帧 —— 线性外推保持轨迹连续
                    const double elapsed = std::chrono::duration<double>(now - prev_time).count();
                    std::array<float, 3> extrap = {
                        smoothed_pos[0] + last_vel[0] * static_cast<float>(elapsed),
                        smoothed_pos[1] + last_vel[1] * static_cast<float>(elapsed),
                        smoothed_pos[2] + last_vel[2] * static_cast<float>(elapsed)
                    };
                    printf("[Smooth] Marker1 soft miss (frame %d/%d), extrapolating: X=%.2f Y=%.2f Z=%.2f\n",
                           miss_frames, kMaxMissFrames, extrap[0], extrap[1], extrap[2]);
                    trajectory.push_back(extrap);
                }
                else
                {
                    // Step D: 硬丢帧 —— 超出容忍范围，回退并重置历史
                    printf("[Smooth] Marker1 hard miss (frame %d), resetting history\n", miss_frames);
                    trajectory.push_back({0.0f, 0.0f, 0.0f});
                    has_prev    = false;
                    last_vel    = {0.0f, 0.0f, 0.0f};
                    miss_frames = 0;
                    smoothed_pos_initialized = false;
                }
            }

            // Rigid body Data
            std::vector<lusternet::LST_RIGID_DATA> FrameRigidBody = MocapData.FrameRigidBody;
            for (int i = 0; i < FrameRigidBody.size(); ++i)
            {
                if (FrameRigidBody[i].IsTrack)
                {
                    printf("RigidID = %d.\n", FrameRigidBody[i].RigidID);  //打印刚体ID
                    printf("RigidName = %s.\n", FrameRigidBody[i].RigidName.c_str()); //打印刚体名称
                    printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].X, FrameRigidBody[i].Y, FrameRigidBody[i].Z); //打印刚体坐标数据
                    printf("Angle: [QX] = %f, [QY] = %f, [QZ] = %f, [QW] = %f\n", FrameRigidBody[i].qx, FrameRigidBody[i].qy, FrameRigidBody[i].qz, FrameRigidBody[i].qw); //打印刚体姿态数据(四元数)
                    printf("Speed: [Speed] = %f, [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fSpeed, FrameRigidBody[i].fXSpeed, FrameRigidBody[i].fYSpeed, FrameRigidBody[i].fZSpeed);//打印刚体速度以及每个轴向的速度
                    printf("AcceleratedSpeed: [AcceleratedSpeed] = %f, [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fAcceleratedSpeed, FrameRigidBody[i].fXAcceleratedSpeed, FrameRigidBody[i].fYAcceleratedSpeed, FrameRigidBody[i].fZAcceleratedSpeed);//打印刚体加速度以及每个轴向的加速度
					printf("EulerAngle: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fXEulerAngle, FrameRigidBody[i].fYEulerAngle, FrameRigidBody[i].fZEulerAngle); //打印刚体欧拉角数据
                    printf("PALSTANCE: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fXPalstance, FrameRigidBody[i].fYPalstance, FrameRigidBody[i].fZPalstance); //打印刚体每个轴的角速度
                    printf("ACCPALSTANCE: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].AccfXPalstance, FrameRigidBody[i].AccfYPalstance, FrameRigidBody[i].AccfZPalstance); //打印刚体每个轴的角加速度
                }
                else
                {
                    printf("RigidID = %d track failed.\n", FrameRigidBody[i].RigidID);
                }

            }

        //     // Skeleton Data
        //     std::vector<lusternet::LST_BODY_DATA> FrameBodysPose = MocapData.FrameBodysPose;
        //     for (int i = 0; i < FrameBodysPose.size(); ++i)
        //     {
        //         if (FrameBodysPose[i].IsTrack)
        //         {
        //             printf("BodyID = %d.\n", FrameBodysPose[i].BodyID); //打印人体ID
        //             printf("BodyName = %s.\n", FrameBodysPose[i].BodyName.c_str()); //打印人体名称
        //             for (int j = 0; j < FrameBodysPose[i].vecJointNodes.size(); j++)
        //             {
        //                 printf("JointNodeID = %d.\n", FrameBodysPose[i].vecJointNodes[j].iJointID); //打印人体内骨骼ID
        //                 printf("JointNodeName = %s.\n", FrameBodysPose[i].vecJointNodes[j].sJointName.c_str()); //打印人体内骨骼名称
        //                 printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", FrameBodysPose[i].vecJointNodes[j].X, FrameBodysPose[i].vecJointNodes[j].Y, FrameBodysPose[i].vecJointNodes[j].Z); //打印人体内骨骼坐标数据
        //                 printf("Angle: [QX] = %f, [QY] = %f, [QZ] = %f, [QW] = %f\n", FrameBodysPose[i].vecJointNodes[j].qx, FrameBodysPose[i].vecJointNodes[j].qy, FrameBodysPose[i].vecJointNodes[j].qz, FrameBodysPose[i].vecJointNodes[j].qw); //打印人体内骨骼姿态数据(四元数)
        //             }
        //         }
        //         else
        //         {
        //             printf("BodyID = %d track failed.\n", FrameBodysPose[i].BodyID);
        //         }

        //     }
        //     //MarkerSet Data
        //     std::vector<lusternet::LST_MARKER_SET> markerSet = MocapData.FrameMarkerSet;
        //     for (int i = 0; i < markerSet.size(); i++)
        //     {
        //         printf("MarkerSetName = %s.\n", markerSet[i].sName.c_str()); //打印点集名称
        //         for (int j = 0; j < markerSet[i].vmarkers.size(); j++)
        //         {
        //             printf("MarkerID = %d.\n", markerSet[i].vmarkers[j].MarkerID); //打印点集内点ID
        //             printf("MarkerName = %s.\n", markerSet[i].vmarkers[j].MarkerName.c_str()); //打印点集内点的名称
        //             printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", markerSet[i].vmarkers[j].X, markerSet[i].vmarkers[j].Y, markerSet[i].vmarkers[j].Z); //打印点集内点的坐标数据
        //         }
        //     }
	    // // 测力台数据
        //     printf("ForceFlate Fx = %f.\n", MocapData.ForcePlateData.Fx); //打印测力台矢量力的分量：Fx
        //     printf("ForceFlate Fy = %f.\n", MocapData.ForcePlateData.Fy); //打印测力台矢量力的分量：Fy
        //     printf("ForceFlate Fz = %f.\n", MocapData.ForcePlateData.Fz); //打印测力台矢量力的分量：Fz
        //     printf("ForceFlate Mx = %f.\n", MocapData.ForcePlateData.Mx); //压心坐标：X
        //     printf("ForceFlate My = %f.\n", MocapData.ForcePlateData.My); //压心坐标：Y
        //     printf("ForceFlate Mz = %f.\n", MocapData.ForcePlateData.Mz); //压心坐标：Z
        //     printf("ForceFlate Lx = %f.\n", MocapData.ForcePlateData.Lx); //力矩
        //     printf("ForceFlate Lz = %f.\n", MocapData.ForcePlateData.Lz); //力矩
            
        //     // 时码信息
        //     printf("TimeCode Hours = %d.\n", MocapData.TimeCode.mHours); //打印时码：时
        //     printf("TimeCode Minutes = %d.\n", MocapData.TimeCode.mMinutes); //打印时码：分
        //     printf("TimeCode Seconds = %d.\n", MocapData.TimeCode.mSeconds); //打印时码：秒
        //     printf("TimeCode Frames = %d.\n", MocapData.TimeCode.mFrames); //打印时码：帧
        //     printf("TimeCode mSubFrame = %d.\n", MocapData.TimeCode.mSubFrame); //打印时码：子帧
		
	    // //肌电数据结构体
        //     std::map<std::string, double> emgData = MocapData.ElectromyographyData.EmgData;
        //     for (const auto& emgDataIter : emgData)
        //     {
        //         printf("ElectromyographyData EmgSN =  %s.\n", emgDataIter.first.c_str()); //打印肌电设备SN号
        //         printf("ElectromyographyData EmgData = %f.\n", emgDataIter.second); //打印肌电设备数据
        //     }
            
        //     //自定义骨骼信息
        //     std::vector<lusternet::LST_CUSTOM_SKELETON> FrameCustomSkeleton = MocapData.FrameCustomSkeleton;
        //     for (int i = 0; i < FrameCustomSkeleton.size(); ++i)
        //     {
		// 		printf("CustomSkeletonID = %d.\n", FrameCustomSkeleton[i].Id); //打印自定义人体ID
		// 		printf("CustomSkeletonName = %s.\n", FrameCustomSkeleton[i].sName.c_str()); //打印自定义人体名称
		// 		printf("CustomSkeletonType = %d.\n", FrameCustomSkeleton[i].type); //打印自定义骨骼类型
		// 		for (int j = 0; j < FrameCustomSkeleton[i].vJointData.size(); j++)
		// 		{
		// 			printf("CustomSkeletonJointNodeID = %d.\n", FrameCustomSkeleton[i].vJointData[j].iJointID); //打印自定义骨骼内骨骼ID
		// 			printf("CustomSkeletonJointNodeName = %s.\n", FrameCustomSkeleton[i].vJointData[j].sJointName.c_str()); //打印自定义骨骼内骨骼名称
		// 			printf("CustomSkeletonJointNodePose: [X] = %f, [Y] = %f, [Z] = %f\n", FrameCustomSkeleton[i].vJointData[j].X, FrameCustomSkeleton[i].vJointData[j].Y, FrameCustomSkeleton[i].vJointData[j].Z); //打印自定义骨骼坐标数据
		// 			printf("CustomSkeletonJointNodeAngle: [QX] = %f, [QY] = %f, [QZ] = %f, [QW] = %f\n", FrameCustomSkeleton[i].vJointData[j].qx, FrameCustomSkeleton[i].vJointData[j].qy, FrameCustomSkeleton[i].vJointData[j].qz, FrameCustomSkeleton[i].vJointData[j].qw); //打印自定义骨骼内骨骼姿态数据(四元数)
		// 			 printf("CustomSkeletonJointNodeConfidence = %f. \n", FrameCustomSkeleton[i].vJointData[j].fConfidence); //打印自定义骨骼内骨骼置信度
        //             printf("CustomSkeletonJointNodePoseAngle = [X] = %f, [Y] = %f, [Z] = %f\n", FrameCustomSkeleton[i].vJointData[j].fAngleX, FrameCustomSkeleton[i].vJointData[j].fAngleY, FrameCustomSkeleton[i].vJointData[j].fAngleZ); //打印自定义骨骼内骨骼置姿态角
		// 		}
        //     }
            printf("[Frame Over].\n");
        }
        else
        {
            printf("******connect failed.\n");
        }
    }
    

    // save Marker1 trajectory to CSV
    {
        std::ofstream csvFile("marker1_trajectory.csv");
        csvFile << "X,Y,Z\n";
        for (auto& p : trajectory)
            csvFile << p[0] << "," << p[1] << "," << p[2] << "\n";
        csvFile.close();
        printf("轨迹已保存至 marker1_trajectory.csv，共 %zu 帧。\n", trajectory.size());
        system("python3 visualize_trajectory.py marker1_trajectory.csv &");
    }

    // input IP and Port, disconnect server
    if (LusterMotionData->IsConnected())
    {
        LusterMotionData->Disconnect(IP);
    }

    // close receiver
    LusterMotionData->Close();

    printf("LuMoSDK Close.\n");
}
