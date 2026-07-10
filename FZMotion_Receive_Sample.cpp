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
#include "LuMoSDKBase.hpp"

int main()
{
    // print version info
    uint8_t ver[4];
    ver[0] = 1; ver[1] = 1; ver[2] = 0; ver[3] = 7;
    printf("LuMoSDK(LuMoSDK ver. %d.%d.%d.%d)\n", ver[0], ver[1], ver[2], ver[3]);

    // Do asynchronous server discovery.
    printf("查找本机服务端.\n");
    printf("请输入发送端IP以连接服务端接收数据，示例：169.254.44.216\n");

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
    bool bExit = false;
    while (1)
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
            // Soccer: MarkerID == 1 
            // std::vector<lusternet::LST_MarkerINFO> Frame3DMarker = MocapData.Frame3DMarker;
            // for (int i = 0; i < Frame3DMarker.size(); ++i)
            // {
            //     printf("MarkerID = %d.\n", Frame3DMarker[i].MarkerID);
            //     printf("MarkerName = %s.\n", Frame3DMarker[i].MarkerName.c_str());
            //     printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", Frame3DMarker[i].X, Frame3DMarker[i].Y, Frame3DMarker[i].Z);
            // }
            std::vector<lusternet::LST_MarkerINFO> Frame3DMarker = MocapData.Frame3DMarker;
            for (int i = 0; i < Frame3DMarker.size(); ++i)
            {
                if (Frame3DMarker[i].MarkerID == 1)
                {
                    printf("足球 >> MarkerID = 1, Name = %s, Pose: [X]=%f, [Y]=%f, [Z]=%f\n",
                        Frame3DMarker[i].MarkerName.c_str(),
                        Frame3DMarker[i].X, Frame3DMarker[i].Y, Frame3DMarker[i].Z);
                }
            }

            // Rigid body Data
            // Camera: RigidID == 4 
            // std::vector<lusternet::LST_RIGID_DATA> FrameRigidBody = MocapData.FrameRigidBody;
            // for (int i = 0; i < FrameRigidBody.size(); ++i)
            // {
            //     if (FrameRigidBody[i].IsTrack)
            //     {
            //         printf("RigidID = %d.\n", FrameRigidBody[i].RigidID);  //打印刚体ID
            //         printf("RigidName = %s.\n", FrameRigidBody[i].RigidName.c_str()); //打印刚体名称
            //         printf("Pose: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].X, FrameRigidBody[i].Y, FrameRigidBody[i].Z); //打印刚体坐标数据
            //         printf("Angle: [QX] = %f, [QY] = %f, [QZ] = %f, [QW] = %f\n", FrameRigidBody[i].qx, FrameRigidBody[i].qy, FrameRigidBody[i].qz, FrameRigidBody[i].qw); //打印刚体姿态数据(四元数)
            //         printf("Speed: [Speed] = %f, [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fSpeed, FrameRigidBody[i].fXSpeed, FrameRigidBody[i].fYSpeed, FrameRigidBody[i].fZSpeed);//打印刚体速度以及每个轴向的速度
            //         printf("AcceleratedSpeed: [AcceleratedSpeed] = %f, [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fAcceleratedSpeed, FrameRigidBody[i].fXAcceleratedSpeed, FrameRigidBody[i].fYAcceleratedSpeed, FrameRigidBody[i].fZAcceleratedSpeed);//打印刚体加速度以及每个轴向的加速度
			// 		   printf("EulerAngle: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fXEulerAngle, FrameRigidBody[i].fYEulerAngle, FrameRigidBody[i].fZEulerAngle); //打印刚体欧拉角数据
            //         printf("PALSTANCE: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].fXPalstance, FrameRigidBody[i].fYPalstance, FrameRigidBody[i].fZPalstance); //打印刚体每个轴的角速度
            //         printf("ACCPALSTANCE: [X] = %f, [Y] = %f, [Z] = %f\n", FrameRigidBody[i].AccfXPalstance, FrameRigidBody[i].AccfYPalstance, FrameRigidBody[i].AccfZPalstance); //打印刚体每个轴的角加速度
            //     }
            //     else
            //     {
            //         printf("RigidID = %d track failed.\n", FrameRigidBody[i].RigidID);
            //     }
            std::vector<lusternet::LST_RIGID_DATA> FrameRigidBody = MocapData.FrameRigidBody;
            for (int i = 0; i < FrameRigidBody.size(); ++i)
            {
                if (FrameRigidBody[i].RigidID == 4)
                {
                    if (FrameRigidBody[i].IsTrack)
                    {
                        printf("相机 >> RigidID = 4, Name = %s, Pose: [X]=%f, [Y]=%f, [Z]=%f\n",
                            FrameRigidBody[i].RigidName.c_str(),
                            FrameRigidBody[i].X, FrameRigidBody[i].Y, FrameRigidBody[i].Z);
                        printf("Angle: [QX] = %f, [QY] = %f, [QZ] = %f, [QW] = %f\n", 
                            FrameRigidBody[i].qx, FrameRigidBody[i].qy, 
                            FrameRigidBody[i].qz, FrameRigidBody[i].qw);
                    }
                    else
                    {
                        printf("RigidID = 4 track failed.\n");
                    }
                }
            }

            printf("[Frame Over].\n\n");
        }
        else
        {
            printf("******connect failed.\n");
        }
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
