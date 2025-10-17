#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

// 聚类数据的结构体，包含了是否被选择，是否被处理等标识
struct ClusterData
{
    int index = -1;
    int selected = -1;
    int seeded = -1;
    int processed = -1;
    float velocity = 0.0;
    ClusterData() : index(-1), selected(-1), seeded(-1), processed(-1), velocity(0.0) {}
    ClusterData(int ind, int sel, int sed, int pro, float vel) : index(ind), selected(sel), seeded(sed), processed(pro), velocity(vel) {}
};

typedef pcl::PointXYZI PointType;
typedef PointXYZIRPYT  PointTypePose;

//参数读取
class ParamServer
{
public:
    ros::NodeHandle nh;

    std::string robot_id;

    // 话题
    string pointCloudTopic; // points_raw 原始点云数据
    string imuTopic;        // imu_raw 对应park数据集，imu_correct对应outdoor数据集，都是原始imu数据，不同的坐标系表示
    string odomTopic;       // odometry/imu，imu里程计，imu积分计算得到
    string gpsTopic;        // odometry/gps，gps里程计

    // 坐标系
    string lidarFrame;    // 激光坐标系
    string baselinkFrame; // 载体坐标系
    string odometryFrame; // 里程计坐标系
    string mapFrame;      // 世界坐标系

    // GPS参数
    bool useImuHeadingInitialization; //
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // 保存PCD
    bool savePCD;            // 是否保存地图
    string savePCDDirectory; // 保存路径

    // 激光传感器参数
    int row;          // 扫描线数，例如16、64
    int column;    // 扫描一周计数，例如每隔0.2°扫描一次，一周360°可以扫描1800次
    int downsampleRate;  // 扫描线降采样，跳过一些扫描线
    float lidarMinRange; // 最小范围
    float lidarMaxRange; // 最大范围

    // IMU参数
    float imuAccNoise; // 加速度噪声标准差
    float imuGyrNoise; // 角速度噪声标准差
    float imuAccBiasN; //
    float imuGyrBiasN;
    float imuGravity; // 重力加速度
    float imuRPYWeight;
    vector<double> extRotV;
    vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;   // xyz坐标系旋转
    Eigen::Matrix3d extRPY;   // RPY欧拉角的变换关系
    Eigen::Vector3d extTrans; // xyz坐标系平移
    Eigen::Quaterniond extQRPY;

    // LOAM
    float edgeThreshold;
    float surfThreshold;
    int edgeFeatureMinValidNum;
    int surfFeatureMinValidNum;

    // 下采样阈值
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize;

    float z_tollerance;
    float rotation_tollerance;

    // CPU Params
    int numberOfCores;
    double mappingProcessInterval;

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;
    float surroundingkeyframeAddingAngleThreshold;
    float surroundingKeyframeDensity;
    float surroundingKeyframeSearchRadius;

    // Loop closure
    bool loopClosureEnableFlag;
    float loopClosureFrequency;
    int surroundingKeyframeSize;
    float historyKeyframeSearchRadius;
    float historyKeyframeSearchTimeDiff;
    int historyKeyframeSearchNum;
    float historyKeyframeFitnessScore;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        
        nh.param<std::string>("/robot_id", robot_id, "roboat");

        // 从param server中读取key为"v_losm/pointCloudTopic"对应的参数，存pointCloudTopic，第三个参数是默认值
        // launch文件中定义<rosparam file="$(find v_losm)/config/params.yaml" command="load" />，从yaml文件加载参数
        nh.param<std::string>("v_losm/pointCloudTopic", pointCloudTopic, "points_raw");
        nh.param<std::string>("v_losm/imuTopic", imuTopic, "imu_correct");
        nh.param<std::string>("v_losm/odomTopic", odomTopic, "odometry/imu");
        nh.param<std::string>("v_losm/gpsTopic", gpsTopic, "odometry/gps");

        nh.param<std::string>("v_losm/lidarFrame", lidarFrame, "base_link");
        nh.param<std::string>("v_losm/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("v_losm/odometryFrame", odometryFrame, "odom");
        nh.param<std::string>("v_losm/mapFrame", mapFrame, "map");

        nh.param<bool>("v_losm/useImuHeadingInitialization", useImuHeadingInitialization, false);
        nh.param<bool>("v_losm/useGpsElevation", useGpsElevation, false);
        nh.param<float>("v_losm/gpsCovThreshold", gpsCovThreshold, 2.0);
        nh.param<float>("v_losm/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("v_losm/savePCD", savePCD, false);
        nh.param<std::string>("v_losm/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        std::string sensorStr;
        nh.param<std::string>("v_losm/sensor", sensorStr, "");

        nh.param<int>("v_losm/row", row, 16);
        nh.param<int>("v_losm/column", column, 1800);
        nh.param<int>("v_losm/downsampleRate", downsampleRate, 1);
        nh.param<float>("v_losm/lidarMinRange", lidarMinRange, 1.0);
        nh.param<float>("v_losm/lidarMaxRange", lidarMaxRange, 1000.0);

        nh.param<float>("v_losm/imuAccNoise", imuAccNoise, 0.01);
        nh.param<float>("v_losm/imuGyrNoise", imuGyrNoise, 0.001);
        nh.param<float>("v_losm/imuAccBiasN", imuAccBiasN, 0.0002);
        nh.param<float>("v_losm/imuGyrBiasN", imuGyrBiasN, 0.00003);
        nh.param<float>("v_losm/imuGravity", imuGravity, 9.80511);
        nh.param<float>("v_losm/imuRPYWeight", imuRPYWeight, 0.01);
        nh.param<vector<double>>("v_losm/extrinsicRot", extRotV, vector<double>());
        nh.param<vector<double>>("v_losm/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("v_losm/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);
        
        nh.param<float>("v_losm/edgeThreshold", edgeThreshold, 0.1);
        nh.param<float>("v_losm/surfThreshold", surfThreshold, 0.1);
        nh.param<int>("v_losm/edgeFeatureMinValidNum", edgeFeatureMinValidNum, 10);
        nh.param<int>("v_losm/surfFeatureMinValidNum", surfFeatureMinValidNum, 100);

        nh.param<float>("v_losm/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        nh.param<float>("v_losm/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        nh.param<float>("v_losm/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        nh.param<float>("v_losm/z_tollerance", z_tollerance, FLT_MAX);
        nh.param<float>("v_losm/rotation_tollerance", rotation_tollerance, FLT_MAX);

        nh.param<int>("v_losm/numberOfCores", numberOfCores, 2);
        nh.param<double>("v_losm/mappingProcessInterval", mappingProcessInterval, 0.15);

        nh.param<float>("v_losm/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        nh.param<float>("v_losm/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        nh.param<float>("v_losm/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        nh.param<float>("v_losm/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        nh.param<bool>("v_losm/loopClosureEnableFlag", loopClosureEnableFlag, false);
        nh.param<float>("v_losm/loopClosureFrequency", loopClosureFrequency, 1.0);
        nh.param<int>("v_losm/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        nh.param<float>("v_losm/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<float>("v_losm/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        nh.param<int>("v_losm/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        nh.param<float>("v_losm/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);

        nh.param<float>("v_losm/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("v_losm/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("v_losm/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);

        usleep(100);
        
    }

    /**
     * imu原始测量数据转换到lidar系，加速度、角速度、RPY
     */
    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu &imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // 对加速度向量做坐标系变换，注意这里要理解成坐标系变换，也就是同一个加速度在IMU坐标系和Lidar坐标系的不同表达。不能想象成对加速度做旋转
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // 对角速度做坐标系变换。将IMU坐标系下的向量变换到雷达坐标系。
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        /**
         * q_from是IMU在全局坐标系下的位姿，q_from: transformation_from_map_to_imu
         * extQRPY如果与extRot对应的话应该是lidar到imu的变换：transformation_from_lidar_to_imu
         * q_final是将雷达点云从雷达坐标系转换到map坐标系的变换，也是：transformation_from_map_to_lidar -> pcd_in_map = q_final * pcd_in_lidar
         */
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        // 为什么是右乘，可以动手画一下看看
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        // cout << "org linear_acceleration: " << "\nx: " << imu_in.linear_acceleration.x << "\ny: " << imu_in.linear_acceleration.y << "\nz: " << imu_in.linear_acceleration.z<< "\n";
        // double roll, pitch, yaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(imu_in.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        // cout << "rpy: " << "\nroll: " << roll << "\npitch: " << pitch << "\nyaw: " << yaw << "\n";

        if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            ros::shutdown();
        }

        return imu_out;
    }

    /**
     * 点云转换函数
     * 对点云cloudIn进行变换transformIn，返回结果点云
     * PCL有自己的点云转换函数，这里用了CPU指令进行多线程执行。具体效率还需要实验分析
     */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
            cloudOut->points[i].y = transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
            cloudOut->points[i].z = transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }
};

/**
 * 提取imu角速度
 */
template <typename T>
void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

/**
 * 提取imu姿态角RPY
 */
template <typename T>
void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

/**
 * 点到坐标系原点距离
 */
float pointDistance(PointType p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

/**
 * 两点之间距离
 * @param p1 点1
 * @param p2 点2
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

/**
 * 将PCL点云转为ROS2 PointCloud2 数据格式
 * 
 * @param thisCloud 要转换的PCL点云
 * @param thisStamp 消息头中要设置的时间戳
 * @param thisFrame 消息头中要设置的frame_id
 * @return tempCloud 返回ROS2 PointCloud2 数据格式
 */
sensor_msgs::PointCloud2 pclPointcloud2Ros(pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    return tempCloud;
}

/**
 * 获取msg时间戳
 */
template <typename T>
double ROS_TIME(T msg)
{
    return msg->header.stamp.toSec();
}

/**
 * float转string，保留2位小数
 *
 * @param dbNum 要转成字符串的float
 * @return strCode 返回的字符串
 */
string floatToString(const float &dbNum)
{
    char *chCode;
    chCode = new (std::nothrow) char[20];
    sprintf(chCode, "%.2lf", dbNum); // .2 是控制输出精度的，两位小数
    string strCode(chCode);
    delete[] chCode;
    return strCode;
}

/**
 * 随机打乱数组
 * @param len 数组长度
 * @return arr 打乱后的数组
 */
int *ShuffleArray_Fisher_Yates(int len)
{
    int *arr = new int[len];
    for (int m = 0; m < len; m++)
    {
        arr[m] = m;
    }
    int i = len;
    int j;
    int temp;
    while (--i)
    {
        j = rand() % (i + 1);
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    return arr;
}

/**
 * 发布thisCloud，返回thisCloud对应msg格式
 */
sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

#endif