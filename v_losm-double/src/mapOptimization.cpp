//**** 设计步骤 ****//
// 对点云进行 scan to submap
//*****************// 
#include "v_losm/cloud_info.h"
#include "v_losm/utility.h"
#include "v_losm/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

struct FMCWPointType
{
    PCL_ADD_POINT4D
    float velocity;
    uint ring;
    uint column;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (FMCWPointType,
    (float, x, x) (float, y, y) (float, z, z) (float, velocity, velocity)
    (uint,ring,ring) (uint,column,column)
)


// 初始20帧用于初始化地图
int initFrameCount = 20;
// 两帧之间的间隔时间（秒）
// float deltaTime = 0.11;
float deltaTime = 0.1;

// 地图的范围
float mapMinX = -100.0;
float mapMaxX = 100.0;
float mapMinY = -100.0;
float mapMaxY = 100.0;

// 用于存储路径文件
// std::ofstream csv_file;

class MapOptimization : public ParamServer
{
public:
    // 因子图
    NonlinearFactorGraph gtSAMgraph;
    // 因子图变量初始值
    Values initialEstimate;
    // 非线性优化器
    ISAM2 *isam;
    // 优化器当前优化结果
    Values isamCurrentEstimate;
    // 当前优化结果的位姿方差。该方差在GPS因子中用到，如果该方差较小，则说明优化结果较好，即使打开GPS开关也不会将GPS因子加入因子图。
    Eigen::MatrixXd poseCovariance;

    ros::NodeHandle nh;
    // 订阅从featureExtraction模块发布出来的点云信息集合
    ros::Subscriber subFMCWCloudInfo;
    // 发布点云信息
    ros::Publisher pubFMCWCloudInfo;
    // 发布雷达里程计
    ros::Publisher pubFMCWLidarOdom;
    // 发布轨迹
    ros::Publisher pubFMCWLidarPath;
    // 发布全局地图
    ros::Publisher pubStationaryCloud;
    // 发布回环
    ros::Publisher pubLoopConstraintEdge;
    // 发布全局地图
    ros::Publisher pubLaserCloudSurround;
    // 地图保存服务
    ros::ServiceServer srvSaveMap;

    // 当前帧点云信息
    v_losm::cloud_info cloudInfo;

    // 标识点云匹配的结果是否较差，当isDegenerate为true的时候，标识本次的点云匹配结果较差，
    // 会在雷达里程计的协方差中置位，在imuPreintegration中会根据这个标志位选择因子图的噪声模型
    bool isDegenerate = false;
    cv::Mat matP;

    /**
     * 注意注意注意！！这是一个非常重要的变量，transformMap[6]缓存的是当前帧
     * 的`最新`位姿x,y,z,roll,pitch,yaw。无论在哪个环节，对位姿的更新都会被缓存到这个
     * 变量供给下一个环节使用！！
     */
    float transformMap[6];
    /**
     * 这也是一个非常重要的变量，存放的是之前帧到当前帧的变换的估计
     */
    float transformBetween[6];

    /**
     * cloudKeyPoses3D保存所有关键帧的三维位姿，x,y,z
     * cloudKeyPoses6D保存所有关键帧的六维位姿，x,y,z,roll,pitch,yaw
     * 带copy_前缀的两个位姿序列是在回环检测线程中使用的，只是为了不干扰主线程计算，实际内容一样。
     */
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 点云信息回调函数锁
    std::mutex mtx;

    // 当新的回环节点出现或者GPS信息被加入校正位置，这个变量被置为true，
    // 因子图优化器会执行多次更新，然后将所有的历史帧位置都更新一遍
    bool aLoopIsClosed = false;

    // 回环的索引字典，从当前帧到回环节点的索引
    map<int, int> loopIndexContainer;
    // 所有回环配对关系
    vector<pair<int, int>> loopIndexQueue;
    // 所有回环的姿态配对关系
    vector<gtsam::Pose3> loopPoseQueue;
    // 每个回环因子的噪声模型
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    // 在构建局部地图时挑选的邻近时间关键帧的三维姿态（构建kdtree加速搜索）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 历史所有关键帧的平面点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    // 全局关键帧轨迹
    nav_msgs::Path globalPath;

    // cloud 
    // 当前帧点云的角点
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    // 当前帧点云的角点下采样后的点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    // 当前帧点云的平面点
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    // 当前帧点云的平面点下采样后的点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;

    // map
    // 角点地图
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 角点地图裁切后
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapCrop;
    // 角点地图下采样(存放最终的地图的)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 平面点地图
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 平面点地图裁切后
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapCrop;
    // 平面点地图下采样(存放最终的地图的)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // 角点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    // 平面点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    // 做回环检测时使用ICP时的点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterICP;

    // 在构建局部地图时挑选的邻近时间关键帧的三维姿态（构建kdtree加速搜索）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    // 在做点云匹配时构建的角点kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    // 在做点云匹配时构建的平面点kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    // 静态点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_static;
    // 动态点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_moving;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_static_filter;
    pcl::PointCloud<PointType>::Ptr cloud_static_out;

    // 在做点云匹配的过程中使用的中间变量
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    bool que_has_update = false;

    // 当前雷达帧的时间戳，秒
    double timeLaserInfoCur;
    // 当前雷达帧的时间戳
    ros::Time timeLaserInfoStamp;

    // transform pre
    float V_now[3] = {0,0,0}; // 当前帧的自身速度分量Vx,Vy,Vz
    float V_last[3] = {0,0,0}; // 上一帧的自身速度分量Vx,Vy,Vz
    float W_now = 0; // 当前帧偏转角速度
    float W_last = 0; // 上一帧偏转角速度
    float Vel_now = 0; // 当前帧速度
    float Vel_last = 0; // 上一帧速度


    // 当前帧scan前记录的上一帧的最终的全局pose
    PointTypePose lastScanPose;
    // 上一帧和当前帧最终的差值
    PointTypePose lastBePose;



    // 构造函数
    MapOptimization()
    {
        // ISAM2 优化器参数
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // 订阅cloudInfo
        subFMCWCloudInfo = nh.subscribe<v_losm::cloud_info>("/cloudInfo_feature",30,&MapOptimization::FMCWLidarCloudHandler,this,ros::TransportHints().tcpNoDelay());
        // 点云信息发布
        pubFMCWCloudInfo = nh.advertise<v_losm::cloud_info>("/cloudInfo_odometry",30);
        // 里程计发布
        pubFMCWLidarOdom = nh.advertise<nav_msgs::Odometry>("/v_losm/Optimization/odometry",10);
        // 雷达轨迹信息发布
        pubFMCWLidarPath = nh.advertise<nav_msgs::Path>("/v_losm/Odometry/lidar_path",10);
        // 局部地图信息发布
        pubStationaryCloud = nh.advertise<sensor_msgs::PointCloud2>("/v_losm/Optimization/cloud_static",10);
        // 发布闭环边，rviz中表现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/v_losm/Optimization/loop_closure_constraints", 1);
        // 发布全局地图
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/map/map_global", 1);
        // 发布地图保存服务
        srvSaveMap = nh.advertiseService("/v_losm/save_map", &MapOptimization::saveMapService, this);

        // 分别给边缘点和平面点设置下采样
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // icp下采样
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        // map初始化
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        // 关键帧位姿初始化
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        // 初始化全局位姿变换为0
        for (int i = 0; i < 6; ++i)
        {
            transformMap[i] = 0;// x y z roll pitch yaw
        }

        // // 打开 CSV 文件用于写入
        // csv_file.open("/home/zwp/disk_1t/slam/bag/jch/1129/csv/lidar_path.csv", std::ios::out);
        // if (!csv_file.is_open())
        // {
        //     ROS_ERROR("无法打开文件进行写入！");
        // }
        // // 初始化时写入 CSV 文件的表头
        // csv_file << "x,y,z" << std::endl;
    }

    // 初始化
    void initialization()
    {
        // cloud
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); 
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); 
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); 
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());

        // map
        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapCrop.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapCrop.reset(new pcl::PointCloud<PointType>());

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(row * column);
        coeffSelCornerVec.resize(row * column);
        laserCloudOriCornerFlag.resize(row * column);
        laserCloudOriSurfVec.resize(row * column);
        coeffSelSurfVec.resize(row * column);
        laserCloudOriSurfFlag.resize(row * column);

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        cloud_static.reset(new pcl::PointCloud<pcl::PointXYZ>());
        cloud_moving.reset(new pcl::PointCloud<pcl::PointXYZ>());
        cloud_static_filter.reset(new pcl::PointCloud<pcl::PointXYZ>());
        cloud_static_out.reset(new pcl::PointCloud<PointType>());

        // 创建6行6列的矩阵，类型为32位浮点
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        // 这是两帧之间的变换，初始化为0
        for (int i = 0; i < 6; ++i){
            transformBetween[i] = 0; // x y z roll pitch yaw
        }
    }

    void FMCWLidarCloudHandler(const v_losm::cloud_infoConstPtr& lidarCloudMsg)
    {
        // 获取当前帧时间
        timeLaserInfoCur = lidarCloudMsg->header.stamp.toSec();
        timeLaserInfoStamp = lidarCloudMsg->header.stamp;

        // cout<< "当前时间：" << fixed << timeLaserInfoCur <<endl;

        // 初始化
        initialization();

        // 数据输入
        cloudInfo = *lidarCloudMsg; 
        // 输入边缘点和平面点
        pcl::PointCloud<FMCWPointType>::Ptr corner_in(new pcl::PointCloud<FMCWPointType>());
        pcl::PointCloud<FMCWPointType>::Ptr surface_in(new pcl::PointCloud<FMCWPointType>());

        // 输入边缘点和平面点
        pcl::fromROSMsg(cloudInfo.cloud_corner, *corner_in);
        pcl::fromROSMsg(cloudInfo.cloud_surface, *surface_in);

        // 边缘点和平面点都拷贝了一下
        pcl::copyPointCloud(*corner_in, *laserCloudCornerLast);
        pcl::copyPointCloud(*surface_in, *laserCloudSurfLast);

        /**
         * 一共三个线程使用到这把锁
         * 1. 雷达里程计线程，也就是当前线程
         * 2. 发布全局地图线程，执行关键帧点云拷贝转换操作
         * 3. 回环检测线程，执行关键帧姿态拷贝操作
         */
        std::lock_guard<std::mutex> lock(mtx);

        // 对平面点与边缘点进行降采样
        downsampleCurrentScan(); 

        // 前20帧当一帧好了
        if (initFrameCount > 0)
        {
            // 直接将降采样后的边缘点和平面点放入MapDS
            *laserCloudCornerFromMapDS += *laserCloudCornerLastDS;
            *laserCloudSurfFromMapDS += *laserCloudSurfLastDS;

            // 发布一下map
            pubSubMap();

            initFrameCount--;

            return;
        }

        // 计算transformBetween并设置初始参数，这里的between是上一帧到当前帧的预测变换
        setInitialTransform();

        // 初始位姿的估计
        updateInitialGuess();

        // double time2 = ros::Time::now().toSec();
        // 更新局部地图
        updateSubmap();

        // 发布局部地图
        pubSubMap();

        // double time3 = ros::Time::now().toSec();
        
        // 匹配优化位姿
        scan2SubmapOptimization(); // scan to submap

        // double time4 = ros::Time::now().toSec();

        // cout << "\033[17;0H" << "\033[K" << "Dowmsample submap time: " << time4 - time3 << endl;

        // 判断当前帧是否为关键帧，是就添加，添加过程中有更新transformMap
        addCurFrame();

        // 更新历史关键帧位姿
        correctPoses();

        pubOdometry(); // 发布里程信息
    }

    /**
     * 当前帧位姿初始估计
     * 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     * 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
     */
    void updateInitialGuess()
    {
        // 记录scan之前的变换
        lastScanPose.x = transformMap[0];
        lastScanPose.y = transformMap[1];
        lastScanPose.z = transformMap[2];
        lastScanPose.roll = transformMap[3];
        lastScanPose.pitch = transformMap[4];
        lastScanPose.yaw = transformMap[5];

        // 用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        // imu里程计可用
        if (cloudInfo.odomAvailable == true)
        // if (false)
        {
            // cout<< "using imu Odometry!!!!!!!!!!!!!!!!!"<<endl;

            // // 当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换, 用增量速度快的好像也行
            // Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX, cloudInfo.initialGuessY, cloudInfo.initialGuessZ,
            //                                                    cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // // 若是第一帧，使用了imu的rpy。xyz则还是000.
            // if (lastImuPreTransAvailable == false)
            // {
            //     transformMap[3] = cloudInfo.imuRollInit;
            //     transformMap[4] = cloudInfo.imuPitchInit;
            //     transformMap[5] = cloudInfo.imuYawInit;
            //     // 赋值给前一帧
            //     lastImuPreTransformation = transBack;
            //     lastImuPreTransAvailable = true;
            // }
            // else
            // {
            //     // 当前帧的初始位姿估计相对于前一帧的位姿之间的变换，imu里程计计算得到
            //     Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
            //     // 前一帧最终优化出来的位姿
            //     Eigen::Affine3f transTobe = trans2Affine3f(transformMap);
            //     // 当前帧的位姿估计
            //     Eigen::Affine3f transFinal = transTobe * transIncre;
            //     // 更新位姿估计到transformMap中
            //     pcl::getTranslationAndEulerAngles(transFinal, transformMap[0], transformMap[1], transformMap[2],
            //                                       transformMap[3], transformMap[4], transformMap[5]);
            //     // 当前帧imu估计值赋值给前一帧
            //     lastImuPreTransformation = transBack;
            // }

            Eigen::Affine3f rotation_t_be = pcl::getTransformation(transformBetween[0], transformBetween[1], transformBetween[2], transformBetween[3], transformBetween[4], transformBetween[5]);
            // 上一帧的最终变换
            Eigen::Affine3f rotation_t_last = pcl::getTransformation(transformMap[0],
                                                                     transformMap[1],
                                                                     transformMap[2],
                                                                     transformMap[3],
                                                                     transformMap[4],
                                                                     transformMap[5]);
            // 上一帧的变换加上到当前帧的变换得到最终变换,这里主要的是给一个初始的位姿估计，后面再进行迭代优化
            Eigen::Affine3f rotation_t = rotation_t_last * rotation_t_be;

            // 从rotation_t获取平移和欧拉角，设置进transformMap
            pcl::getTranslationAndEulerAngles(rotation_t, transformMap[0], transformMap[1], transformMap[2],
                                              transformMap[3], transformMap[4], transformMap[5]);

            // transformMap[0] = cloudInfo.initialGuessX;
            // transformMap[1] = cloudInfo.initialGuessY;
            // transformMap[2] = cloudInfo.initialGuessZ;
            transformMap[3] = cloudInfo.initialGuessRoll;
            transformMap[4] = cloudInfo.initialGuessPitch;
            transformMap[5] = cloudInfo.initialGuessYaw;
        }else{ // imu里程计不可用的情况下用速度来进行位姿估计
            // cout << "using v **********************" << endl;
            // 用速度估计出来的两帧之间的变换
            Eigen::Affine3f rotation_t_be = pcl::getTransformation(transformBetween[0], transformBetween[1], transformBetween[2], transformBetween[3], transformBetween[4], transformBetween[5]);
            // 上一帧的最终变换
            Eigen::Affine3f rotation_t_last = pcl::getTransformation(transformMap[0],
                                                                     transformMap[1],
                                                                     transformMap[2],
                                                                     transformMap[3],
                                                                     transformMap[4],
                                                                     transformMap[5]);
            // 上一帧的变换加上到当前帧的变换得到最终变换,这里主要的是给一个初始的位姿估计，后面再进行迭代优化
            Eigen::Affine3f rotation_t = rotation_t_last * rotation_t_be;
            // Eigen::Affine3f rotation_t = rotation_t_last;

            // 从rotation_t获取平移和欧拉角，设置进transformMap
            pcl::getTranslationAndEulerAngles(rotation_t, transformMap[0], transformMap[1], transformMap[2],
                                                transformMap[3], transformMap[4], transformMap[5]);           
        }
    }

    /**
     * Eigen格式的位姿变换
     */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[0], transformIn[1], transformIn[2], transformIn[3], transformIn[4], transformIn[5]);
    }

    /**
     * 位姿格式变换
     */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    /**
     * 位姿格式变换
     */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                            gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
    }

    /**
     * 位姿格式变换
     */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[3], transformIn[4], transformIn[5]),
                            gtsam::Point3(transformIn[0], transformIn[1], transformIn[2]));
    }

    /**
     * 对平面点和边缘点降采样
     */
    void downsampleCurrentScan()
    {
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
    }

    /**
     * 计算transformBetween
     */
    void setInitialTransform()
    {
        V_last[0] = V_now[0]; // Vx
        V_last[1] = V_now[1]; // Vy
        V_last[2] = V_now[2]; // Vz
        W_last = W_now;
        Vel_last = Vel_now;

        // 计算上一帧到当前帧的变换
        if(Vel_now<5.0){ // 低速情况下的阿克曼转向原理
            transformBetween[0] = V_last[0] * deltaTime; // Vx*∆t
            transformBetween[1] = V_last[1] * deltaTime; // Vy*∆t
            transformBetween[2] = V_last[2] * deltaTime; // Vz*∆t
            transformBetween[3] = lastBePose.roll; // roll
            transformBetween[4] = lastBePose.pitch; // pitch
            transformBetween[5] = W_last * deltaTime;  // yaw
        }else{
            transformBetween[0] = V_last[0] * deltaTime; // x
            transformBetween[1] = V_last[1] * deltaTime; // y
            transformBetween[2] = V_last[2] * deltaTime; // z
            static uint nn=0;
            nn++;
            if(nn>3){ // 第4帧开始用lastBePose
                transformBetween[3] = lastBePose.roll; // roll
                transformBetween[4] = lastBePose.pitch; // pitch
                transformBetween[5] = lastBePose.yaw; // yaw
            }else{
                transformBetween[3] = 0.0; // roll
                transformBetween[4] = 0.0; // pitch
                transformBetween[5] = 0.0; // yaw
            }
        }

        // 将当前帧的数据设置进去
        // 设置当前帧速度分量
        V_now[0] = cloudInfo.self_v[0];
        V_now[1] = cloudInfo.self_v[1];
        V_now[2] = cloudInfo.self_v[2];
        // 根据分量获得当前速度
        Vel_now = sqrt(V_now[0]*V_now[0] + V_now[1]*V_now[1] + V_now[2]*V_now[2]);
        if(Vel_now < 0.01){
            W_now = 0;
        }else{
            // 计算当前偏转角速度，1.1为激光雷达到后轴的距离
            W_now =  Vel_now * sin(atan2(V_now[1],V_now[0])) / (1.1);
        }
    }

    /**
     * 更新局部地图
     */
    void updateSubmap()
    {
        // 将最新帧加入局部地图
        if(que_has_update==true){
            // 更新原始地图
            laserCloudCornerFromMap->clear();
            laserCloudSurfFromMap->clear();

            // 这里转了一手，后面的结果还是放在MapDS里的
            pcl::copyPointCloud(*laserCloudCornerFromMapDS,  *laserCloudCornerFromMap);
            pcl::copyPointCloud(*laserCloudSurfFromMapDS,  *laserCloudSurfFromMap);

            // 这里其实把前面的一帧关键帧经过变换放入了map
            addSubmap();

            // 对map降采样
            laserCloudCornerFromMapCrop->clear();
            downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
            downSizeFilterCorner.filter(*laserCloudCornerFromMapCrop);

            laserCloudSurfFromMapCrop->clear();
            downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
            downSizeFilterSurf.filter(*laserCloudSurfFromMapCrop);

            // 切割，保留部分局部地图
            laserCloudCornerFromMapDS->clear();
            laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
            laserCloudCornerFromMapDS->reserve(laserCloudCornerFromMapCrop->points.size());

            laserCloudSurfFromMapDS->clear();
            laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());
            laserCloudSurfFromMapDS->reserve(laserCloudSurfFromMapCrop->points.size());

            cropSubMap();

            // 将裁剪后的map放回MapDS
            vector<int> indices;
            laserCloudCornerFromMapCrop->is_dense = false;
            laserCloudSurfFromMapCrop->is_dense = false;
            pcl::removeNaNFromPointCloud<PointType>(*laserCloudCornerFromMapCrop, *laserCloudCornerFromMapDS, indices);
            pcl::removeNaNFromPointCloud<PointType>(*laserCloudSurfFromMapCrop, *laserCloudSurfFromMapDS, indices);

            // 重置标志位
            que_has_update = false;
        }
    }

    /**
     * 添加地图
     * 将队列里的corner和surface做变换后分别加入laserCloudCornerFromMap和laserCloudSurfFromMap
     */
    void addSubmap()
    {
        // 将地图转换到雷达坐标系下
        Eigen::Affine3f rotation_t = pcl::getTransformation(cloudKeyPoses6D->back().x,
                                                            cloudKeyPoses6D->back().y,
                                                            cloudKeyPoses6D->back().z,
                                                            cloudKeyPoses6D->back().roll,
                                                            cloudKeyPoses6D->back().pitch,
                                                            cloudKeyPoses6D->back().yaw);

        pcl::PointCloud<PointType>::Ptr thisCorner(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurface(new pcl::PointCloud<PointType>());
        thisCorner->resize(cornerCloudKeyFrames.back()->size());
        thisSurface->resize(surfCloudKeyFrames.back()->size());

        // corner
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(cornerCloudKeyFrames.back()->points.size()); i++)
        {
            thisCorner->points[i].x = rotation_t(0, 0) * cornerCloudKeyFrames.back()->points[i].x + rotation_t(0, 1) * cornerCloudKeyFrames.back()->points[i].y + rotation_t(0, 2) * cornerCloudKeyFrames.back()->points[i].z + rotation_t(0, 3);
            thisCorner->points[i].y = rotation_t(1, 0) * cornerCloudKeyFrames.back()->points[i].x + rotation_t(1, 1) * cornerCloudKeyFrames.back()->points[i].y + rotation_t(1, 2) * cornerCloudKeyFrames.back()->points[i].z + rotation_t(1, 3);
            thisCorner->points[i].z = rotation_t(2, 0) * cornerCloudKeyFrames.back()->points[i].x + rotation_t(2, 1) * cornerCloudKeyFrames.back()->points[i].y + rotation_t(2, 2) * cornerCloudKeyFrames.back()->points[i].z + rotation_t(2, 3);
        }

        // surface
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(surfCloudKeyFrames.back()->points.size()); i++)
        {
            thisSurface->points[i].x = rotation_t(0, 0) * surfCloudKeyFrames.back()->points[i].x + rotation_t(0, 1) * surfCloudKeyFrames.back()->points[i].y + rotation_t(0, 2) * surfCloudKeyFrames.back()->points[i].z + rotation_t(0, 3);
            thisSurface->points[i].y = rotation_t(1, 0) * surfCloudKeyFrames.back()->points[i].x + rotation_t(1, 1) * surfCloudKeyFrames.back()->points[i].y + rotation_t(1, 2) * surfCloudKeyFrames.back()->points[i].z + rotation_t(1, 3);
            thisSurface->points[i].z = rotation_t(2, 0) * surfCloudKeyFrames.back()->points[i].x + rotation_t(2, 1) * surfCloudKeyFrames.back()->points[i].y + rotation_t(2, 2) * surfCloudKeyFrames.back()->points[i].z + rotation_t(2, 3);
        }

        // 将做变换后的corner和surface分别放入map
        *laserCloudCornerFromMap += *thisCorner;
        *laserCloudSurfFromMap += *thisSurface;
    }

    /**
     * 切割部分局部地图
     * 地图越加越大，去掉超出范围的那部分
     */
    void cropSubMap()
    {
        // 这应该是将地图转换至雷达坐标系
        Eigen::Affine3f rotation_t = pcl::getTransformation(cloudKeyPoses6D->back().x,
                                                            cloudKeyPoses6D->back().y,
                                                            cloudKeyPoses6D->back().z,
                                                            cloudKeyPoses6D->back().roll,
                                                            cloudKeyPoses6D->back().pitch,
                                                            cloudKeyPoses6D->back().yaw);

        rotation_t = rotation_t.inverse();

        // corner
        #pragma omp parallel for num_threads(numberOfCores)
        for(int i=0;i<int(laserCloudCornerFromMapCrop->points.size());i++){
            float x = rotation_t(0,0)*laserCloudCornerFromMapCrop->points[i].x+rotation_t(0,1)*laserCloudCornerFromMapCrop->points[i].y+rotation_t(0,2)*laserCloudCornerFromMapCrop->points[i].z+rotation_t(0,3);
            float y = rotation_t(1,0)*laserCloudCornerFromMapCrop->points[i].x+rotation_t(1,1)*laserCloudCornerFromMapCrop->points[i].y+rotation_t(1,2)*laserCloudCornerFromMapCrop->points[i].z+rotation_t(1,3);
            if(x < mapMinX || x > mapMaxX || y < mapMinY || y > mapMaxY){
                laserCloudCornerFromMapCrop->points[i].x = NAN;
                laserCloudCornerFromMapCrop->points[i].y = NAN;
                laserCloudCornerFromMapCrop->points[i].z = NAN;
            }
        }

        // surface
        #pragma omp parallel for num_threads(numberOfCores)
        for(int i=0;i<int(laserCloudSurfFromMapCrop->points.size());i++){
            float x = rotation_t(0,0)*laserCloudSurfFromMapCrop->points[i].x+rotation_t(0,1)*laserCloudSurfFromMapCrop->points[i].y+rotation_t(0,2)*laserCloudSurfFromMapCrop->points[i].z+rotation_t(0,3);
            float y = rotation_t(1,0)*laserCloudSurfFromMapCrop->points[i].x+rotation_t(1,1)*laserCloudSurfFromMapCrop->points[i].y+rotation_t(1,2)*laserCloudSurfFromMapCrop->points[i].z+rotation_t(1,3);
            if (x < mapMinX || x > mapMaxX || y < mapMinY || y > mapMaxY){
                laserCloudSurfFromMapCrop->points[i].x = NAN;
                laserCloudSurfFromMapCrop->points[i].y = NAN;
                laserCloudSurfFromMapCrop->points[i].z = NAN;
            }
        }
    }

    void scan2SubmapOptimization()
    {
        // 要求有关键帧
        if (cloudKeyPoses3D->points.empty())
            return;

        // 当前激光帧的角点、平面点数量足够多
        if (int(laserCloudCornerLastDS->points.size())>edgeFeatureMinValidNum && int(laserCloudSurfLastDS->points.size())>surfFeatureMinValidNum){


            // 地图放进kdtree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            static float meanCount = 0;
            static float numCount = 0;

            // 20次迭代优化
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (GNOptimization(iterCount) == true){
                    numCount++;
                    meanCount += iterCount;
                    // cout << "true : " << iterCount << endl;
                    break;     
                }
                if (iterCount >= 29){
                    cout << "false!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
                }
            }

            transformUpdate();

        }else{
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDS->points.size(), laserCloudSurfLastDS->points.size());
        }
    }

    /**
     * 当前激光帧角点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     */
    void cornerOptimization()
    {
        Eigen::Affine3f rotation_t = pcl::getTransformation(transformMap[0], transformMap[1], transformMap[2], transformMap[3], transformMap[4], transformMap[5]);

        // 遍历当前帧角点集合
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(laserCloudCornerLastDS->points.size()); i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 角点（坐标还是lidar系）
            pointOri = laserCloudCornerLastDS->points[i];
            
            // 转到地图坐标系
            pointSel.x = rotation_t(0,0)*pointOri.x+rotation_t(0,1)*pointOri.y+rotation_t(0,2)*pointOri.z+rotation_t(0,3);
            pointSel.y = rotation_t(1,0)*pointOri.x+rotation_t(1,1)*pointOri.y+rotation_t(1,2)*pointOri.z+rotation_t(1,3);
            pointSel.z = rotation_t(2,0)*pointOri.x+rotation_t(2,1)*pointOri.y+rotation_t(2,2)*pointOri.z+rotation_t(2,3);
            pointSel.intensity = pointOri.intensity;

            // 找到最近5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

            // 5个点中最远距离1米内        
            if (pointSearchSqDis[4] < 1.0) {
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                // 5个点的平均值，记为中心点
                cx /= 5; cy /= 5;  cz /= 5;

                // 计算协方差
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    // 计算点与中心点之间的距离
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
                // 特征值分解
                cv::eigen(matA1, matD1, matV1);

                // 如果最大的特征值相比次大特征值，大很多，认为构成了线，角点是合格的
                if (matD1.at<float>(0, 0) > 5 * matD1.at<float>(0, 1)) {

                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float ld2 = a012 / l12;

                    // 如果点pointSel刚好在直线上，则ld2=0,s=1；
                    // 点到直线的距离越远，s越小，则赋予的比重越低
                    float s = 1 - 0.9 * fabs(ld2);

                    // 使用系数对法向量加权，实际上相当于对导数（雅克比矩阵加权了）
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    // 经验阈值，判断点到直线的距离是否够近，足够近才采纳为优化目标点
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 当前激光帧平面点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     * 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     */
    void surfOptimization()
    {
        Eigen::Affine3f rotation_t=pcl::getTransformation(transformMap[0], transformMap[1], transformMap[2], transformMap[3], transformMap[4], transformMap[5]);
        
        // 遍历当前帧平面点集合
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(laserCloudSurfLastDS->points.size()); i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 平面点（坐标还是lidar系）
            pointOri = laserCloudSurfLastDS->points[i];

            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointSel.x = rotation_t(0,0)*pointOri.x+rotation_t(0,1)*pointOri.y+rotation_t(0,2)*pointOri.z+rotation_t(0,3);
            pointSel.y = rotation_t(1,0)*pointOri.x+rotation_t(1,1)*pointOri.y+rotation_t(1,2)*pointOri.z+rotation_t(1,3);
            pointSel.z = rotation_t(2,0)*pointOri.x+rotation_t(2,1)*pointOri.y+rotation_t(2,2)*pointOri.z+rotation_t(2,3);
            pointSel.intensity = pointOri.intensity;

            // 在局部平面点map中查找当前平面点相邻的5个平面点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 要求距离都小于1m
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 假设平面方程为ax+by+cz+1=0，这里就是求方程的系数abc，d=1
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 平面方程的系数，也是法向量的分量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // 单位法向量
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 检查平面是否合格，如果5个点中有点到平面的距离超过0.2m，那么认为这些点太分散了，不构成平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                            pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                            pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 平面合格
                if (planeValid) {
                    // 当前激光帧点到平面距离
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 与cornerOptimization中类似，使用距离计算一个权重
                    // keep in mind, 后面部分（0.9*fabs.....）越小越好。因此，这里可以理解为对点到平面距离的加权
                    // 越远的平面对匹配具有更好的约束性，因此要赋予更大的比重。
                    // 值得注意的是，pd2的计算是在地图坐标系下计算，而这里权重的分母的计算是使用pointOri，也就是
                    // 雷达坐标系下的点进行计算
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    // 点到平面垂线单位法向量（其实等价于平面法向量）
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 如果点到平面的距离够小，则采纳为优化目标点，否则跳过
                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     */
    void combineOptimizationCoeffs()
    {
        int cornerNum=0;
        int surfaceNum=0;
        laserCloudOri->reserve(30000);
        coeffSel->reserve(30000);
        // combine corner coeffs
        for (int i = 0; i < int(laserCloudCornerLastDS->points.size()); ++i){
            if (laserCloudOriCornerFlag[i] == true){
                cornerNum++;
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < int(laserCloudSurfLastDS->points.size()); ++i){
            if (laserCloudOriSurfFlag[i] == true){
                surfaceNum++;
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // 清空标记
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool GNOptimization(int iterCount)
    {
        // 计算三轴欧拉角的sin、cos，后面使用旋转矩阵对欧拉角求导中会使用到
        float srx = sin(transformMap[3]);
        float crx = cos(transformMap[3]);
        float sry = sin(transformMap[4]);
        float cry = cos(transformMap[4]);
        float srz = sin(transformMap[5]);
        float crz = cos(transformMap[5]);

        // 当前帧匹配特征点数太少则跳过此次优化
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum*2, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum*2, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum*2, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        // 两种约束权重,lamda2值越大，处理速度越快，但里程计精度可能会下降
        float lamda2 = 0;
        float lamda1 = 1-lamda2;
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSelNum; i++) {
            PointType pointOri, coeff;
            // 点向量
            pointOri.x = laserCloudOri->points[i].x;
            pointOri.y = laserCloudOri->points[i].y;
            pointOri.z = laserCloudOri->points[i].z;
            // 距离向量
            coeff.x = coeffSel->points[i].x;
            coeff.y = coeffSel->points[i].y;
            coeff.z = coeffSel->points[i].z;
            coeff.intensity = coeffSel->points[i].intensity;
            // 雅可比矩阵
            float arx = (0*pointOri.x + (crz*sry*crx+srx*srz)*pointOri.y + (srz*crx-crz*srx*sry)*pointOri.z) * coeff.x
                    + (0*pointOri.x + (-crz*srx+srz*sry*crx)*pointOri.y + (-srx*srz*sry-crz*crx)*pointOri.z) * coeff.y 
                    + (0*pointOri.x + (cry*crx)*pointOri.y + (-cry*srx)*pointOri.z) * coeff.z ;

            float ary = (-crz*sry*pointOri.x + (crz*cry*srx)*pointOri.y + (crz*crx*cry)*pointOri.z) * coeff.x
                    + (-sry*srz*pointOri.x + (srz*cry*srx)*pointOri.y + (crx*srz*cry)*pointOri.z) * coeff.y
                    + (-cry*pointOri.x + (-sry*srx)*pointOri.y + (-sry*crx)*pointOri.z) * coeff.z;

            float arz = (-srz*cry*pointOri.x + (-srz*sry*srx-crx*crz)*pointOri.y + (crz*srx-srz*crx*sry)*pointOri.z) * coeff.x
                    + (cry*crz*pointOri.x + (-srz*crx+crz*sry*srx)*pointOri.y + (crx*crz*sry+srz*srx)*pointOri.z) * coeff.y
                    + (0*pointOri.x + 0*pointOri.y + 0*pointOri.z) * coeff.z;


            // 多普勒速度的雅可比矩阵
            float dist = sqrt(pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z);
            float dx = pointOri.x/dist;
            float dy = pointOri.y/dist;
            float dz = pointOri.z/dist;
            float arx_v = ((0*dx + (crz*sry*crx+srx*srz)*dy + (srz*crx-crz*srx*sry)*dz) * (transformMap[0]-lastScanPose.x)
                        + (0*dx + (-crz*srx+srz*sry*crx)*dy + (-srx*srz*sry-crz*crx)*dz) * (transformMap[1]-lastScanPose.y)
                        + (0*dx + (cry*crx)*dy + (-cry*srx)*dz) * (transformMap[2]-lastScanPose.z)) / 0.1;

            float ary_v = ((-crz*sry*dx + (crz*cry*srx)*dy + (crz*crx*cry)*dz) * (transformMap[0]-lastScanPose.x)
                        + (-sry*srz*dx + (srz*cry*srx)*dy + (crx*srz*cry)*dz) * (transformMap[1]-lastScanPose.y)
                        + (-cry*dx + (-sry*srx)*dy + (-sry*crx)*dz) * (transformMap[2]-lastScanPose.z)) / 0.1;
            
            float arz_v = ((-srz*cry*dx + (-srz*sry*srx-crx*crz)*dy + (crz*srx-srz*crx*sry)*dz) * (transformMap[0]-lastScanPose.x)
                        + (cry*crz*dx + (-srz*crx+crz*sry*srx)*dy + (crx*crz*sry+srz*srx)*dz) * (transformMap[1]-lastScanPose.y)
                        + (0*dx + 0*dy + 0*dz) * (transformMap[2]-lastScanPose.z)) / 0.1;

            float x_v = (crz*cry*dx+(crz*sry*srx-crx*srz)*dy+(srz*srx+crz*crx*sry)*dz) / 0.1;

            float y_v = (cry*srz*dx+(crz*crx+srz*sry*srx)*dy+(crx*srz*sry-crz*srx)*dz) / 0.1; 

            float z_v = ((-sry)*dx+(cry*srx)*dy+(cry*crx)*dz) / 0.1;

            float v_v = -(dx*cloudInfo.self_v[0] + dy*cloudInfo.self_v[1] + dz*cloudInfo.self_v[2])
                        + x_v*(transformMap[0]-lastScanPose.x)
                        + y_v*(transformMap[1]-lastScanPose.y)
                        + z_v*(transformMap[2]-lastScanPose.z);

            // x y z roll pitch yaw
            matA.at<float>(i*2, 0) = coeff.x * lamda1;
            matA.at<float>(i*2, 1) = coeff.y * lamda1;
            matA.at<float>(i*2, 2) = coeff.z * lamda1;
            matA.at<float>(i*2, 3) = arx * lamda1;
            matA.at<float>(i*2, 4) = ary * lamda1;
            matA.at<float>(i*2, 5) = arz * lamda1;
            matB.at<float>(i*2, 0) = -coeff.intensity * lamda1;

            matA.at<float>(i*2+1, 0) = x_v * lamda2;
            matA.at<float>(i*2+1, 1) = y_v * lamda2;
            matA.at<float>(i*2+1, 2) = z_v * lamda2;
            matA.at<float>(i*2+1, 3) = arx_v * lamda2;
            matA.at<float>(i*2+1, 4) = ary_v * lamda2;
            matA.at<float>(i*2+1, 5) = arz_v * lamda2;
            matB.at<float>(i*2+1, 0) = -v_v * lamda2;

            // //x y z roll pitch yaw
            // matA.at<float>(i, 0) = coeff.x;
            // matA.at<float>(i, 1) = coeff.y;
            // matA.at<float>(i, 2) = coeff.z;
            // matA.at<float>(i, 3) = arx;
            // matA.at<float>(i, 4) = ary;
            // matA.at<float>(i, 5) = arz;
            // matB.at<float>(i, 0) = -coeff.intensity ;
        }

        // float lamda1 = 0.81;

        // if(use_doppler == false){
        //     lamda2 = 0;
        //     lamda1 = 1;
        // }
        
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // cv::transpose(matA2, matA2t);
        // matA2tA2 = lamda2 * matA2t * matA2;
        // matA2tB2 = lamda2 * matA2t * matB2;
        // // cv::solve(matA2tA2, matA2tB2, matX2, cv::DECOMP_QR);

        // cv::solve(matAtA+matA2tA2, matAtB+matA2tB2, matX, cv::DECOMP_QR);

        // 如果是第一次迭代，判断求解出来的近似Hessian矩阵，也就是J^{T}J:=matAtA是否退化
        /**
         * 这部分的计算说实话没有找到很好的理论出处，这里只能大概说一下这段代码想要做的事情
         * 这里用matAtA也就是高斯-牛顿中的近似海瑟（Hessian）矩阵H。求解增量方程：J^{T}J\Delta{x} = -Jf(x)
         * 要求H:=J^{T}J可逆，但H不一定可逆。下面的代码通过H的特征值判断H是否退化，并将退化的方向清零matV2。而后又根据
         * matV.inv()*matV2作为更新向量的权重系数，matV是H的特征向量矩阵。
         */
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformMap[0] += matX.at<float>(0, 0);
        transformMap[1] += matX.at<float>(1, 0);
        transformMap[2] += matX.at<float>(2, 0);
        transformMap[3] += matX.at<float>(3, 0);
        transformMap[4] += matX.at<float>(4, 0);
        transformMap[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(3, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(4, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(5, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(0, 0) * 100, 2) +
                            pow(matX.at<float>(1, 0) * 100, 2) +
                            pow(matX.at<float>(2, 0) * 100, 2));
        
        // if(deltaR < 0.05 && deltaT < 0.05){
        //     use_doppler = false;
        // }

        // 如果迭代的步长达到设定阈值，则认为已经收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            // cout << iterCount <<":   deltaR: "<<deltaR<<"   deltaT: "<<deltaT<<endl;
            return true; // converged
        }
        return false; // keep optimizing
    }

    /**
     * rp的加权平均以及z轴的约束，更新到transformMap
    */
    void transformUpdate(){
        if (cloudInfo.odomAvailable == true)
        {
            // 俯仰角小于1.4
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformMap[3], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformMap[3] = rollMid;

                // pitch角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(0, transformMap[4], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformMap[4] = pitchMid;
            }
        }

        // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        // transformMap[3] = constraintTransformation(transformMap[3], rotation_tollerance);
        // transformMap[4] = constraintTransformation(transformMap[4], rotation_tollerance);

        // cout << "beforez!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<< transformMap[2]<< endl;
        // cout << "beforer!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << transformMap[3] << endl;
        // cout << "beforep!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << transformMap[4] << endl;
        // cout << "beforey!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << transformMap[5] << endl;
        transformMap[3] = constraintTransformation(transformMap[3], 0);
        transformMap[4] = constraintTransformation(transformMap[4], 0);
        transformMap[2] = constraintTransformation(transformMap[2], z_tollerance);
    }

    /**
     * 值约束
     */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    /**
     * 判断是否需要添加当前帧
     */
    bool saveFrame()
    {
        // 第一帧初始化info中的两帧之间变换，然后直接返回true
        if (cloudKeyPoses6D->points.empty())
        {
            cloudInfo.transform_between[0] = 0.0;
            cloudInfo.transform_between[1] = 0.0;
            cloudInfo.transform_between[2] = 0.0;
            cloudInfo.transform_between[3] = 0.0;
            cloudInfo.transform_between[4] = 0.0;
            cloudInfo.transform_between[5] = 0.0;
            return true;
        }

        // 上一次的全局变换
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 优化后最终的变换
        Eigen::Affine3f transFinal = pcl::getTransformation(transformMap[0], transformMap[1], transformMap[2], 
                                                            transformMap[3], transformMap[4], transformMap[5]);
        // 优化后的两者之间的变换
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 计算两帧之间的xyzrpy
        static int num=0;
        static float mean = 0;
        // 上一帧优化后的变换
        Eigen::Affine3f transStart_be = pclPointToAffine3f(lastScanPose);
        // 上一帧优化后的变换和本帧优化后最终的变换的差值
        Eigen::Affine3f transBetween_be = transStart_be.inverse() * transFinal;
        pcl::getTranslationAndEulerAngles(transBetween_be, lastBePose.x, lastBePose.y, lastBePose.z, lastBePose.roll, lastBePose.pitch, lastBePose.yaw);
        if(abs(lastBePose.yaw)>0.003){
            num++;
            mean += abs(lastBePose.yaw-transformBetween[5]);
        }

        // 对cloudInfo中的transform_between赋值
        cloudInfo.transform_between[0] = lastBePose.x;
        cloudInfo.transform_between[1] = lastBePose.y;
        cloudInfo.transform_between[2] = lastBePose.z;
        cloudInfo.transform_between[3] = lastBePose.roll;
        cloudInfo.transform_between[4] = lastBePose.pitch;
        cloudInfo.transform_between[5] = lastBePose.yaw;

        // cout << "\033[19;0H" <<"\033[K"<< "scan-subm yaw: " << lastBePose.yaw << endl;
        // cout << "\033[20;0H" <<"\033[K"<< "ackermann yaw: " << transformBetween[5] << endl;
        // cout << "\033[21;0H" <<"\033[K"<< "预测平均误差: " << mean/num << endl;

        // 不超过阈值就不往地图添加该帧
        if (abs(roll) < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw) < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x * x + y * y + z * z) < surroundingkeyframeAddingDistThreshold)
        {
            return true;
        }

        return true;
    }

    /**
     * 更新所有历史关键帧位姿
     * 1. 只当aLoopGpsIsClosed标志位为真时才执行历史关键帧位姿更新
     * 2. 从因子图优化器中拿出所有关键帧的位姿（优化结果）
     * 3. 清空全局路径变量，替换成当前的关键帧位姿序列
     * 4. 将优化后的位姿更新为当前的位姿
     */
    void correctPoses(){
        if(cloudKeyPoses3D->points.empty())
            return;
        
        if(aLoopIsClosed == true){
            // 清空全部之前的轨迹
            globalPath.poses.clear();
            // csv_file.open("/home/zwp/disk_1t/slam/bag/jch/1129/csv/lidar_path.csv", std::ios::trunc);
            // if (!csv_file.is_open())
            // {
            //     ROS_ERROR("无法打开文件进行写入！");
            // }
            // // 初始化时写入 CSV 文件的表头
            // csv_file << "x,y,z" << std::endl;

            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 重新添加轨迹
                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    /**
     * 更新里程计轨迹
     * @param pose_in 需要添加到全局路径的位姿
     */
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = mapFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        // csv_file << pose_in.x << "," << pose_in.y << "," << pose_in.z << std::endl;
        globalPath.poses.push_back(pose_stamped);
        
    }
    

    /**
     * 添加激光里程计因子
     */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧初始化先验因子
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformMap), priorNoise));
            // 变量节点设置初始值
            initialEstimate.insert(0, trans2gtsamPose(transformMap));
        }
        else
        {
            // 添加激光里程计因子
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo = trans2gtsamPose(transformMap);
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size() - 1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 变量节点设置初始值
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * 添加GPS因子
     */
    void addGPSFactor()
    {
        return;
    }

    /**
     * 添加闭环因子
     */
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        // 闭环队列
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 闭环边的位姿变换
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    /**
     * 添加当前帧到队列
     */
    void addCurFrame()
    {
        // 判断一下当前帧是否作为关键需要添加
        if (saveFrame() == true){

            // 激光里程计因子
            addOdomFactor();

            // GPS因子
            addGPSFactor();

            // 闭环因子
            addLoopFactor();

            isam->update(gtSAMgraph, initialEstimate);
            isam->update();

            if(aLoopIsClosed == true){
                isam->update();
                isam->update();
                isam->update();
                isam->update();
                isam->update();
            }
            // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
            gtSAMgraph.resize(0);
            initialEstimate.clear();

            Pose3 latestEstimate;
            // 优化结果
            isamCurrentEstimate = isam->calculateEstimate();
            // 当前帧位姿结果
            latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

            // cloudKeyPoses3D加入当前帧位姿
            PointType thisPose3D;
            thisPose3D.x = latestEstimate.translation().x();
            thisPose3D.y = latestEstimate.translation().y();
            thisPose3D.z = latestEstimate.translation().z();
            thisPose3D.intensity = cloudKeyPoses3D->size();
            cloudKeyPoses3D->push_back(thisPose3D);
            // PointType thisPose3D;
            // thisPose3D.x = transformMap[0];
            // thisPose3D.y = transformMap[1];
            // thisPose3D.z = transformMap[2];
            // thisPose3D.intensity = cloudKeyPoses3D->size();
            // cloudKeyPoses3D->push_back(thisPose3D);

            // cloudKeyPoses6D加入当前帧位姿
            PointTypePose thisPose6D;
            thisPose6D.x = thisPose3D.x;
            thisPose6D.y = thisPose3D.y;
            thisPose6D.z = thisPose3D.z;
            thisPose6D.intensity = thisPose3D.intensity;
            thisPose6D.roll = latestEstimate.rotation().roll();
            thisPose6D.pitch = latestEstimate.rotation().pitch();
            thisPose6D.yaw = latestEstimate.rotation().yaw();
            thisPose6D.time = timeLaserInfoCur;
            cloudKeyPoses6D->push_back(thisPose6D);
            // PointTypePose thisPose6D;
            // thisPose6D.x = thisPose3D.x;
            // thisPose6D.y = thisPose3D.y;
            // thisPose6D.z = thisPose3D.z;
            // thisPose6D.intensity = thisPose3D.intensity;
            // thisPose6D.roll = transformMap[3];
            // thisPose6D.pitch = transformMap[4];
            // thisPose6D.yaw = transformMap[5];
            // thisPose6D.time = timeLaserInfoCur;
            // cloudKeyPoses6D->push_back(thisPose6D);

            // transformMap更新当前帧位姿
            transformMap[0] = latestEstimate.translation().x();
            transformMap[1] = latestEstimate.translation().y();
            transformMap[2] = latestEstimate.translation().z();
            transformMap[3] = latestEstimate.rotation().roll();
            transformMap[4] = latestEstimate.rotation().pitch();
            transformMap[5] = latestEstimate.rotation().yaw();

            // 当前帧激光角点、平面点，降采样集合
            pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
            pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

            // 保存特征点降采样集合
            cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
            surfCloudKeyFrames.push_back(thisSurfKeyFrame);

            // 标识队列有更新，用于后续加入map
            que_has_update = true;

            // 更新全局路径
            updatePath(thisPose6D);

            // 设置全局变换
            cloudInfo.saveFrame = 1;
            cloudInfo.transform_map[0] = transformMap[0];
            cloudInfo.transform_map[1] = transformMap[1];
            cloudInfo.transform_map[2] = transformMap[2];
            cloudInfo.transform_map[3] = transformMap[3];
            cloudInfo.transform_map[4] = transformMap[4];
            cloudInfo.transform_map[5] = transformMap[5];
        }else{
            cloudInfo.saveFrame = 0;
            // 记录transformMap到cloudinfo中
            cloudInfo.transform_map[0] = transformMap[0];
            cloudInfo.transform_map[1] = transformMap[1];
            cloudInfo.transform_map[2] = transformMap[2];
            cloudInfo.transform_map[3] = transformMap[3];
            cloudInfo.transform_map[4] = transformMap[4];
            cloudInfo.transform_map[5] = transformMap[5];
        }
    }

    /**
     * 发布雷达全局路径、里程计、点云等信息
     */
    void pubOdometry()
    {
        // Publish TF
        // 正常输出的是map坐标系的点云，通过发布这个TF，实时的将map系的点云转到lidar系下
        static tf::TransformBroadcaster br;
        tf::Transform t_map_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformMap[3], transformMap[4], transformMap[5]),
                                                      tf::Vector3(transformMap[0], transformMap[1], transformMap[2]));
        tf::StampedTransform trans_map_to_lidar = tf::StampedTransform(t_map_to_lidar, cloudInfo.header.stamp, mapFrame, lidarFrame);
        br.sendTransform(trans_map_to_lidar);

        // 设置里程计
        nav_msgs::Odometry thisOdometry;
        thisOdometry.header = cloudInfo.header;
        thisOdometry.header.frame_id = mapFrame;
        thisOdometry.pose.pose.position.x = transformMap[0];
        thisOdometry.pose.pose.position.y = transformMap[1];
        thisOdometry.pose.pose.position.z = transformMap[2];
        thisOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformMap[3], transformMap[4], transformMap[5]);

        // 雷达路径信息发布
        if(pubFMCWLidarPath.getNumSubscribers()!=0){
            globalPath.header.stamp = timeLaserInfoStamp;
            // 这里用了mapFrame，会根据TF转到雷达坐标系
            globalPath.header.frame_id = mapFrame;
            pubFMCWLidarPath.publish(globalPath);
        }

        // 里程计发布
        pubFMCWLidarOdom.publish(thisOdometry);

        // 点云信息发布
        pubFMCWCloudInfo.publish(cloudInfo);
    }

    /**
     * 发布局部地图
    */
    void pubSubMap()
    {
        //发布局部地图信息
        pcl::PointCloud<PointType>::Ptr sub_map_cloud(new pcl::PointCloud<PointType>());
        *sub_map_cloud += *laserCloudCornerFromMapDS;
        *sub_map_cloud += *laserCloudSurfFromMapDS;

        sensor_msgs::PointCloud2 tmpCloudMsg;
        pcl::toROSMsg(*sub_map_cloud, tmpCloudMsg);
        tmpCloudMsg.header = cloudInfo.header;
        tmpCloudMsg.header.frame_id = mapFrame;
        pubStationaryCloud.publish(tmpCloudMsg);
    }

    /**
     * 保存地图服务
    */
    bool saveMapService(v_losm::save_mapRequest &req, v_losm::save_mapResponse &res)
    {
        string saveMapDirectory;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // 请求中没有路径则使用配置中的默认值
        if (req.destination.empty())
            saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
        else
            saveMapDirectory = std::getenv("HOME") + req.destination;
        cout << "Save destination: " << saveMapDirectory << endl;
        // 下面这两行代码会删除旧文件，要注意
        // int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
        // unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
        // 保存历史关键帧位姿
        pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
        // 提取历史关键帧角点、平面点集合
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
        {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *globalSurfCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }

        if (req.resolution != 0)
        {
            cout << "\n\nSave resolution: " << req.resolution << endl;

            // 降采样
            downSizeFilterCorner.setInputCloud(globalCornerCloud);
            downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterCorner.filter(*globalCornerCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
            // 降采样
            downSizeFilterSurf.setInputCloud(globalSurfCloud);
            downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
            downSizeFilterSurf.filter(*globalSurfCloudDS);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
        }
        else
        {
            pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
        }

        // 保存到一起，全局关键帧特征点集合
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;

        int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
        res.success = ret == 0;

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed\n"
             << endl;
        return true;
    }

    /**
     * 回环检测独立线程
     * 1. 由于回环检测中用到了点云匹配，较为耗时，所以独立为单独的线程运行
     * 2. 新的回环关系被检测出来时被主线程加入因子图中优化
     */
    void loopClosureThread()
    {
        // 回环检测是否开启的标志位
        if (!loopClosureEnableFlag)
        {
            return;
        }

        cout<<"回环启动！"<<endl;

        // 设置循环的执行频率，默认1hz
        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            // 执行回环检测
            performLoopClosure();
            // 可视化回环检测结果
            visualizeLoopClosure();
        }
    }

    /**
     * 回环检测函数
     * 1. 关键帧队列为空，直接返回
     * 2. 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
     * 3. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。回环关系用一个全局map缓存
     * 4. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
     * 5. 将当前帧转到Map坐标系并降采样
     * 6. 对匹配帧前后几帧转换到Map坐标系下，融合并降采样，构建局部地图
     * 7. 调用ICP降当前帧匹配到局部地图，得到当前帧位姿的偏差，将偏差应用到当前帧的位姿，得到修正后的当前帧位姿。
     * 8. 根据修正后的当前帧位姿和匹配帧的位姿，计算帧间相对位姿，这个位姿被用来作为回环因子。同时，将ICP的匹配分数当作因子的噪声模型
     * 9. 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
     */
    void performLoopClosure()
    {

        // 1. 关键帧队列为空，直接返回
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // 2. 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // 当前关键帧索引
        int loopKeyCur;
        // 候选闭环匹配帧索引
        int loopKeyPre;
        // 3. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。如果找到的回环对应帧相差时间过短也返回false。回环关系用一个全局map缓存
        // 4. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        {
            // 没有检测到回环就return
            cout<< "没检测到回环！！！"<< endl;
            return;
        }
        cout << "检测到回环！！！" << endl;

        // 提取点云
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 5. 将当前帧转到Map坐标系并降采样，注意这里第三个参数是0, 也就是不加上前后其他帧
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 6. 对匹配帧前后几帧转换到Map坐标系下，融合并降采样，构建局部地图
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            // 如果特征点较少，返回
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
        }

        cout << "开启icp！！！" << endl;
        // 7. 调用ICP降当前帧匹配到局部地图，得到当前帧位姿的偏差，将偏差应用到当前帧的位姿，得到修正后的当前帧位姿。
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // scan-to-map，调用icp匹配
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 未收敛，或者匹配不够好
        cout << "hasConverged!!!!!!!!!!!!!!!!!" << icp.hasConverged() << endl;
        cout << "getFitnessScore!!!!!!!!!!!!!!!!!!!!" << icp.getFitnessScore() << endl;
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        cout << "结束icp！！！" << endl;
        // 8. 根据修正后的当前帧位姿和匹配帧的位姿，计算帧间相对位姿，这个位姿被用来作为回环因子。同时，将ICP的匹配分数当作因子的噪声模型
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();

        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // 9. 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * @brief 根据位置关系寻找当前帧与对应帧的索引
     * 1. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。如果找到的回环对应帧相差时间过短也返回false。回环关系用一个全局map缓存
     * 2. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
     *
     * @param latestID 传出参数，找到的当前帧索引，实际就是用最后一帧关键帧
     * @param closestID 传出参数，找到的当前帧对应的匹配帧
     */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 最新一帧关键帧作为当前帧
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // 确认当前帧没有被加入过回环关系中
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // 将关键帧的3D位置构建kdtree，并检索空间位置相近的关键帧
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 寻找空间距离相近的关键帧
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        // 确保空间距离相近的帧是较久前采集的，排除是前面几个关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        // 如果没有找到位置关系、时间关系都符合要求的关键帧，则返回false
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @brief 根据当前帧索引key，从前后多帧（searchNum）构建局部地图
     * @param nearKeyframes 传出参数，构建出的局部地图
     * @param key 当前帧的索引
     * @param searchNum 从当前帧的前后各searchNum个关键帧构建局部点云地图
     */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr &nearKeyframes, const int &key, const int &searchNum)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize)
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // 降采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * 可视化回环关系，主要是根据回环关系的构建Rivz可以直接显示的MarkerArray
     */
    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        // 闭环顶点
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = mapFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // 闭环边
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = mapFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        // 遍历闭环
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    /**
     * 展示线程
     * 1、发布局部关键帧map的特征点云
     * 2、保存全局关键帧特征点集合
     */
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(5);
        while (ros::ok())
        {
            rate.sleep();
            // 发布局部关键帧map的特征点云
            publishGlobalMap();
        }
    }

    /**
     * 发布局部关键帧map的特征点云
     */
    void publishGlobalMap()
    {
        // cout << "start pub time " << fixed << ros::Time::now().toSec() << endl;
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 半径为globalMapVisualizationSearchRadius的其他关键帧
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // 地图位姿降采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i)
        {
            // 距离过大
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // 提取出来的点云降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;  // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.2, 0.2, 0.2); // for global map visualization
        // downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "map");
        // cout << "end pub time " << fixed << ros::Time::now().toSec() << endl;
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "mapOptimization");

    // 创建实例
    MapOptimization MO;

    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    // 启动回环检测线程
    // std::thread loopthread(&MapOptimization::loopClosureThread, &MO);

    // // 全局地图显示线程
    // std::thread visualizeMapThread(&MapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    // loopthread.join();

    // visualizeMapThread.join();

    // 节点关闭时关闭 CSV 文件
    // csv_file.close();

    return 0;
}