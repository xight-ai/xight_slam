//**** 设计步骤 ****//
// submap 融合多帧获取超分辨点云
//*****************// 

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <queue>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <tf/tf.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/common/transforms.h>

#include <ceres/ceres.h>

#include "v_losm/lshapedFitting.h"
#include "v_losm/cloud_info.h"
#include "v_losm/utility.h"

using namespace std;


struct BoundingData{
    int index_min;
    int index_max;
    float dist2_min = 10000000.0;
    float dist2_max = -1.0;
};

struct BoundPointType
{
    PCL_ADD_POINT4D
    uint st_idx;
    uint ed_idx;
    float width;
    float length;
    float height;
    float yaw;
    float v_x;
    float v_y;
    float v_z;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (BoundPointType,
    (float, x, x) (float, y, y) (float, z, z) (uint, st_idx, st_idx) (uint, ed_idx, ed_idx)
    (float, width, width) (float, length, length) (float, height,height) (float, yaw, yaw)
    (float, v_x, v_x) (float, v_y, v_y) (float, v_z, v_z)
)

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

struct CURVE_FITTING_COST_FOR_MODEL_VELOCITY
{
    CURVE_FITTING_COST_FOR_MODEL_VELOCITY(double x,double y,double z,double v,double Vx,double Vy,double Vz):_x(x),_y(y),_z(z),_v(v),_Vx(Vx),_Vy(Vy),_Vz(Vz){}

    template <typename T>
    bool operator()(
        const T* const model_v,
        T* residual ) const
    {
        residual[0] = (T(_Vx)*T(_x)+T(_Vy)*T(_y)+T(_Vz)*T(_z))/sqrt(T(_x)*T(_x)+T(_y)*T(_y)+T(_z)*T(_z)) + 
                       T(_v) - (model_v[0]*T(_x)+model_v[1]*T(_y)+model_v[2]*T(_z))/sqrt(T(_x)*T(_x)+T(_y)*T(_y)+T(_z)*T(_z));
        
        residual[1] = model_v[0]*T(0.0)+model_v[1]*T(0.0)+model_v[2]*T(1.0);
        return true;
    }
    const double _x,_y,_z,_v,_Vx,_Vy,_Vz;
};

typedef long long myint64;

class SpawnStaticMap: public ParamServer{
private:
    ros::NodeHandle nh;
    // 订阅点云
    ros::Subscriber sub;
    // 发布静态点云
    ros::Publisher pubStaticObjCloud;
    // 发布动态点云
    ros::Publisher pubMovingObjCloud;
    // 发布局部地图
    ros::Publisher pubSubmapCloud;
    // 发布物体检测框
    ros::Publisher pubBox;
    // 发布速度显示的maker
    ros::Publisher pubMarker;
    // 发布处理后的10帧融合
    ros::Publisher pubFrameMix;
    // 发布原始的10帧融合
    ros::Publisher pubOrgMix;
    

    /**
     * cloudKeyPoses3D保存所有关键帧的三维位姿，x,y,z
     * cloudKeyPoses6D保存所有关键帧的六维位姿，x,y,z,roll,pitch,yaw
     * 带copy_前缀的两个位姿序列是在回环检测线程中使用的，只是为了不干扰主线程计算，实际内容一样。
     */
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
    // 历史所有关键帧的静止点集合
    vector<pcl::PointCloud<PointType>::Ptr> staticCloudKeyFrames;
    // 历史所有关键帧的运动点集合
    vector<pcl::PointCloud<PointType>::Ptr> movingCloudKeyFrames;

    // 当前雷达帧的时间戳
    ros::Time timeLaserInfoStamp;

    // 雷达高度
    const float lidarHeight = 2;
    // 核心数
    const int numberOfCores = 6;

    // 阈值
    const float distThreshold = 1.2;
    const float velThreshold = 1.0;

    // 两帧之间的间隔时间（秒）
    float deltaTime = 0.1;

    // 是否初始化
    bool is_init = false;

    // 信号量
    std::mutex mtx;

    std_msgs::Header thisHeader;

    // cloud points
    // cloud info
    v_losm::cloud_info cloudInfo;
    pcl::PointCloud<FMCWPointType>::Ptr cloud;
    // 当前帧的静止物体
    pcl::PointCloud<FMCWPointType>::Ptr cloud_static;
    // 当前帧的运动物体
    pcl::PointCloud<FMCWPointType>::Ptr cloud_moving;
    // 上一帧的运动物体
    pcl::PointCloud<FMCWPointType>::Ptr cloud_moving_last;
    // 局部地图
    pcl::PointCloud<FMCWPointType>::Ptr cloud_submap;
    // 当前帧的原始点云
    pcl::PointCloud<FMCWPointType>::Ptr cloud_org;
    // 原始点云的帧间融合
    pcl::PointCloud<FMCWPointType>::Ptr cloud_org_mix;
    // 运动物体中心点、速度、索引等信息
    pcl::PointCloud<BoundPointType>::Ptr clusters_bound;

    // 聚类出来的物体点云
    vector<pcl::PointCloud<FMCWPointType>::Ptr> clusters;
    // 半动态物体聚类前物体的个数
    int boundSt;

    // 保存运动点云用于运动物体去畸变
    vector<pcl::PointCloud<FMCWPointType>::Ptr> cloud_moving_vector_deskew;
    // 保存运动物体的信息
    vector<pcl::PointCloud<BoundPointType>::Ptr> clusters_bound_vector_deskew;
    // 用于存放位姿变换
    vector<vector<float>> transform_between_reserve;

    // 用于原始帧间融合的队列
    queue<pcl::PointCloud<FMCWPointType>::Ptr> orgFrameQueue;
    // 用于原始帧间融合的临时队列
    queue<pcl::PointCloud<FMCWPointType>::Ptr> orgTempQueue;

    // pose
    float transformBetween[6] = {0};

    // self velocity
    double self_v[3] = {0};

    // downsample
    pcl::VoxelGrid<FMCWPointType> submapDownSizeFilter;

    // 局部地图下采样半径大小
    double submapDownsizeLeaf = 0.2;

public:
    SpawnStaticMap()
    {
        // init
        cloud_submap.reset(new pcl::PointCloud<FMCWPointType>());

        // 订阅点云信息
        sub = nh.subscribe<v_losm::cloud_info>("/cloudInfo_odometry", 100, &SpawnStaticMap::subCallback, this, ros::TransportHints().tcpNoDelay());
        // 发布当前帧静态物体
        pubStaticObjCloud = nh.advertise<sensor_msgs::PointCloud2>("/map/static_obj",10);
        // 发布当前帧运动物体
        pubMovingObjCloud = nh.advertise<sensor_msgs::PointCloud2>("/map/moving_obj",10);
        // 发布局部地图
        pubSubmapCloud = nh.advertise<sensor_msgs::PointCloud2>("/map/sub_map",10);
        // 发布当前帧运动物体边界框
        pubBox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/map/bounding_box",1);
        // 发布自身速度的marker
        pubMarker = nh.advertise<visualization_msgs::MarkerArray>("/map/self_velocity_marker",10);
        // 发布10帧原始帧间融合
        pubOrgMix = nh.advertise<sensor_msgs::PointCloud2>("/map/org_mix", 10);
        
        // 关键帧位姿存储初始化
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    }

    void subCallback(const v_losm::cloud_infoConstPtr &lidarCloudMsg)
    {
        // double time1 = ros::Time::now().toSec();

        thisHeader = lidarCloudMsg->header;
        timeLaserInfoStamp = lidarCloudMsg->header.stamp;

        cloudInfo = *lidarCloudMsg; // 数据输入

        // 初始化一些变量
        initialization(); 

        vector<vector<ClusterData>> dataMat(row, vector<ClusterData>(column));

        // 第一帧进行初始化
        if (is_init == false)
        {
            // 获得当前cloudinfo
            getCurrentCloudInfo(dataMat);
            // 预测运动点云下一帧的位置放入cloud_moving_last
            predictMovingClouds();
            // 初始化完成
            is_init = true;
            return;
        }

        // 获得当前cloudinfo
        getCurrentCloudInfo(dataMat);

        // 将上一帧时，预测的运动点云转换到当前帧的坐标系下放入cloud_moving_last
        transformToCurrFrame();

        // 注意，这里的cloud为静止点云       
        pcl::copyPointCloud(*cloud_static, *cloud);    

        // 这里主要是要找上一帧在运动，当前帧为静止的物体
        // 计算预测的运动点在当前帧的周围是否存在点，若存在则作为半动态物体种子点
        vector<int> seeds= extractSeeds(dataMat);
        
        // 根据种子点进行聚类
        motionClustering(seeds, dataMat);

        // 记录当前帧信息
        recordThisKeyFrame();

        // 用lshapedfit算法设置物体边间框
        lShapedFit();

        // 估计运动物体速度
        Estimate_Model_Velocity(self_v, clusters, clusters_bound);
        // 当前帧加入局部地图
        Update_SubMap(dataMat);
        // 估计点云在下一帧的位置
        predictMovingClouds();

        // 显示速度和框
        Show_Velocity_And_Box(self_v, clusters_bound);

        // double time2 = ros::Time::now().toSec();
        // cout << "\033[9;0H" <<"\033[K"<< "SpawnStaticMap time: " << time2 - time1 <<endl;

        // 发布常规点云信息及地图
        publishFinalCloud();

        // 原始帧间融合及发布
        pubOrgMixCloud();

        // cout << "end time " << fixed << ros::Time::now().toSec() << endl;
    }

    /**
     * 初始化
     */
    void initialization()
    {
        cloud.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_static.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_moving.reset(new pcl::PointCloud<FMCWPointType>());
        clusters_bound.reset(new pcl::PointCloud<BoundPointType>());
        cloud_org.reset(new pcl::PointCloud<FMCWPointType>());
        clusters.clear();
    }

    // 将新出现物体的上一帧从局部地图上删除
    inline myint64 getkey(FMCWPointType thisPoint, int m, int n, int h)
    {
        return (myint64((thisPoint.x + 40000) / 0.40 + m)) + (myint64((thisPoint.y + 40000) / 0.40 + n)) * myint64(1e7) + (myint64((thisPoint.z + 200) / 0.20 + h)) * myint64(1e14);

        // return (myint64((thisPoint.x+200000)/0.2)+m) + (myint64((thisPoint.y+200000)/0.2)+n)*1e7;
    }

    inline myint64 getkey(float x, float y, float z)
    {
        return (myint64((x + 40000) / 0.40)) + (myint64((y + 40000) / 0.40)) * myint64(1e7) + (myint64((z + 200) / 0.20)) * myint64(1e14);

        // return (myint64((x+200000)/0.2)) + (myint64((y+200000)/0.2))*1e7;
    }

    /**
     * 从cloudInfo中获取当前帧的信息
     * @param dataMat 二维数组存放标识，在该函数中进行初始化
     */
    void getCurrentCloudInfo(vector<vector<ClusterData>> &dataMat)
    {
        // cloud points
        pcl::fromROSMsg(cloudInfo.cloud_static, *cloud_static);     // 静止点云
        pcl::fromROSMsg(cloudInfo.cloud_moving, *cloud_moving);     // 运动点云
        pcl::fromROSMsg(cloudInfo.cloud_bounding, *clusters_bound); // 运动物体中心点、速度、索引等信息
        // 获取用于帧间融合的原始点云
        pcl::copyPointCloud(*cloud_static, *cloud_org);
        //    pcl::fromROSMsg(cloudInfo.cloud_org, *cloud_org); // 原始点云

        /***************************************************************运动物体去畸变的代码begin*******************************************************************/
        // 拷贝运动点云存入cloud_moving_queue_deskew
        pcl::PointCloud<FMCWPointType>::Ptr cloud_moving_copy;
        cloud_moving_copy.reset(new pcl::PointCloud<FMCWPointType>());
        pcl::copyPointCloud(*cloud_moving, *cloud_moving_copy);
        cloud_moving_vector_deskew.push_back(cloud_moving_copy);
        // 总共存10帧的运动点，大于10帧就把最前面那帧pop掉
        if (cloud_moving_vector_deskew.size() > 10)
        {
            cloud_moving_vector_deskew.erase(cloud_moving_vector_deskew.begin());
        }

        // 拷贝运动物体的信息存入clusters_bound_queue_deskew
        pcl::PointCloud<BoundPointType>::Ptr clusters_bound_copy;
        clusters_bound_copy.reset(new pcl::PointCloud<BoundPointType>());
        pcl::copyPointCloud(*clusters_bound, *clusters_bound_copy);
        clusters_bound_vector_deskew.push_back(clusters_bound_copy);
        // 总共存10帧，大于10帧就把最前面那帧pop掉
        if (clusters_bound_vector_deskew.size() > 10)
        {
            clusters_bound_vector_deskew.erase(clusters_bound_vector_deskew.begin());
        }

        // 保存位姿转换信息
        vector<float> transform_between_temp;
        transform_between_temp.push_back(cloudInfo.transform_between[0]);
        transform_between_temp.push_back(cloudInfo.transform_between[1]);
        transform_between_temp.push_back(cloudInfo.transform_between[2]);
        transform_between_temp.push_back(cloudInfo.transform_between[3]);
        transform_between_temp.push_back(cloudInfo.transform_between[4]);
        transform_between_temp.push_back(cloudInfo.transform_between[5]);
        transform_between_reserve.push_back(transform_between_temp);
        // 同样存10帧
        if (transform_between_reserve.size() > 10)
        {
            transform_between_reserve.erase(transform_between_reserve.begin());
        }

        /***************************************************************运动物体去畸变的代码end*******************************************************************/

        // 相邻两帧间的位姿转换
        transformBetween[0] = cloudInfo.transform_between[0];
        transformBetween[1] = cloudInfo.transform_between[1];
        transformBetween[2] = cloudInfo.transform_between[2];
        transformBetween[3] = cloudInfo.transform_between[3];
        transformBetween[4] = cloudInfo.transform_between[4];
        transformBetween[5] = cloudInfo.transform_between[5];
        // self velocity
        self_v[0] = cloudInfo.self_v[0];
        self_v[1] = cloudInfo.self_v[1];
        self_v[2] = cloudInfo.self_v[2];

        // 初始化dataMat
        int cloudSize = cloud_static->points.size();
        for (int i = 0; i < cloudSize; i++)
        {
            int rowIdx = cloud_static->points[i].ring;
            int columnIdx = cloud_static->points[i].column;

            dataMat[rowIdx][columnIdx].index = i;
            dataMat[rowIdx][columnIdx].selected = -1; // selected表示上一帧是否出现
            dataMat[rowIdx][columnIdx].seeded = -1;
            dataMat[rowIdx][columnIdx].processed = -1;
            dataMat[rowIdx][columnIdx].velocity = cloud_static->points[i].velocity;
        }
    }

    /**
     * 预测运动点云下一帧的位置放入cloud_moving_last
     */
    void predictMovingClouds()
    {
        // 将当前帧运动点放入cloud_moving_last
        cloud_moving_last.reset(new pcl::PointCloud<FMCWPointType>());
        pcl::copyPointCloud(*cloud_moving, *cloud_moving_last);
        // 运动物体个数
        int boundSize = clusters_bound->points.size();
        //
        for (int i = 0; i < boundSize; i++)
        {
            // 运动物体点的个数小于60个则不处理
            if (clusters_bound->points[i].ed_idx - clusters_bound->points[i].st_idx < 60)
            {
                continue;
            }

            // 速度乘时间预测下一时刻位置，然后将cloud_moving_last中对应的点修改成预测的位置
            float dx = clusters_bound->points[i].v_x * 0.1;
            float dy = clusters_bound->points[i].v_y * 0.1;
            float dz = clusters_bound->points[i].v_z * 0.1;
            for (int j = clusters_bound->points[i].st_idx; j < (int)clusters_bound->points[i].ed_idx; j++)
            {
                cloud_moving_last->points[j].x = cloud_moving->points[j].x + dx;
                cloud_moving_last->points[j].y = cloud_moving->points[j].y + dy;
                cloud_moving_last->points[j].z = cloud_moving->points[j].z + dz;
            }
        }
    }

    /**
     * 将预测的运动点云转换到当前帧坐标系下，论文中见前向点级追踪
     */
    void transformToCurrFrame()
    {
        // 上一帧转换到当前帧坐标系
        Eigen::Affine3f trans_cur_to_last = pcl::getTransformation(transformBetween[0], transformBetween[1], transformBetween[2],
                                                                   transformBetween[3], transformBetween[4], transformBetween[5]);
        Eigen::Affine3f rotation_t = trans_cur_to_last.inverse();

        for (int i = 0; i < int(cloud_moving_last->points.size()); i++)
        {
            PointType thisPoint;
            thisPoint.x = rotation_t(0, 0) * cloud_moving_last->points[i].x + rotation_t(0, 1) * cloud_moving_last->points[i].y + rotation_t(0, 2) * cloud_moving_last->points[i].z + rotation_t(0, 3);
            thisPoint.y = rotation_t(1, 0) * cloud_moving_last->points[i].x + rotation_t(1, 1) * cloud_moving_last->points[i].y + rotation_t(1, 2) * cloud_moving_last->points[i].z + rotation_t(1, 3);
            thisPoint.z = rotation_t(2, 0) * cloud_moving_last->points[i].x + rotation_t(2, 1) * cloud_moving_last->points[i].y + rotation_t(2, 2) * cloud_moving_last->points[i].z + rotation_t(2, 3);
            cloud_moving_last->points[i].x = thisPoint.x;
            cloud_moving_last->points[i].y = thisPoint.y;
            cloud_moving_last->points[i].z = thisPoint.z;

            float horizonAngle = atan2(cloud_moving_last->points[i].y, cloud_moving_last->points[i].x) / 3.1415926 * 180;
            float ringAngle = atan2(cloud_moving_last->points[i].z, sqrt(cloud_moving_last->points[i].x * cloud_moving_last->points[i].x + cloud_moving_last->points[i].y * cloud_moving_last->points[i].y)) / 3.1415926 * 180;

            // int rowIdx = round((ringAngle*180/3.1415926+15.0)/ring_res);
            int columnIdx = round((horizonAngle * 180 / 3.1415926 + 60.0) / 0.2);
            int rowIdx;
            if (ringAngle >= -15 && ringAngle < -5)
            {
                rowIdx = round((ringAngle + 15) / 0.5);
            }
            else if (ringAngle >= -5 && ringAngle <= 5)
            {
                rowIdx = round((ringAngle + 5) / 0.2) + 20;
            }
            else
            {
                rowIdx = round((ringAngle - 5) / 0.5 + 71);
            }

            cloud_moving_last->points[i].ring = rowIdx;
            cloud_moving_last->points[i].column = columnIdx;
        }
    }

    /**
     * 提取种子点
    */
    vector<int> extractSeeds(vector<vector<ClusterData>> &dataMat)
    {

        vector<int> seeds;
        seeds.reserve(20000);

        for (int i = 0; i < int(cloud_moving_last->points.size()); i++)
        {
            int rowIdx = cloud_moving_last->points[i].ring;
            int columnIdx = cloud_moving_last->points[i].column;
            if (rowIdx < 0 || rowIdx >= row || columnIdx < 0 || columnIdx >= column)
            {
                continue;
            }
            dataMat[rowIdx][columnIdx].selected = 1;

            // 在dataMat中寻找周围存在的邻近点作为种子点
            for (int m = -2; m <= 2; m++)
            {
                for (int n = -3; n <= 3; n++)
                {
                    // 越界点略过
                    if ((rowIdx + m) < 0 || (rowIdx + m) >= row || (columnIdx + n) < 0 || (columnIdx + n) >= column)
                    {
                        continue;
                    }
                    // 不存在点或者已经成为种子点则略过
                    if (dataMat[rowIdx + m][columnIdx + n].index == -1 || dataMat[rowIdx + m][columnIdx + n].seeded == 1)
                    {
                        continue;
                    }

                    // 记录下目标id
                    int targetIdx = dataMat[rowIdx + m][columnIdx + n].index;

                    // 计算一下两点之间的距离
                    float dist_sub = sqrt((cloud->points[targetIdx].x - cloud_moving_last->points[i].x) * (cloud->points[targetIdx].x - cloud_moving_last->points[i].x) + (cloud->points[targetIdx].y - cloud_moving_last->points[i].y) * (cloud->points[targetIdx].y - cloud_moving_last->points[i].y));
                    // + (cloud->points[targetIdx].z - cloud_moving_last->points[i].z) * (cloud->points[targetIdx].z - cloud_moving_last->points[i].z));

                    // 距离小于阈值20cm则作为种子
                    if (dist_sub < 0.2)
                    {
                        dataMat[rowIdx + m][columnIdx + n].seeded = 1; // 标识为种子点
                        seeds.push_back(targetIdx);                    // 保存目标id
                    }
                }
            }
        }
        return seeds;
    }

    /**
     * 聚类,更新动态点云和静止点云
    */
    void motionClustering(vector<int> seeds, vector<vector<ClusterData>> &dataMat){
        // 过滤一半点云数据,并降采样放入kdtree,用于获取点周围的最低点
        // 使用pcl的PassThrough过滤出z轴-5到-0.5的点云，结果存放在cloud_filter中
        pcl::PointCloud<FMCWPointType>::Ptr cloud_all(new pcl::PointCloud<FMCWPointType>());
        *cloud_all = *cloud_static + *cloud_moving;
        pcl::PointCloud<FMCWPointType>::Ptr cloud_filter(new pcl::PointCloud<FMCWPointType>());
        pcl::PassThrough<FMCWPointType> passThrough;
        passThrough.setFilterFieldName("z");
        passThrough.setFilterLimits(-5.0, -0.5);
        passThrough.setInputCloud(cloud_all);
        passThrough.filter(*cloud_filter);

        // 再进行一波下采样，采用了0.3*0.3*0.3的体素，结果存放在cloud_ds中
        pcl::PointCloud<FMCWPointType>::Ptr cloud_ds(new pcl::PointCloud<FMCWPointType>());
        pcl::VoxelGrid<FMCWPointType> downSizeFilter;
        downSizeFilter.setLeafSize(0.3, 0.3, 0.3);
        downSizeFilter.setInputCloud(cloud_filter);
        downSizeFilter.filter(*cloud_ds);

        // 再把cloud_ds放入cloud_kdtree，用于后面计算地面高度
        pcl::PointCloud<PointType>::Ptr cloud_kdtree(new pcl::PointCloud<PointType>());
        int cloudSize = cloud_ds->points.size();
        cloud_kdtree->points.resize(cloudSize);
        for (int i = 0; i < cloudSize; i++)
        {
            cloud_kdtree->points[i].x = cloud_ds->points[i].x;
            cloud_kdtree->points[i].y = cloud_ds->points[i].y;
            cloud_kdtree->points[i].z = 0.0f;
            cloud_kdtree->points[i].intensity = cloud_ds->points[i].z;
        }

        pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());
        kdtree->setInputCloud(cloud_kdtree);

        // 获取由点级追踪所得半动态物体
        // 根据种子点进行聚类,根据当前点周围的最低点估计地面高度,聚类地面以上的点云
        boundSt = clusters_bound->points.size();
        
        clusters.reserve(100);
        int seedSize = seeds.size();
        for (int i = 0; i < seedSize; i++)
        {
            int rowIdx = cloud->points[seeds[i]].ring;
            int columnIdx = cloud->points[seeds[i]].column;
            // 如果种子点未处理,则首先确定地面高度,再聚类周围点
            if (dataMat[rowIdx][columnIdx].processed == -1)
            {
                // 获取地面高度信息
                float range = sqrt(cloud->points[seeds[i]].x * cloud->points[seeds[i]].x + cloud->points[seeds[i]].y * cloud->points[seeds[i]].y);
                float groundHeight = 0.0;
                if (range < 6.0)
                {
                    // 水平距离8米以内的地面为雷达高度减0.15
                    groundHeight = -lidarHeight + 0.15;
                }
                else
                {
                    groundHeight = Get_Low_Points_Height(kdtree, cloud_kdtree, cloud->points[seeds[i]], range * 0.1);
                }

                // RV视角下直接聚类
                pcl::PointCloud<FMCWPointType>::Ptr cloud_extract(new pcl::PointCloud<FMCWPointType>());
                Search_Points(cloud, dataMat, rowIdx, columnIdx, dataMat[rowIdx][columnIdx].index, groundHeight, cloud_extract);
                // 聚类出来的物体必须大于20个点
                if (cloud_extract->points.size() < 20)
                {
                    continue;
                }
                // 新聚类出来的半动态物体放入clusters中
                clusters.push_back(cloud_extract);
                // objects保存到cloud_moving,并记录每个object的索引
                BoundPointType thisBound;
                thisBound.st_idx = cloud_moving->points.size();
                // 注意这里将半动态物体加入了cloud_moving中，所以后面预测运动点云位置时也包含了半动态物体
                *cloud_moving += *cloud_extract;
                thisBound.ed_idx = cloud_moving->points.size();
                clusters_bound->push_back(thisBound);
            }
        }

        // 更新静止点云，就是将去除了半动态物体的点云更新为静止点
        cloud_static.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_static->points.reserve(90000);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (dataMat[i][j].index == -1 || dataMat[i][j].processed != -1)
                {
                    continue;
                }
                cloud_static->push_back(cloud->points[dataMat[i][j].index]);
            }
        }
    }
    
    /**
     * 记录当前帧信息
    */
    void recordThisKeyFrame(){
        PointType thisPose3D;
        thisPose3D.x = cloudInfo.transform_map[0];
        thisPose3D.y = cloudInfo.transform_map[1];
        thisPose3D.z = cloudInfo.transform_map[2];
        thisPose3D.intensity = cloudKeyPoses3D->size();
        cloudKeyPoses3D->push_back(thisPose3D);

        PointTypePose thisPose6D;
        thisPose6D.x = cloudInfo.transform_map[0];
        thisPose6D.y = cloudInfo.transform_map[1];
        thisPose6D.z = cloudInfo.transform_map[2];
        thisPose6D.intensity = cloudKeyPoses6D->size();
        thisPose6D.roll = cloudInfo.transform_map[3];
        thisPose6D.pitch = cloudInfo.transform_map[4];
        thisPose6D.yaw = cloudInfo.transform_map[5];
        cloudKeyPoses6D->push_back(thisPose6D);

        // 当前帧激光运动点、平静止点
        pcl::PointCloud<PointType>::Ptr thisStaticKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisMovingKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*cloud_static, *thisStaticKeyFrame);
        pcl::copyPointCloud(*cloud_moving, *thisMovingKeyFrame);

        // 保存特征点降采样集合
        staticCloudKeyFrames.push_back(thisStaticKeyFrame);
        movingCloudKeyFrames.push_back(thisMovingKeyFrame);
    }

    float Get_Low_Points_Height(const pcl::KdTreeFLANN<PointType>::Ptr &kdtree,
                                const pcl::PointCloud<PointType>::Ptr &cloud_kdtree,
                                const FMCWPointType &target,
                                float range)
    {
        PointType thisPoint;
        thisPoint.x = target.x;
        thisPoint.y = target.y;
        thisPoint.z = 0.0f;

        vector<int> indices;
        vector<float> sqr_distance;
        // 搜索附近 range 范围的点
        kdtree->radiusSearch(thisPoint, range, indices, sqr_distance);
        // 维护一个大顶堆优先队列,保留高度最小的前10个点
        unordered_map<int, priority_queue<float, vector<float>, less<float>>> height_map;

        // priority_queue<float, vector<float>, less<float>> height_queue;
        int indiceSize = indices.size();
        for (int i = 0; i < indiceSize; i++)
        {
            int angle = atan2(thisPoint.y - cloud_kdtree->points[indices[i]].y, thisPoint.x - cloud_kdtree->points[indices[i]].x) * 3 / 3.1415927;
            height_map[angle].push(cloud_kdtree->points[indices[i]].intensity);
            if (height_map[angle].size() > 5)
            {
                height_map[angle].pop();
            }
        }
        // 计算最小高度的平均值,取 0.1 作为非地面点阈值
        float mean_h = 0, mean_num = 0;
        for (auto iter = height_map.begin(); iter != height_map.end(); iter++)
        {
            mean_num += iter->second.size();
            while (!iter->second.empty())
            {
                mean_h += iter->second.top();
                iter->second.pop();
            }
        }

        return (mean_h / mean_num + 0.2);
    }

    /**
     * 聚类
     * @param cloud 点云
     * @param dataMat 存放处理信息的二维数组
     * @param rowIdx 目标点的行id
     * @param columnIdx 目标点的列id
     * @param pointIdx 种子点在cloud中的索引
     * @param ground 地面高度
     * @param cloud_extract 聚类出来的点放在cloud_extract中
     */
    void Search_Points(pcl::PointCloud<FMCWPointType>::Ptr &cloud,
                       vector<vector<ClusterData>> &dataMat,
                       int rowIdx, int columnIdx, int pointIdx, float ground,
                       pcl::PointCloud<FMCWPointType>::Ptr &cloud_extract)
    {
        if (rowIdx < 0 || rowIdx >= row || columnIdx < 0 || columnIdx >= column)
        {
            return;
        }
        if (dataMat[rowIdx][columnIdx].index == -1 || dataMat[rowIdx][columnIdx].processed != -1)
        {
            return;
        }

        int targetIdx = dataMat[rowIdx][columnIdx].index;
        // 如果目标点高度小于地面阈值,则同样不做处理
        if (cloud->points[targetIdx].z <= ground)
        {
            dataMat[rowIdx][columnIdx].processed = 1;
            return;
        }

        // 计算速度和距离阈值,满足要求则聚类
        float dist_sub = sqrt((cloud->points[targetIdx].x - cloud->points[pointIdx].x) * (cloud->points[targetIdx].x - cloud->points[pointIdx].x) + (cloud->points[targetIdx].y - cloud->points[pointIdx].y) * (cloud->points[targetIdx].y - cloud->points[pointIdx].y) + (cloud->points[targetIdx].z - cloud->points[pointIdx].z) * (cloud->points[targetIdx].z - cloud->points[pointIdx].z));
        float vel_sub = abs(cloud->points[targetIdx].velocity - cloud->points[pointIdx].velocity);

        if ((dist_sub < distThreshold) && (vel_sub < velThreshold))
        {
            dataMat[rowIdx][columnIdx].processed = 1;
            // 符合条件的点加入 cloud_extract 中
            cloud_extract->points.push_back(cloud->points[targetIdx]);
            // 根据该点寻找其周围点
            for (int i = -4; i <= 4; i++)
            {
                for (int j = -6; j <= 6; j++)
                {
                    Search_Points(cloud, dataMat, rowIdx + i, columnIdx + j, targetIdx, ground, cloud_extract);
                }
            }
        }
    }

    /**
     * 进行lshapedfit
    */
    void lShapedFit(){
        // 绘制boundingbox
        LShapedFIT lshaped;
        lshaped.criterion_ = LShapedFIT::VARIANCE;
        for (int i = 0; i < int(clusters.size()); i++)
        {
            // 寻找物体的最低点和最高点
            float minHeight = 100, maxHeight = -100;
            // 找到每一列的最近最远点, 以及物体的最低最高点
            unordered_map<int, BoundingData> bound;
            for (int j = 0; j < int(clusters[i]->points.size()); j++)
            {
                int columnIdx = clusters[i]->points[j].column;
                float dist2 = clusters[i]->points[j].x * clusters[i]->points[j].x + clusters[i]->points[j].y * clusters[i]->points[j].y;
                // 最近最远
                if (bound[columnIdx].dist2_min > dist2)
                {
                    bound[columnIdx].dist2_min = dist2;
                    bound[columnIdx].index_min = j;
                }
                if (bound[columnIdx].dist2_max < dist2)
                {
                    bound[columnIdx].dist2_max = dist2;
                    bound[columnIdx].index_max = j;
                }
                // 最高最低
                if (minHeight > clusters[i]->points[j].z)
                {
                    minHeight = clusters[i]->points[j].z;
                }
                if (maxHeight < clusters[i]->points[j].z)
                {
                    maxHeight = clusters[i]->points[j].z;
                }
            }
            // 进行 L-shaped Fitting
            cv::RotatedRect thisRect;
            if (bound.size() >= 5)
            {
                vector<cv::Point2f> hull_min, hull_max;
                for (auto iter = bound.begin(); iter != bound.end(); iter++)
                {
                    hull_min.push_back(cv::Point2f(clusters[i]->points[iter->second.index_min].x,
                                                   clusters[i]->points[iter->second.index_min].y));

                    hull_max.push_back(cv::Point2f(clusters[i]->points[iter->second.index_max].x,
                                                   clusters[i]->points[iter->second.index_max].y));
                }
                thisRect = lshaped.FitBox(&hull_min, &hull_max);
            }
            else
            {
                float centerX = 0.0, centerY = 0.0;
                for (auto iter = bound.begin(); iter != bound.end(); iter++)
                {
                    centerX += clusters[i]->points[iter->second.index_min].x;
                    centerY += clusters[i]->points[iter->second.index_min].y;
                }
                thisRect = cv::RotatedRect(cv::Point2f(centerX / bound.size(), centerY / bound.size()), cv::Size2f(0.3, 0.3), 0);
            }
            // 匹配完成后设置进clusters_bound中
            clusters_bound->points[boundSt + i].x = thisRect.center.x;
            clusters_bound->points[boundSt + i].y = thisRect.center.y;
            clusters_bound->points[boundSt + i].z = (maxHeight + minHeight) / 2.0;
            clusters_bound->points[boundSt + i].width = thisRect.size.width;
            clusters_bound->points[boundSt + i].length = thisRect.size.height;
            clusters_bound->points[boundSt + i].height = maxHeight - minHeight;
            clusters_bound->points[boundSt + i].yaw = thisRect.angle / 180.0 * 3.14159;
        }
    }

    /**
     * 估计聚类出来的物体的速度
     * @param self_v
     * @param clusters 各个物体的点云
     * @param clusters_bound 各个物体的信息说明
     */
    void Estimate_Model_Velocity(double self_v[],
                                 vector<pcl::PointCloud<FMCWPointType>::Ptr> &clusters,
                                 pcl::PointCloud<BoundPointType>::Ptr &clusters_bound)
    {
        // 物体个数
        int clusterSize = clusters_bound->points.size();
        for (int i = 0; i < clusterSize; i++)
        {
            int N = clusters_bound->points[i].ed_idx - clusters_bound->points[i].st_idx;
            // 物体点的个数要大于40
            if (N < 40)
            {
                continue;
            }

            // 小于80个点直接设置成速度为0
            if (N <= 80)
            {
                clusters_bound->points[i].v_x = 0.0;
                clusters_bound->points[i].v_y = 0.0;
                clusters_bound->points[i].v_z = 0.0;
                continue;
            }

            double model_v[3] = {0, 0, 0};
            ceres::Problem problem;
            for (int j = clusters_bound->points[i].st_idx; j < (int)clusters_bound->points[i].ed_idx; j++)
            {
                if (rand() % N > 600)
                {
                    continue;
                }
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_FOR_MODEL_VELOCITY, 2, 3>(
                        new CURVE_FITTING_COST_FOR_MODEL_VELOCITY(cloud_moving->points[j].x, cloud_moving->points[j].y, cloud_moving->points[j].z, cloud_moving->points[j].velocity, self_v[0], self_v[1], self_v[2])),
                    // new ceres::HuberLoss(1.0),
                    nullptr,
                    model_v);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 2;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // 设置估计出的速度
            clusters_bound->points[i].v_x = model_v[0];
            clusters_bound->points[i].v_y = model_v[1];
            clusters_bound->points[i].v_z = model_v[2];
        }
    }

    void Show_Velocity_And_Box(double self_v[], pcl::PointCloud<BoundPointType>::Ptr &clusters_bound)
    {
        // 发布速度信息
        visualization_msgs::MarkerArray marker;
        // 自身速度信息
        // 显示文字
        visualization_msgs::Marker thisTextMarker;
        thisTextMarker.header = thisHeader;
        thisTextMarker.ns = "text";
        thisTextMarker.id = 0;
        thisTextMarker.lifetime = ros::Duration(0.15);
        thisTextMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        thisTextMarker.action = visualization_msgs::Marker::ADD;
        thisTextMarker.pose.orientation.w = 1.0;
        thisTextMarker.pose.position.x = 0.0;
        thisTextMarker.pose.position.y = 0.0;
        thisTextMarker.pose.position.z = 2.5;
        thisTextMarker.scale.z = 1.5;
        thisTextMarker.color.g = 1.0f;
        thisTextMarker.color.a = 1;
        thisTextMarker.text = "vel:" + floatToString(float(self_v[0])) + " " + floatToString(float(self_v[1])) + " " + floatToString(float(self_v[2]));
        marker.markers.push_back(thisTextMarker);
        // 显示箭头
        visualization_msgs::Marker thisArrowMarker;
        thisArrowMarker.header = thisHeader;
        thisArrowMarker.ns = "arrow";
        thisArrowMarker.lifetime = ros::Duration(0.15);
        thisArrowMarker.id = 0;
        thisArrowMarker.pose.orientation.w = 1.0;
        thisArrowMarker.type = visualization_msgs::Marker::ARROW;
        thisArrowMarker.action = visualization_msgs::Marker::ADD;
        thisArrowMarker.scale.x = 0.15;
        thisArrowMarker.scale.y = 0.3;
        thisArrowMarker.color.g = 1.0f;
        thisArrowMarker.color.a = 1.0;
        geometry_msgs::Point p1, p2;
        p1.x = 0.0;
        p1.y = 0.0;
        p1.z = 0.0;
        p2.x = p1.x + self_v[0] * 0.8;
        p2.y = p1.y + self_v[1] * 0.8;
        p2.z = p1.z + self_v[2] * 0.8;
        thisArrowMarker.points.push_back(p1);
        thisArrowMarker.points.push_back(p2);
        marker.markers.push_back(thisArrowMarker);

        // 目标物体速度信息
        for (int i = 0; i < int(clusters_bound->points.size()); i++)
        {
            if (clusters_bound->points[i].ed_idx - clusters_bound->points[i].st_idx < 80)
            {
                continue;
            }

            // if((clusters_bound->points[i].ed_idx - clusters_bound->points[i].st_idx > 50) &&
            //     (abs(clusters_bound->points[i].v_x)>0.05 ||
            //     abs(clusters_bound->points[i].v_y)>0.05 ||
            //     abs(clusters_bound->points[i].v_z)>0.05)){
            //     clusters_bound->points[i].yaw = atan(clusters_bound->points[i].v_y/clusters_bound->points[i].v_x);
            //     if(clusters_bound->points[i].width < clusters_bound->points[i].length){
            //         float temp = clusters_bound->points[i].length;
            //         clusters_bound->points[i].length = clusters_bound->points[i].width;
            //         clusters_bound->points[i].width = temp;
            //     }
            // }

            // 可视化文字
            visualization_msgs::Marker thisTextMarker;
            thisTextMarker.header = thisHeader;
            thisTextMarker.ns = "text";
            thisTextMarker.id = i + 1;
            thisTextMarker.lifetime = ros::Duration(0.105);
            thisTextMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            thisTextMarker.action = visualization_msgs::Marker::ADD;
            thisTextMarker.pose.orientation.w = 1.0;
            thisTextMarker.pose.position.x = clusters_bound->points[i].x;
            thisTextMarker.pose.position.y = clusters_bound->points[i].y;
            thisTextMarker.pose.position.z = clusters_bound->points[i].z + 2.0;
            thisTextMarker.scale.z = 1.5;
            thisTextMarker.color.r = 1.0f;
            thisTextMarker.color.g = 0.0f;
            thisTextMarker.color.b = 0.0f;
            thisTextMarker.color.a = 1;
            thisTextMarker.text = "size: " + to_string(clusters_bound->points[i].ed_idx - clusters_bound->points[i].st_idx) + "\n" + "v: " + floatToString(clusters_bound->points[i].v_x) + "," + floatToString(clusters_bound->points[i].v_y) + "," + floatToString(clusters_bound->points[i].v_z);
            marker.markers.push_back(thisTextMarker);

            // 可视化箭头
            visualization_msgs::Marker thisArrowMarker;
            thisArrowMarker.header = thisHeader;
            thisArrowMarker.ns = "arrow";
            thisArrowMarker.lifetime = ros::Duration(0.105);
            thisArrowMarker.id = i + 1;
            thisArrowMarker.pose.orientation.w = 1.0;
            thisArrowMarker.type = visualization_msgs::Marker::ARROW;
            thisArrowMarker.action = visualization_msgs::Marker::ADD;
            thisArrowMarker.scale.x = 0.18;
            thisArrowMarker.scale.y = 0.36;
            thisArrowMarker.scale.z = 0.48;
            thisArrowMarker.color.r = 1.0f;
            thisArrowMarker.color.g = 0.0f;
            thisArrowMarker.color.b = 0.0f;
            thisArrowMarker.color.a = 1.0;
            geometry_msgs::Point p1, p2;
            p1.x = clusters_bound->points[i].x;
            p1.y = clusters_bound->points[i].y;
            p1.z = clusters_bound->points[i].z;
            p2.x = p1.x + clusters_bound->points[i].v_x * 0.8; // sq;
            p2.y = p1.y + clusters_bound->points[i].v_y * 0.8; // sq;
            p2.z = p1.z + clusters_bound->points[i].v_z * 0.8; // sq;
            thisArrowMarker.points.push_back(p1);
            thisArrowMarker.points.push_back(p2);
            marker.markers.push_back(thisArrowMarker);
        }

        pubMarker.publish(marker);

        // 发布 boundingbox 数据
        jsk_recognition_msgs::BoundingBoxArray arr_box;
        arr_box.boxes.clear();
        for (int i = 0; i < int(clusters_bound->points.size()); i++)
        {
            jsk_recognition_msgs::BoundingBox box;
            box.header = thisHeader;
            box.pose.position.x = clusters_bound->points[i].x;
            box.pose.position.y = clusters_bound->points[i].y;
            box.pose.position.z = clusters_bound->points[i].z;
            box.dimensions.x = clusters_bound->points[i].width;
            box.dimensions.y = clusters_bound->points[i].length;
            box.dimensions.z = clusters_bound->points[i].height;
            geometry_msgs::Quaternion quat = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, clusters_bound->points[i].yaw);
            box.pose.orientation = quat;

            arr_box.boxes.push_back(box);
        }
        arr_box.header = thisHeader;
        pubBox.publish(arr_box);
    }

    /**
     * 更新局部地图，会进行反向栅格消除
     * @param dataMat
     */
    void Update_SubMap(vector<vector<ClusterData>> &dataMat)
    {
        // if(cloudInfo.saveFrame == 0){
        //     return;
        // }

        // 当前帧降采样
        pcl::PointCloud<FMCWPointType>::Ptr cloud_static_ds(new pcl::PointCloud<FMCWPointType>());
        // pcl::copyPointCloud(*cloud_static, *cloud_static_ds);
        submapDownSizeFilter.setInputCloud(cloud_static);
        submapDownSizeFilter.setLeafSize(submapDownsizeLeaf, submapDownsizeLeaf, submapDownsizeLeaf);
        submapDownSizeFilter.filter(*cloud_static_ds);

        // 添加新帧
        Eigen::Affine3f rotation_t = pcl::getTransformation(cloudInfo.transform_map[0],
                                                            cloudInfo.transform_map[1],
                                                            cloudInfo.transform_map[2],
                                                            cloudInfo.transform_map[3],
                                                            cloudInfo.transform_map[4],
                                                            cloudInfo.transform_map[5]);
        pcl::PointCloud<FMCWPointType>::Ptr thisCloud(new pcl::PointCloud<FMCWPointType>());
        thisCloud->resize(cloud_static_ds->points.size());
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(cloud_static_ds->points.size()); i++)
        {
            thisCloud->points[i].x = rotation_t(0, 0) * cloud_static_ds->points[i].x + rotation_t(0, 1) * cloud_static_ds->points[i].y + rotation_t(0, 2) * cloud_static_ds->points[i].z + rotation_t(0, 3);
            thisCloud->points[i].y = rotation_t(1, 0) * cloud_static_ds->points[i].x + rotation_t(1, 1) * cloud_static_ds->points[i].y + rotation_t(1, 2) * cloud_static_ds->points[i].z + rotation_t(1, 3);
            thisCloud->points[i].z = rotation_t(2, 0) * cloud_static_ds->points[i].x + rotation_t(2, 1) * cloud_static_ds->points[i].y + rotation_t(2, 2) * cloud_static_ds->points[i].z + rotation_t(2, 3);
        }

        // 将当前帧加入map
        *cloud_submap += *thisCloud;

        /**************************************************************原始点云的10帧融合start*******************************************************/
        pcl::PointCloud<FMCWPointType>::Ptr orgCloud(new pcl::PointCloud<FMCWPointType>());
        orgCloud->resize(cloud_org->points.size());
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(cloud_org->points.size()); i++)
        {
            orgCloud->points[i].x = rotation_t(0, 0) * cloud_org->points[i].x + rotation_t(0, 1) * cloud_org->points[i].y + rotation_t(0, 2) * cloud_org->points[i].z + rotation_t(0, 3);
            orgCloud->points[i].y = rotation_t(1, 0) * cloud_org->points[i].x + rotation_t(1, 1) * cloud_org->points[i].y + rotation_t(1, 2) * cloud_org->points[i].z + rotation_t(1, 3);
            orgCloud->points[i].z = rotation_t(2, 0) * cloud_org->points[i].x + rotation_t(2, 1) * cloud_org->points[i].y + rotation_t(2, 2) * cloud_org->points[i].z + rotation_t(2, 3);
            orgCloud->points[i].velocity = cloud_org->points[i].velocity;
        }
        orgFrameQueue.push(orgCloud);
        if (orgFrameQueue.size() > 5)
        {
            orgFrameQueue.front()->clear();
            orgFrameQueue.pop();
        }
        /***************************************************************原始点云的10帧融合end*********************************************************/

        deleteMovingRoute(dataMat, rotation_t);
        
    }

    /**
     * 从地图中删除地图中半动态物体的运动路线
    */
    void deleteMovingRoute(vector<vector<ClusterData>> &dataMat, Eigen::Affine3f rotation_t)
    {
        // 局部地图降采样
        pcl::PointCloud<FMCWPointType>::Ptr cloud_submap_ds(new pcl::PointCloud<FMCWPointType>());
        // pcl::copyPointCloud(*cloud_submap, *cloud_submap_ds);
        submapDownSizeFilter.setInputCloud(cloud_submap);
        submapDownSizeFilter.setLeafSize(submapDownsizeLeaf, submapDownsizeLeaf, submapDownsizeLeaf);
        submapDownSizeFilter.filter(*cloud_submap_ds);

        // 接下来去除先静止后运动的半动态物体
        // 找出新出现的点云，若当前点云在dataMat中没有selected=1的点，则认为该物体为 新加入的物体
        vector<int> new_obj_seeds;
        for (int i = 0; i < int(clusters_bound->points.size()); i++)
        {
            bool is_new = true;
            int st = clusters_bound->points[i].st_idx;
            int ed = clusters_bound->points[i].ed_idx;

            for (int j = st; j < ed; j++)
            {
                int rowIdx = cloud_moving->points[j].ring;
                int columnIdx = cloud_moving->points[j].column;
                if (rowIdx < 0 || rowIdx >= row || columnIdx < 0 || columnIdx >= column)
                {
                    continue;
                }
                if (dataMat[rowIdx][columnIdx].selected == 1)
                {
                    is_new = false;
                    break;
                }
            }

            if (is_new)
            {
                new_obj_seeds.push_back(i);
            }
        }
        // 估计物体上一时刻的位置
        // 记录所占的栅格
        unordered_set<myint64> delete_set; // key 取string类型 int(x/0.1)+int(y/0.1)+int(z/0.1)
        for (int i = 0; i < int(new_obj_seeds.size()); i++)
        {
            int st = clusters_bound->points[new_obj_seeds[i]].st_idx;
            int ed = clusters_bound->points[new_obj_seeds[i]].ed_idx;

            float dx = 0.0;
            float dy = 0.0;
            float dz = 0.0;

            // 物体点的个数大于60才进行处理
            if ((ed - st) >= 60)
            {
                dx = clusters_bound->points[new_obj_seeds[i]].v_x * 0.1;
                dy = clusters_bound->points[new_obj_seeds[i]].v_y * 0.1;
                dz = clusters_bound->points[new_obj_seeds[i]].v_z * 0.1;
            }

            for (int j = st; j < ed; j++)
            {
                FMCWPointType thisPoint;
                thisPoint.x = cloud_moving->points[j].x - dx;
                thisPoint.y = cloud_moving->points[j].y - dy;
                thisPoint.z = cloud_moving->points[j].z - dz;

                for (int m = -2; m <= 2; m++)
                {
                    for (int n = -2; n <= 2; n++)
                    {
                        for (int h = 0; h <= 1; h++)
                        {
                            myint64 key = getkey(thisPoint, m, n, h);
                            delete_set.insert(key);
                        }
                    }
                }
            }
        }

        // 所有运动点走过的地方，点都被消除
        for (int i = 0; i < int(cloud_moving->points.size()); i++)
        {
            FMCWPointType thisPoint;
            thisPoint.x = cloud_moving->points[i].x;
            thisPoint.y = cloud_moving->points[i].y;
            thisPoint.z = cloud_moving->points[i].z;

            for (int m = -2; m <= 2; m++)
            {
                for (int n = -2; n <= 2; n++)
                {
                    for (int h = 0; h <= 1; h++)
                    {
                        myint64 key = getkey(thisPoint, m, n, h);
                        delete_set.insert(key);
                    }
                }
            }
        }

        // 局部地图切割 以及 消除半动态物体
        int num = 0;
        Eigen::Affine3f rotation_t_inv = rotation_t.inverse();
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(cloud_submap_ds->points.size()); i++)
        {
            float x = rotation_t_inv(0, 0) * cloud_submap_ds->points[i].x + rotation_t_inv(0, 1) * cloud_submap_ds->points[i].y + rotation_t_inv(0, 2) * cloud_submap_ds->points[i].z + rotation_t_inv(0, 3);
            float y = rotation_t_inv(1, 0) * cloud_submap_ds->points[i].x + rotation_t_inv(1, 1) * cloud_submap_ds->points[i].y + rotation_t_inv(1, 2) * cloud_submap_ds->points[i].z + rotation_t_inv(1, 3);
            float z = rotation_t_inv(2, 0) * cloud_submap_ds->points[i].x + rotation_t_inv(2, 1) * cloud_submap_ds->points[i].y + rotation_t_inv(2, 2) * cloud_submap_ds->points[i].z + rotation_t_inv(2, 3);
            myint64 key = getkey(x, y, z);
            if (x < -1000.0 || x > 1000.0 || y < -1000 || y > 1000)
            {
                cloud_submap_ds->points[i].x = NAN;
                cloud_submap_ds->points[i].y = NAN;
                cloud_submap_ds->points[i].z = NAN;
            }
            // if(z < -lidarHeight+0.15){
            //     continue;
            // }
            // 点在需要删除的栅格里就进行删除
            if (delete_set.find(key) != delete_set.end())
            {
                cloud_submap_ds->points[i].x = NAN;
                cloud_submap_ds->points[i].y = NAN;
                cloud_submap_ds->points[i].z = NAN;
                num++;
            }
        }
        // cout<<"\033[10;0H" <<"\033[K"<< "num: "<<num<<" new obj:"<<new_obj_seeds.size()<<endl;

        vector<int> indices;
        cloud_submap_ds->is_dense = false;
        pcl::removeNaNFromPointCloud<FMCWPointType>(*cloud_submap_ds, *cloud_submap, indices);

        // // 局部地图切割
        // Eigen::Affine3f rotation_t_inv = rotation_t.inverse();
        // #pragma omp parallel for num_threads(numberOfCores)
        // for(int i=0;i<int(cloud_submap_ds->points.size());i++){
        //     int x = rotation_t_inv(0,0)*cloud_submap_ds->points[i].x+rotation_t_inv(0,1)*cloud_submap_ds->points[i].y+rotation_t_inv(0,2)*cloud_submap_ds->points[i].z+rotation_t_inv(0,3);
        //     int y = rotation_t_inv(1,0)*cloud_submap_ds->points[i].x+rotation_t_inv(1,1)*cloud_submap_ds->points[i].y+rotation_t_inv(1,2)*cloud_submap_ds->points[i].z+rotation_t_inv(1,3);
        //     if(x < -100.0 || x > 100.0 || y < -100 || y > 100){
        //         cloud_submap_ds->points[i].x = NAN;
        //         cloud_submap_ds->points[i].y = NAN;
        //         cloud_submap_ds->points[i].z = NAN;
        //     }
        // }

        // vector<int> indices;
        // cloud_submap_ds->is_dense = false;
        // pcl::removeNaNFromPointCloud<FMCWPointType>(*cloud_submap_ds, *cloud_submap, indices);
    }

    // 发布点云
    void publishFinalCloud(){
        // 发布当前帧静止点云
        sensor_msgs::PointCloud2 StaticObjCloudMsg;
        pcl::toROSMsg(*cloud_static, StaticObjCloudMsg);
        StaticObjCloudMsg.header = thisHeader;
        pubStaticObjCloud.publish(StaticObjCloudMsg);

        // 发布当前帧运动点云
        sensor_msgs::PointCloud2 MovingObjCloudMsg;
        pcl::toROSMsg(*cloud_moving, MovingObjCloudMsg);
        MovingObjCloudMsg.header = thisHeader;
        pubMovingObjCloud.publish(MovingObjCloudMsg);

        // 发布局部地图
        sensor_msgs::PointCloud2 SubmapCloudMsg;
        pcl::toROSMsg(*cloud_submap, SubmapCloudMsg);
        SubmapCloudMsg.header = thisHeader;
        SubmapCloudMsg.header.frame_id = mapFrame;
        pubSubmapCloud.publish(SubmapCloudMsg);
    }

    /**
     * 发布原始帧间融合
    */
    void pubOrgMixCloud()
    {
        /**************************************************************原始帧间融合start****************************************************************/
        // 最终输出的运动点
        pcl::PointCloud<FMCWPointType>::Ptr cloud_moving_out(new pcl::PointCloud<FMCWPointType>());
        // 标识每个物体的处理状态
        vector<vector<bool>> is_processed;

        // 存有两帧以上就开始进行运动物体去畸变处理
        if (clusters_bound_vector_deskew.size() > 2)
        {

            // 初始化处理状态
            for (int i = 0; i < clusters_bound_vector_deskew.size(); i++)
            {
                vector<bool> indicator;
                for (int j = 0; j < clusters_bound_vector_deskew[i]->points.size(); j++)
                {
                    indicator.push_back(false);
                }
                // 存放每个物体的处理状态
                is_processed.push_back(indicator);
            }

            // i为第几帧，j为该帧中第几个物体
            for (int i = 0; i < clusters_bound_vector_deskew.size(); i++)
            {

                // 第i帧没有运动物体就跳过
                if (clusters_bound_vector_deskew[i]->points.size() == 0)
                {
                    continue;
                }

                // 开始遍历第i帧中的物体
                for (int j = 0; j < clusters_bound_vector_deskew[i]->points.size(); j++)
                {
                    // 处理过了就下一个
                    if (is_processed[i][j] == true)
                    {
                        continue;
                    }

                    // 没处理的先设置成处理过了
                    is_processed[i][j] = true;

                    // 针对每个物体设置找到相同物体的帧的id和对应帧中的物体id
                    vector<int> frame_index;
                    vector<int> object_index;

                    // 初始直接将当前物体加入队列
                    frame_index.push_back(i);
                    object_index.push_back(j);

                    // 如果当前帧为最后一帧，则直接加入输出
                    if (i == clusters_bound_vector_deskew.size() - 1)
                    {
                        // frame_index.push_back(i);
                        // object_index.push_back(j);
                        for (int index = clusters_bound_vector_deskew[i]->points[j].st_idx; index < clusters_bound_vector_deskew[i]->points[j].ed_idx; index++)
                        {
                            cloud_moving_out->points.push_back(cloud_moving_vector_deskew[i]->points[index]);
                        }
                        continue;
                    }
                    // 不是最后一帧就进行处理，count表示几帧没找到
                    int count = 0;

                    // 当前物体的中心点
                    pcl::PointXYZ center_tmp;
                    center_tmp.x = clusters_bound_vector_deskew[i]->points[j].x;
                    center_tmp.y = clusters_bound_vector_deskew[i]->points[j].y;
                    center_tmp.z = clusters_bound_vector_deskew[i]->points[j].z;

                    // 当前物体的速度
                    pcl::PointXYZ velocity_tmp;
                    velocity_tmp.x = clusters_bound_vector_deskew[i]->points[j].v_x;
                    velocity_tmp.y = clusters_bound_vector_deskew[i]->points[j].v_y;
                    velocity_tmp.z = clusters_bound_vector_deskew[i]->points[j].v_z;

                    // 从后一帧开始查找同一物体
                    int o = i + 1;
                    while (o <= clusters_bound_vector_deskew.size() - 1)
                    {
                        // 5帧没找到就跳出循环不找了，表示该物体已消失或者已静止
                        if (count >= 5)
                        {
                            break;
                        }

                        // 标识位，表示第o帧中有没有找到相同物体
                        bool found = false;

                        // 第o帧有物体则遍历第o帧查找同一个物体
                        for (int k = 0; k < clusters_bound_vector_deskew[o]->points.size(); k++)
                        {
                            // 第o帧第k个物体的中心点
                            pcl::PointXYZ center_tmp2;
                            center_tmp2.x = clusters_bound_vector_deskew[o]->points[k].x;
                            center_tmp2.y = clusters_bound_vector_deskew[o]->points[k].y;
                            center_tmp2.z = clusters_bound_vector_deskew[o]->points[k].z;

                            // 并且中心点也预测到第o帧坐标系下
                            center_tmp.x += deltaTime * velocity_tmp.x;
                            center_tmp.y += deltaTime * velocity_tmp.y;
                            center_tmp.z += deltaTime * velocity_tmp.z;
                            // 转换到o帧坐标系
                            Eigen::Affine3f trans_cur_to_o = pcl::getTransformation(transform_between_reserve[o][0], transform_between_reserve[o][1], transform_between_reserve[o][2],
                                                                                    transform_between_reserve[o][3], transform_between_reserve[o][4], transform_between_reserve[o][5]);
                            Eigen::Affine3f rotation_t = trans_cur_to_o.inverse();
                            center_tmp.x = rotation_t(0, 0) * center_tmp.x + rotation_t(0, 1) * center_tmp.y + rotation_t(0, 2) * center_tmp.z + rotation_t(0, 3);
                            center_tmp.y = rotation_t(1, 0) * center_tmp.x + rotation_t(1, 1) * center_tmp.y + rotation_t(1, 2) * center_tmp.z + rotation_t(1, 3);
                            center_tmp.z = rotation_t(2, 0) * center_tmp.x + rotation_t(2, 1) * center_tmp.y + rotation_t(2, 2) * center_tmp.z + rotation_t(2, 3);

                            // 用预测的中心点距离判断是不是同一个物体
                            float dist = sqrt((center_tmp.x - center_tmp2.x) * (center_tmp.x - center_tmp2.x) + (center_tmp.y - center_tmp2.y) * (center_tmp.y - center_tmp2.y) + (center_tmp.z - center_tmp2.z) * (center_tmp.z - center_tmp2.z));

                            // 小于0.5米则认为是同一物体
                            if (dist < 0.5)
                            {
                                // 设置已处理
                                is_processed[o][k] = true;
                                // 记录id
                                frame_index.push_back(o);
                                object_index.push_back(k);

                                // 更新中心点和速度
                                center_tmp = center_tmp2;
                                velocity_tmp.x = clusters_bound_vector_deskew[o]->points[k].v_x;
                                velocity_tmp.y = clusters_bound_vector_deskew[o]->points[k].v_y;
                                velocity_tmp.z = clusters_bound_vector_deskew[o]->points[k].v_z;

                                // 找到后 o + 1即进入下一帧
                                o = o + 1;
                                found = true;
                                count = 0;
                                break;
                            }
                        }

                        // o帧中没找到没找到，更新中心点，进入下一帧再找
                        if (found == false)
                        {
                            // 并且中心点也预测到第o帧坐标系下
                            center_tmp.x += deltaTime * velocity_tmp.x;
                            center_tmp.y += deltaTime * velocity_tmp.y;
                            center_tmp.z += deltaTime * velocity_tmp.z;
                            // 转换到o帧坐标系
                            Eigen::Affine3f trans_cur_to_o = pcl::getTransformation(transform_between_reserve[o][0], transform_between_reserve[o][1], transform_between_reserve[o][2],
                                                                                    transform_between_reserve[o][3], transform_between_reserve[o][4], transform_between_reserve[o][5]);
                            Eigen::Affine3f rotation_t = trans_cur_to_o.inverse();
                            center_tmp.x = rotation_t(0, 0) * center_tmp.x + rotation_t(0, 1) * center_tmp.y + rotation_t(0, 2) * center_tmp.z + rotation_t(0, 3);
                            center_tmp.y = rotation_t(1, 0) * center_tmp.x + rotation_t(1, 1) * center_tmp.y + rotation_t(1, 2) * center_tmp.z + rotation_t(1, 3);
                            center_tmp.z = rotation_t(2, 0) * center_tmp.x + rotation_t(2, 1) * center_tmp.y + rotation_t(2, 2) * center_tmp.z + rotation_t(2, 3);

                            o = o + 1;
                            count = count + 1;
                        }
                    }

                    // 遍历完毕且有找到相同物体
                    if (o == clusters_bound_vector_deskew.size() && count <= 5)
                    {
                        for (int k = 0; k < frame_index.size(); k++)
                        {
                            int trans_frame_index = frame_index[k];
                            int trans_object_index = object_index[k];
                            // 针对每个点转换到当前帧
                            for (int point_id = clusters_bound_vector_deskew[trans_frame_index]->points[trans_object_index].st_idx;
                                 point_id < clusters_bound_vector_deskew[trans_frame_index]->points[trans_object_index].ed_idx; ++point_id)
                            {

                                FMCWPointType point = cloud_moving_vector_deskew[trans_frame_index]->points[point_id];
                                for (int d = trans_frame_index; d < clusters_bound_vector_deskew.size() - 1; ++d)
                                {

                                    point.x += deltaTime * clusters_bound_vector_deskew[trans_frame_index]->points[trans_object_index].v_x;
                                    point.y += deltaTime * clusters_bound_vector_deskew[trans_frame_index]->points[trans_object_index].v_y;
                                    point.z += deltaTime * clusters_bound_vector_deskew[trans_frame_index]->points[trans_object_index].v_z;

                                    // 转换到下一帧坐标系
                                    Eigen::Affine3f trans = pcl::getTransformation(transform_between_reserve[d + 1][0], transform_between_reserve[d + 1][1], transform_between_reserve[d + 1][2],
                                                                                   transform_between_reserve[d + 1][3], transform_between_reserve[d + 1][4], transform_between_reserve[d + 1][5]);
                                    Eigen::Affine3f rotation_t = trans.inverse();
                                    point.x = rotation_t(0, 0) * point.x + rotation_t(0, 1) * point.y + rotation_t(0, 2) * point.z + rotation_t(0, 3);
                                    point.y = rotation_t(1, 0) * point.x + rotation_t(1, 1) * point.y + rotation_t(1, 2) * point.z + rotation_t(1, 3);
                                    point.z = rotation_t(2, 0) * point.x + rotation_t(2, 1) * point.y + rotation_t(2, 2) * point.z + rotation_t(2, 3);
                                }
                                cloud_moving_out->points.push_back(point);
                            }
                        }
                    }
                }
            }
        }

        cloud_org_mix.reset(new pcl::PointCloud<FMCWPointType>());
        while (!orgFrameQueue.empty())
        {
            *cloud_org_mix += *orgFrameQueue.front();
            orgTempQueue.push(orgFrameQueue.front());
            orgFrameQueue.pop();
        }

        while (!orgTempQueue.empty())
        {
            orgFrameQueue.push(orgTempQueue.front());
            orgTempQueue.pop();
        }
        *cloud_org_mix += *cloud_moving_out;

        // 添加新帧
        Eigen::Affine3f rotation_lidar_to_map = pcl::getTransformation(cloudInfo.transform_map[0],
                                                            cloudInfo.transform_map[1],
                                                            cloudInfo.transform_map[2],
                                                            cloudInfo.transform_map[3],
                                                            cloudInfo.transform_map[4],
                                                            cloudInfo.transform_map[5]);
        Eigen::Affine3f rotation_map_to_lidar = rotation_lidar_to_map.inverse();
        pcl::PointCloud<FMCWPointType>::Ptr orgTempCloud(new pcl::PointCloud<FMCWPointType>());
        orgTempCloud->resize(cloud_org_mix->points.size());
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < int(cloud_org_mix->points.size()); i++)
        {
            orgTempCloud->points[i].x = rotation_map_to_lidar(0, 0) * cloud_org_mix->points[i].x + rotation_map_to_lidar(0, 1) * cloud_org_mix->points[i].y + rotation_map_to_lidar(0, 2) * cloud_org_mix->points[i].z + rotation_map_to_lidar(0, 3);
            orgTempCloud->points[i].y = rotation_map_to_lidar(1, 0) * cloud_org_mix->points[i].x + rotation_map_to_lidar(1, 1) * cloud_org_mix->points[i].y + rotation_map_to_lidar(1, 2) * cloud_org_mix->points[i].z + rotation_map_to_lidar(1, 3);
            orgTempCloud->points[i].z = rotation_map_to_lidar(2, 0) * cloud_org_mix->points[i].x + rotation_map_to_lidar(2, 1) * cloud_org_mix->points[i].y + rotation_map_to_lidar(2, 2) * cloud_org_mix->points[i].z + rotation_map_to_lidar(2, 3);
            orgTempCloud->points[i].velocity = cloud_org_mix->points[i].velocity;
        }
        sensor_msgs::PointCloud2 OrgMixMsg;
        pcl::toROSMsg(*orgTempCloud, OrgMixMsg);
        OrgMixMsg.header = thisHeader;
        // 地图坐标系
        OrgMixMsg.header.frame_id = lidarFrame;
        pubOrgMix.publish(OrgMixMsg);
        /**************************************************************原始帧间融合end****************************************************************/
    }


};

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "SpawnStaticMap");
    
    SpawnStaticMap ssm;

    ROS_INFO("\033[1;32m----> SpawnStaticMap Started.\033[0m");

    // 全局地图显示线程
    // std::thread visualizeMapThread(&SpawnStaticMap::visualizeGlobalMapThread, &ssm);

    ros::spin();

    // visualizeMapThread.join();

    return 0;
}

