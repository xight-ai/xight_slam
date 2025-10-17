//**** 设计步骤 ****//
// 1、计算自身速度
// 2、获取种子点
// 3、种子点捕获周围最低点，作为地面点
// 4、种子点聚类周围非地面点，得到运动物体
//*****************//

#define PCL_NO_PRECOMPILE

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <set>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/search.hpp>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>

#include "v_losm/lshapedFitting.h"
#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <tf/tf.h>

#include "v_losm/cloud_info.h"
#include "v_losm/utility.h"

using namespace std;

struct BoundingData
{
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

POINT_CLOUD_REGISTER_POINT_STRUCT(BoundPointType,
                                  (float, x, x)(float, y, y)(float, z, z)(uint, st_idx, st_idx)(uint, ed_idx, ed_idx)(float, width, width)(float, length, length)(float, height, height)(float, yaw, yaw)(float, v_x, v_x)(float, v_y, v_y)(float, v_z, v_z))

struct FMCWPointType
{
    PCL_ADD_POINT4D
    float velocity;
    double time;
    uint ring;
    uint column;
    float column_angle;
    float ring_angle;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(FMCWPointType,
                                  (float, x, x)(float, y, y)(float, z, z)(float, velocity, velocity)(double, time, time)(uint, ring, ring)(uint, column, column)(float, column_angle, column_angle)(float, ring_angle, ring_angle))

typedef pcl::PointXYZI PointType;

struct CURVE_FITTING_COST_SELF_VELOCITY
{
    CURVE_FITTING_COST_SELF_VELOCITY(double x, double y, double z, double v) : _x(x), _y(y), _z(z), _v(v) {}

    template <typename T>
    bool operator()(
        const T *const abc,
        T *residual) const
    {
        residual[0] = T(_v) * sqrt(T(_x) * T(_x) + T(_y) * T(_y) + T(_z) * T(_z)) + abc[0] * T(_x) + abc[1] * T(_y) + abc[2] * T(_z);
        // residual[0] = T(_v)*sqrt(T(_x)*T(_x)+T(_y)*T(_y)+T(_z)*T(_z))+abc[0]*T(_x)+abc[1]*T(_y)+T(0.0)*T(_z);

        residual[1] = abc[0] * T(0.0) + abc[1] * T(0.0) + abc[2] * T(1.0);
        return true;
    }
    const double _x, _y, _z, _v;
};

class FastClustering : public ParamServer
{
private:
    ros::NodeHandle nh;
    // 订阅点云信息
    ros::Subscriber subCloud;
    // 发布静止点云
    ros::Publisher pubStaticObjCloud;
    // 发布动态点云
    ros::Publisher pubMovingObjCloud;
    // 动态物体识别框发布
    ros::Publisher pubBox;
    // 自身速度marker发布
    ros::Publisher pubMarker;
    // 点云信息的发布
    ros::Publisher pubCloudInfo;

    // 自身速度估计时行列的分块 (zwp没用这两个参数)
    const int row_velocity_es = 3;
    const int column_velocity_es = 20;

    // 聚类时的距离阈值和速度阈值
    const float distThreshold = 1.2;
    const float velThreshold = 1.0;

    // 每行用于速度估计的点的个数
    // const float pointsEveryRow = 250; // 每行250个点用于速度估计
    const float pointsEveryRow = 20; // 每行50个点用于速度估计

    // 雷达距离地面的高度，单位m
    const float lidarHeight = 2;

    // 速度差阈值，用来判断是否为运动点
    const float velocityThreshold = 1.0;

    // 聚类出来的物体最小点数
    const int clusterPointsThreshold = 20;

    // 存放上一帧自身速度
    double last_self_v[3] = {0,0,0};

    // 用于存放当前帧自身速度
    double self_v[3] = {0,0,0};

    // 是否初始化
    bool is_inited = false;

    // 自身速度突变的阈值
    double check_self_v_threshold = 3.0;

    // 用来记录行号的集合
    std::set<int> rowIndexRecord;

    // 当前雷达帧的header
    std_msgs::Header thisHeader;

    // 当前输入的点云
    pcl::PointCloud<FMCWPointType>::Ptr cloud;

    // 当前帧的运动点云
    pcl::PointCloud<FMCWPointType>::Ptr cloud_moving;

    // 当前帧的静止点云
    pcl::PointCloud<FMCWPointType>::Ptr cloud_static;

    // 聚类出来的物体详细说明
    pcl::PointCloud<BoundPointType>::Ptr clusters_bound;

    // 聚类出来的物体点云
    vector<pcl::PointCloud<FMCWPointType>::Ptr> clusters;

public:
    FastClustering() 
    {
        // 订阅点云数据
        subCloud = nh.subscribe<v_losm::cloud_info>("/deskew/cloud_info", 100, &FastClustering::subCallback, this, ros::TransportHints().tcpNoDelay());
        // 发布静止点云
        pubStaticObjCloud = nh.advertise<sensor_msgs::PointCloud2>("/static_obj", 10);
        // 发布动态点云
        pubMovingObjCloud = nh.advertise<sensor_msgs::PointCloud2>("/moving_obj", 10);
        // 动态物体识别框发布
        pubBox = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("/bounding_box", 10);
        // 自身速度marker发布
        pubMarker = nh.advertise<visualization_msgs::MarkerArray>("/self_velocity_marker", 10);
        // 点云信息的发布
        pubCloudInfo = nh.advertise<v_losm::cloud_info>("/cloudInfo_cluster", 10);

        allocateMemory();
    }

    /**
     * @brief 为动态指针和动态数组分配内存
     */
    void allocateMemory(){
        cloud.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_moving.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_static.reset(new pcl::PointCloud<FMCWPointType>());
        clusters_bound.reset(new pcl::PointCloud<BoundPointType>());

        resetParameters();
    }

    /**
     * 重置参数
    */
    void resetParameters()
    {
        cloud->clear();
        cloud_moving->clear();
        cloud_static->clear();
        clusters_bound->clear();
        clusters.clear();

        rowIndexRecord.clear();
    }

    void subCallback(const v_losm::cloud_infoConstPtr &CloudInfoMsg)
    {
        thisHeader = CloudInfoMsg->header;
        // dataMat 初始化，注意dataMat中ClusterData的index默认就是-1的
        vector<vector<ClusterData>> dataMat(row, vector<ClusterData>(column));

        // 将msg转化为cloud
        pcl::fromROSMsg(CloudInfoMsg->cloud_deskewed, *cloud);

        // cout << "进来的点：" << cloud->points.size() <<endl;

        // 将点云存成pcd
        // double ts = ros::Time::now().toSec();
        // pcl::io::savePCDFileASCII("/home/zwp/slam/bag/70/pcd/"+std::to_string(ts) + ".pcd", *cloud);

        // 记录时间
        // double time1 = ros::Time::now().toSec();

        // 初始化dataMat
        initDataMat(dataMat);

        int count = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if(dataMat[i][j].index == -1){
                    continue;
                }else{
                    count++;
                }
            }
        }
        
        // cout << "初始化后的点：" << count << endl;

        // 加了这个去噪会漂
        // sampleDeNoise(dataMat);

        int count2 = 0;
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (dataMat[i][j].index == -1)
                {
                    continue;
                }
                else
                {
                    count2++;
                }
            }
        }

        // cout << "去噪后的点：" << count2 << endl;

        // 自身速度估计
        // Estimate_Self_Velocity_1(cloud, dataMat, self_v);
        // Estimate_Self_Velocity_2(cloud, dataMat, self_v);
        // Estimate_Self_Velocity_3(cloud, self_v);
        Estimate_Self_Velocity_zwp(cloud, dataMat, self_v);

        // cout << "测速后的点：" << count2 << endl;

        // 防止速度突变
        if(is_inited){
            checkSelfV();
        }else{
            is_inited = true;
        }

        // 速度估计用时
        // double time2 = ros::Time::now().toSec();
        // cout << "\033[1;0H" <<"\033[K"<< "Self Velocity Estimation cost: "<< time2-time1 <<" s."<<endl;

        // 根据自身速度提取运动种子点
        vector<int> seeds = extractSeeds(dataMat, self_v);

        // 用运动种子点聚类
        motionClustering(seeds, dataMat);
 
        // 聚类用时
        // double time3 = ros::Time::now().toSec();
        // cout << "\033[3;0H" <<"\033[K"<< "Clustering time cost: "<< time3-time2 <<" s."<<endl;

        // 发布相关点云
        publishClouds(CloudInfoMsg, self_v);

        // 显示速度
        showVelocity(self_v);

        resetParameters();

        last_self_v[0] = self_v[0];
        last_self_v[1] = self_v[1];
        last_self_v[2] = self_v[2];
    }

    /**
     * 初始化dataMat
    */
    void initDataMat(vector<vector<ClusterData>> &dataMat)
    {
        // 初始化dataMat
        for (int i = 0; i < cloud->points.size(); i++)
        {
            int rowIdx = cloud->points[i].ring;
            int columnIdx = cloud->points[i].column;

            // 记录行号
            rowIndexRecord.insert(rowIdx);


            // 60行，10000列之外的点不要了
            if (rowIdx < 0 || rowIdx >= row || columnIdx < 0 || columnIdx >= column)
            {
                continue;
            }
            // 若给的数据行列相同的点很多，那这里就被覆盖掉了, 实际测试下损失了近1/4的点
            dataMat[rowIdx][columnIdx].index = i;
            dataMat[rowIdx][columnIdx].selected = -1;
            dataMat[rowIdx][columnIdx].seeded = -1;
            dataMat[rowIdx][columnIdx].processed = -1;
            dataMat[rowIdx][columnIdx].velocity = cloud->points[i].velocity;
        }

    }

    /**
     * 简单的去噪
    */
    void sampleDeNoise(vector<vector<ClusterData>> &dataMat)
    {
        // 做一步简单的去噪，即判断一个点周围有没有速度相似的点，有两个以上速度相似则认为不是噪声
        for(int i=0;i<row;i++){
            for(int j=0;j<column;j++){
                // 不存在点就跳过
                if(dataMat[i][j].index == -1){
                    continue;
                }
                
                // 隧道过滤
                float thisVel = dataMat[i][j].velocity;

                // z轴大于5的进行去除
                // float thisZ = cloud->points[dataMat[i][j].index].z;
                // if(thisZ > 5){
                //     dataMat[i][j].index = -2;
                // }

                // 速度大于10的去除
                // if(abs(thisVel) > 10){
                //     dataMat[i][j].index = -2;
                // }

                // 去除了40m开外的点
                // float thisx = cloud->points[dataMat[i][j].index].x;
                // float thisy = cloud->points[dataMat[i][j].index].y;
                // if(sqrt(pow(thisx, 2) + pow(thisy,2)) > 40 ){
                //     dataMat[i][j].index = -2;
                // }

                // 比较该点与周围点的多普勒速度，如果相差太大就把该点过滤掉
                int is_noise = 0;
                for(int m=-1;m<=1;m++){
                    for(int n=-1;n<=1;n++){
                        if(m==0&&n==0){
                            continue;
                        }

                        int rowIdx = i+m;
                        int columnIdx = j+n;

                        if(rowIdx<0 || rowIdx>=row || columnIdx<0 || columnIdx>=column){
                            continue;
                        }
                        if(dataMat[rowIdx][columnIdx].index==-1){
                            continue;
                        }
                        
                        if(abs(thisVel-dataMat[rowIdx][columnIdx].velocity)<1){
                            is_noise ++;
                        }
                    }
                }
                // 周围有至少2个多普勒速度相似的点，则认为该点不是噪声,为噪声index设置成-2
                if(is_noise<2){
                    dataMat[i][j].index = -2;
                }
            }
        }

        // 又把index为-2的统一设置成了-1
        for(int i=0;i<row;i++){
            for(int j=0;j<column;j++){
                if(dataMat[i][j].index == -2){
                    dataMat[i][j].index = -1;
                }
            }
        }
    }

    /**
     * 自身速度估计1
     * @param cloud 点云
     * @param dataMat 点处理状态记录矩阵
     * @param self_v 自身速度估计结果存储
     */
    void Estimate_Self_Velocity_1(pcl::PointCloud<FMCWPointType>::Ptr &cloud,
                                  vector<vector<ClusterData>> &dataMat,
                                  double self_v[])
    {
        // 将 60*10000 划分为 3*20 个区域，每个区域大小为20*500，取其中70个点。一共取1400个点。
        // 使用柯西核函数降低误差
        int xx = row / row_velocity_es;
        int yy = column / column_velocity_es;
        ceres::Problem problem;
        for (int i = 0; i < row_velocity_es; i++)
        {
            // 开始的行索引
            int rowSt = i * xx;
            for (int j = 0; j < column_velocity_es; j++)
            {
                // 开始的列索引
                int columnSt = j * yy;

                for (int num = 0; num < 70;)
                {
                    // 随机选择行和列
                    srand(time(0) + rand());
                    int rowChosen = rowSt + rand() % xx;
                    int columnChosen = columnSt + rand() % yy;

                    if (dataMat[rowChosen][columnChosen].selected == 1)
                    {
                        continue;
                    }
                    else
                    {
                        dataMat[rowChosen][columnChosen].selected = 1;
                        int indexChosen = dataMat[rowChosen][columnChosen].index;
                        //  索引为负数 或 距离太近的点 不取
                        if (indexChosen != -1)
                        {
                            // if((cloud->points[indexChosen].x>=5 && abs(cloud->points[indexChosen].y)>=5)
                            //     || cloud->points[indexChosen].z < -lidarHeight+0.15 ){
                            // add samples for estimation
                            // 2表示输出的两个残差，3表示输入是三维的速度向量
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_SELF_VELOCITY, 2, 3>(
                                    new CURVE_FITTING_COST_SELF_VELOCITY(
                                        cloud->points[indexChosen].x,
                                        cloud->points[indexChosen].y,
                                        cloud->points[indexChosen].z,
                                        cloud->points[indexChosen].velocity)),
                                // new ceres::CauchyLoss(1.0),
                                nullptr,
                                self_v);
                            // }
                        }
                        num++;
                    }
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    /**
     * 自身速度估计2
     * @param cloud 点云
     * @param dataMat 点处理状态记录矩阵
     * @param self_v 自身速度估计结果存储
     */
    void Estimate_Self_Velocity_2(pcl::PointCloud<FMCWPointType>::Ptr &cloud,
                                  vector<vector<ClusterData>> &dataMat,
                                  double self_v[])
    {
        // 将 60*10000 划分为 2*10 个区域，每个区域大小为30*1000，取其中70个点。一共取1400个点。
        // 使用柯西核函数降低误差
        int xx = row / row_velocity_es;
        int yy = column / column_velocity_es;

        ceres::Problem problem;
        for (int i = 0; i < row_velocity_es; i++)
        {
            int rowSt = i * xx;
            for (int j = 0; j < column_velocity_es; j++)
            {
                int columnSt = j * yy;

                for (int num = 0; num < 70;)
                {
                    srand(time(0) + rand());
                    int rowChosen = rowSt + rand() % xx;
                    int columnChosen = columnSt + rand() % yy;

                    if (dataMat[rowChosen][columnChosen].selected == 2)
                    {
                        continue;
                    }
                    else
                    {
                        dataMat[rowChosen][columnChosen].selected = 2;
                        int ind = dataMat[rowChosen][columnChosen].index;
                        //  索引为负数 或 距离太近的点 不取
                        if (ind != -1)
                        {
                            float dist = sqrt(cloud->points[ind].x * cloud->points[ind].x + cloud->points[ind].y * cloud->points[ind].y + cloud->points[ind].z * cloud->points[ind].z);
                            // 估计静止点的多普勒速度
                            float staticVel = -1.0 / dist * (cloud->points[ind].x * self_v[0] + cloud->points[ind].y * self_v[1] + cloud->points[ind].z * self_v[2]);
                            // 将符合静止条件的点作为种子点
                            if (abs(staticVel - cloud->points[ind].velocity) < 1.0)
                            {
                                // add samples for estimation
                                problem.AddResidualBlock(
                                    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_SELF_VELOCITY, 2, 3>(
                                        new CURVE_FITTING_COST_SELF_VELOCITY(
                                            cloud->points[ind].x,
                                            cloud->points[ind].y,
                                            cloud->points[ind].z,
                                            cloud->points[ind].velocity)),
                                    // new ceres::CauchyLoss(0.1),
                                    nullptr,
                                    self_v);
                            }
                        }
                        num++;
                    }
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    /**
     * 自身速度估计3
     * @param cloud 点云
     * @param self_v 自身速度估计结果存储
     */
    void Estimate_Self_Velocity_3(pcl::PointCloud<FMCWPointType>::Ptr &cloud,
                                  double self_v[])
    {
        // 一共取4000个点。
        int pick_num = 0.8 * cloud->points.size();
        if (pick_num > 4000)
        {
            pick_num = 4000;
        }

        int *arr = ShuffleArray_Fisher_Yates(cloud->points.size());
        int num = 0;

        ceres::Problem problem;
        for (int i = 0; i < int(cloud->points.size()); i++)
        {
            int ind = arr[i];
            float dist = sqrt(cloud->points[ind].x * cloud->points[ind].x + cloud->points[ind].y * cloud->points[ind].y + cloud->points[ind].z * cloud->points[ind].z);
            // 估计静止点的多普勒速度
            float staticVel = -1.0 / dist * (cloud->points[ind].x * self_v[0] + cloud->points[ind].y * self_v[1] + cloud->points[ind].z * self_v[2]);
            // 将符合静止条件的点作为种子点
            if (abs(staticVel - cloud->points[ind].velocity) < 0.3)
            {
                // add samples for estimation
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_SELF_VELOCITY, 2, 3>(
                        new CURVE_FITTING_COST_SELF_VELOCITY(
                            cloud->points[ind].x,
                            cloud->points[ind].y,
                            cloud->points[ind].z,
                            cloud->points[ind].velocity)),
                    new ceres::CauchyLoss(1.0),
                    // nullptr,
                    self_v);
                num++;
            }

            if (num == pick_num)
            {
                break;
            }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 4;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    /**
     * 自身速度估计4
     * @param cloud 点云
     * @param dataMat 点处理状态记录矩阵
     * @param self_v 自身速度估计结果存储
     */
    void Estimate_Self_Velocity_zwp(pcl::PointCloud<FMCWPointType>::Ptr &cloud,
                                    vector<vector<ClusterData>> &dataMat,
                                    double self_v[])
    {
        int yy = column;
        ceres::Problem problem;
        // 遍历行
        for (auto it = rowIndexRecord.begin(); it != rowIndexRecord.end(); ++it)
        {
            int rowChosen = *it;
            // cout << "速度估计行号：" << rowChosen << endl;
            // 每行pointsEveryRow个点
            for (int num = 0; num < pointsEveryRow;)
            {
                // cout << "速度估计第 "<< num<< "个点！" << endl;
                // 列号随机
                srand(time(0) + rand());
                int columnChosen = rand() % yy;

                if (dataMat[rowChosen][columnChosen].selected == 1)
                {
                    // 已经选择过了
                    continue;
                }
                else
                {
                    dataMat[rowChosen][columnChosen].selected = 1;
                    int ind = dataMat[rowChosen][columnChosen].index;
                    //  索引为负数 或 距离太近的点 不取
                    if (ind != -1)
                    {
                        float dist = sqrt(cloud->points[ind].x * cloud->points[ind].x + cloud->points[ind].y * cloud->points[ind].y + cloud->points[ind].z * cloud->points[ind].z);
                        // 用自身的速度估计静止点的多普勒速度
                        float staticVel = -1.0 / dist * (cloud->points[ind].x * self_v[0] + cloud->points[ind].y * self_v[1] + cloud->points[ind].z * self_v[2]);
                        // 将符合静止条件的点作为种子点
                        if (abs(staticVel - cloud->points[ind].velocity) < 1.0)
                        {
                            // add samples for estimation
                            problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_SELF_VELOCITY, 2, 3>(
                                    new CURVE_FITTING_COST_SELF_VELOCITY(
                                        cloud->points[ind].x,
                                        cloud->points[ind].y,
                                        cloud->points[ind].z,
                                        cloud->points[ind].velocity)),
                                new ceres::CauchyLoss(1.0),
                                // nullptr,
                                self_v);
                        }
                    }
                    num++;
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }

    /**
     * 检查自身速度估计，防止突变
    */
    void checkSelfV(){
        double delte_v = sqrt((last_self_v[0] - self_v[0]) * (last_self_v[0] - self_v[0]) + (last_self_v[1] - self_v[1]) * (last_self_v[1] - self_v[1]) + (last_self_v[2] - self_v[2]) * (last_self_v[2] - self_v[2]));
        if (delte_v > check_self_v_threshold)
        {
            self_v[0] = last_self_v[0];
            self_v[1] = last_self_v[1];
            self_v[2] = last_self_v[2];
        }
    }

    /**
     * 提取种子点
     * 
    */
    vector<int> extractSeeds(vector<vector<ClusterData>> &dataMat, double self_v[])
    {
        // 筛选出种子点
        // 从低到高，从左至右进行依次筛选
        // pcl::PointCloud<FMCWPointType>::Ptr cloud_seeds(new pcl::PointCloud<FMCWPointType>);
        vector<int> seeds;
        seeds.reserve(40000); // 分配40000个元素
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < column; j++)
            {
                if (dataMat[i][j].index == -1)
                {
                    continue;
                }
                int ind = dataMat[i][j].index;
                float dist = sqrt(cloud->points[ind].x * cloud->points[ind].x + cloud->points[ind].y * cloud->points[ind].y + cloud->points[ind].z * cloud->points[ind].z);
                // 估计静止点的多普勒速度
                float staticVel = -1.0 / dist * (cloud->points[ind].x * self_v[0] + cloud->points[ind].y * self_v[1] + cloud->points[ind].z * self_v[2]);

                // if(cloud->points[ind].velocity>0.2){
                //     cloud_seeds->points.push_back(cloud->points[ind]);
                // }
                // 将不符合静止条件的点作为种子点
                if (abs(staticVel - cloud->points[ind].velocity) > velocityThreshold)
                {
                    dataMat[i][j].seeded = 1;      // 设置标识为种子点
                    cloud->points[ind].time = 1.0; // 这里将时间用于统计运动点个数
                    seeds.push_back(ind);          // 点的索引放入数组seeds中
                    // cloud_seeds->points.push_back(cloud->points[ind]);
                }
            }
        }
        return seeds;
    }

    /**
     * 获取地面点高度
    */
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

        int indiceSize = indices.size();
        for (int i = 0; i < indiceSize; i++)
        {
            int angle = atan2(thisPoint.y - cloud_kdtree->points[indices[i]].y, thisPoint.x - cloud_kdtree->points[indices[i]].x) * 3 / 3.1415927;
            height_map[angle].push(cloud_kdtree->points[indices[i]].intensity);
            if (height_map[angle].size() > 8)
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

        return (mean_h / mean_num + 0.15);
    }

    /**
     * 聚类
     * @param seeds 运动种子点个数
     * @param dataMat 点处理状态记录矩阵
     */
    void motionClustering(vector<int> seeds, vector<vector<ClusterData>> &dataMat)
    {
        // 过滤一半点云数据,并降采样放入kdtree,用于获取点周围的最低点
        // 使用pcl的PassThrough过滤出z轴-5到-0.5的点云，结果存放在cloud_filter中
        pcl::PointCloud<FMCWPointType>::Ptr cloud_filter(new pcl::PointCloud<FMCWPointType>());
        pcl::PassThrough<FMCWPointType> passThrough;
        passThrough.setFilterFieldName("z");
        passThrough.setFilterLimits(-5.0, -0.5);
        passThrough.setInputCloud(cloud);
        passThrough.filter(*cloud_filter);

        // 再对cloud_filter进行一波下采样，采用了0.3*0.3*0.3的体素，结果存放在cloud_ds中
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
        if (cloudSize>0){
            kdtree->setInputCloud(cloud_kdtree);
        }

        // 根据种子点进行聚类,根据当前点周围的最低点估计地面高度,聚类地面以上的点云
        clusters.reserve(100);
        int seedSize = seeds.size();
        // 针对每一个种子点进行处理
        for (int i = 0; i < seedSize; i++)
        {
            int rowIdx = cloud->points[seeds[i]].ring;
            int columnIdx = cloud->points[seeds[i]].column;
            // 如果种子点未处理,则首先确定地面高度,再聚类周围点
            if (dataMat[rowIdx][columnIdx].processed == -1)
            {
                // 获取地面高度信息
                // range是水平距离
                float range = sqrt(cloud->points[seeds[i]].x * cloud->points[seeds[i]].x + cloud->points[seeds[i]].y * cloud->points[seeds[i]].y);
                float groundHeight = 0.0;
                if (range < 8 || cloudSize == 0)
                {
                    // 水平距离8米以内的地面为雷达高度减0.15
                    groundHeight = -lidarHeight + 0.15;
                }
                else
                {
                    // 8米以外计算地面高度
                    groundHeight = Get_Low_Points_Height(kdtree, cloud_kdtree, cloud->points[seeds[i]], range * 0.1);
                }
                // RV视角下直接聚类，聚类结果放入cloud_extract中
                pcl::PointCloud<FMCWPointType>::Ptr cloud_extract(new pcl::PointCloud<FMCWPointType>());
                Search_Points(cloud, dataMat, rowIdx, columnIdx, dataMat[rowIdx][columnIdx].index, groundHeight, cloud_extract);
                // 若聚类出来的点的个数小于一定值则不进行处理
                if (cloud_extract->points.size() < clusterPointsThreshold)
                {
                    continue;
                }

                // 统计运动点所占比例,比例太小则跳过
                float num_moving = 0;
                for (int k = 0; k < (int)cloud_extract->points.size(); k++)
                {
                    if (cloud_extract->points[k].time == 1.0)
                    {
                        num_moving++;
                    }
                }
                if (num_moving / float(cloud_extract->points.size()) < 0.2)
                {
                    continue;
                }

                // 聚类的结果放入clusters数组
                clusters.push_back(cloud_extract);
                // objects保存到cloud_moving,并记录每个object的起始索引和结束索引
                BoundPointType thisBound;
                thisBound.st_idx = cloud_moving->points.size();
                *cloud_moving += *cloud_extract;
                thisBound.ed_idx = cloud_moving->points.size();
                clusters_bound->push_back(thisBound);
            }
        }

        // 获取静止点云，即非运动点则为静止点
        cloud_static->points.reserve(100000);
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
     * 根据种子点聚类
     * @param cloud 点云
     * @param dataMat 数据标识矩阵
     * @param rowIdx 行id
     * @param columnIdx 列id
     * @param pointIdx 点的id
     * @param ground 地面高度
     * @param cloud_extract 存放聚类结果的指针
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

        if ((dataMat[rowIdx][columnIdx].seeded == -1 && (dist_sub < 0.2) && (vel_sub < 0.1)) ||
            (dataMat[rowIdx][columnIdx].seeded == 1 && (dist_sub < distThreshold) && (vel_sub < velThreshold)))
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
     * 计算物体框的大小
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
            clusters_bound->points[i].x = thisRect.center.x;
            clusters_bound->points[i].y = thisRect.center.y;
            clusters_bound->points[i].z = (maxHeight + minHeight) / 2.0;
            clusters_bound->points[i].width = thisRect.size.width;
            clusters_bound->points[i].length = thisRect.size.height;
            clusters_bound->points[i].height = maxHeight - minHeight;
            clusters_bound->points[i].yaw = thisRect.angle / 180.0 * 3.14159;
        }
    }

    /**
     * 发布相关数据
     * @param CloudInfoMsg 当前帧输入的原始msg
     * @param self_v 自身速度
     */
    void publishClouds(const v_losm::cloud_infoConstPtr &CloudInfoMsg, double self_v[])
    {

        // 发布静止点云
        sensor_msgs::PointCloud2 StaticObjCloudMsg;
        pcl::toROSMsg(*cloud_static, StaticObjCloudMsg);
        StaticObjCloudMsg.header = thisHeader;  
        pubStaticObjCloud.publish(StaticObjCloudMsg);
        // cout << "output static cloud size: " << cloud_static->points.size() << endl;

        // 发布动态点云
        sensor_msgs::PointCloud2 MovingObjCloudMsg;
        pcl::toROSMsg(*cloud_moving, MovingObjCloudMsg);
        MovingObjCloudMsg.header = thisHeader;
        pubMovingObjCloud.publish(MovingObjCloudMsg);
        // cout << "output moving cloud size: " << cloud_moving->points.size() << endl;

        // 绘制boundingbox
        lShapedFit();

        // recording time 这边总共大概0.01
        // double time4 = ros::Time::now().toSec();
        // cout << "\033[4;0H" <<"\033[K"<< "BoundingBox time cost: " << time4-time3 <<" s."<<endl;
        // cout << "\033[5;0H" <<"\033[K"<< "Total time cost: " << time4-time1 <<" s."<<endl;
        // cout << "\033[6;0H" <<"\033[K"<< "Vel: " << self_v[0] <<" "<< self_v[1] <<" "<< self_v[2] <<endl;
        // cout << "\033[7;0H" <<"\033[K"<< "GT : " <<endl;
        // cout <<"\033[K"<<"\033[8;0H" <<"\033[K";

        // 发布 boundingbox 数据
        jsk_recognition_msgs::BoundingBoxArray arr_box;
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

        // 记录 clusters_bound
        sensor_msgs::PointCloud2 MovingObjCloudBound;
        pcl::toROSMsg(*clusters_bound, MovingObjCloudBound);
        MovingObjCloudBound.header = thisHeader;

        // 发布 cloud_info 数据
        v_losm::cloud_info cloudInfo;
        cloudInfo = *CloudInfoMsg;
        cloudInfo.self_v[0] = self_v[0];
        cloudInfo.self_v[1] = self_v[1];
        cloudInfo.self_v[2] = self_v[2];
        cloudInfo.cloud_static = StaticObjCloudMsg;
        cloudInfo.cloud_moving = MovingObjCloudMsg;
        cloudInfo.cloud_bounding = MovingObjCloudBound;
        cloudInfo.cloud_org = CloudInfoMsg->cloud_deskewed;
        pubCloudInfo.publish(cloudInfo);
    }

    /**
     * 发布自身速度marker
     */
    void showVelocity(double self_v[])
    {
        visualization_msgs::MarkerArray marker;
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
        thisTextMarker.scale.z = 2.0;
        thisTextMarker.color.g = 1.0f;
        thisTextMarker.color.a = 1;
        thisTextMarker.text = "vel:" + floatToString(float(self_v[0])) + " " + floatToString(float(self_v[1])) + " " + floatToString(float(self_v[2]));
        marker.markers.push_back(thisTextMarker);
        // 显示箭头
        visualization_msgs::Marker thisArrowMarker;
        thisArrowMarker.header = thisHeader;
        thisArrowMarker.ns = "arrow";
        thisArrowMarker.lifetime = ros::Duration(0.11);
        thisArrowMarker.id = 0;
        thisArrowMarker.pose.orientation.w = 1.0;
        thisArrowMarker.pose.orientation.x = 0.0;
        thisArrowMarker.pose.orientation.y = 0.0;
        thisArrowMarker.pose.orientation.z = 0.0;
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
        p2.x = p1.x + self_v[0] * 0.5;
        p2.y = p1.y + self_v[1] * 0.5;
        p2.z = p1.z + self_v[2] * 0.5;
        thisArrowMarker.points.push_back(p1);
        thisArrowMarker.points.push_back(p2);
        marker.markers.push_back(thisArrowMarker);

        pubMarker.publish(marker);
    }
};

int main(int argc, char **argv)
{
    cout << "\033[2J"
         << "\033[?25h" << endl;
    ros::init(argc, argv, "FastClustering");

    FastClustering FC;

    ROS_INFO("\033[1;32m----> FastClustering Started.\033[0m");

    ros::spin();
    return 0;
}
