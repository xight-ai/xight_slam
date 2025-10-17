//**** 设计步骤 ****//
// 得到点云的平面点和边缘点
//*****************// 
#define PCL_NO_PRECOMPILE

#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/impl/search.hpp>

#include "v_losm/cloud_info.h"
#include "v_losm/utility.h"

using namespace std;

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

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

typedef pcl::PointXYZI PointType;

class FeatureExtract : public ParamServer
{
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    // 用于发布角点
    ros::Publisher pubCornerCloud;
    // 用于发布平面点
    ros::Publisher pubSurfaceCloud;
    // 发布整个点云信息
    ros::Publisher pubCloudInfo;
    // 当前帧的header
    std_msgs::Header thisHeader;

    // 当前输入的点云
    pcl::PointCloud<FMCWPointType>::Ptr cloud;
    // 提取出的角点
    pcl::PointCloud<FMCWPointType>::Ptr cloud_corner;
    // 提取出的平面点
    pcl::PointCloud<FMCWPointType>::Ptr cloud_surface;

    // 平面和边缘的阈值
    float edgeThreshold = 1.5;
    float surfThreshold = 0.1;

    // 记录行的id
    std::set<int> rowIndexRecord;

public:
    FeatureExtract(){       
        // 订阅点云信息
        sub = nh.subscribe<v_losm::cloud_info>("/cloudInfo_cluster", 100, &FeatureExtract::subCallback, this, ros::TransportHints().tcpNoDelay());
        // 发布边缘点
        pubCornerCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_corner", 1);
        // 发布面点
        pubSurfaceCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_surface", 1);
        // 发布点云信息
        pubCloudInfo = nh.advertise<v_losm::cloud_info>("/cloudInfo_feature", 1);

        allocateMemory();
    }

    /**
     * @brief 为动态指针和动态数组分配内存
     */
    void allocateMemory()
    {
        cloud.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_corner.reset(new pcl::PointCloud<FMCWPointType>());
        cloud_surface.reset(new pcl::PointCloud<FMCWPointType>());

        resetParameters();
    }

    /**
     * 重置参数
     */
    void resetParameters()
    {
        cloud->clear();
        cloud_corner->clear();
        cloud_surface->clear();

        rowIndexRecord.clear();
    }

    void subCallback(const v_losm::cloud_infoConstPtr &CloudInfoMsg)
    {
        thisHeader = CloudInfoMsg->header;
        // 将静态点云转化为cloud
        pcl::fromROSMsg(CloudInfoMsg->cloud_static, *cloud);

        // cout<< "input cloud size: " << cloud->points.size() << endl;

        // double time1 = ros::Time::now().toSec();

        // 计算每个点的深度
        // 深度矩阵
        vector<vector<float>> rangeMat(row, vector<float>(column));
        // 索引矩阵
        vector<vector<int>> indexMat(row, vector<int>(column));
        rowIndexRecord.clear();
        for (size_t i = 0; i < cloud->points.size(); i++)
        {
            // 记录行的hid
            rowIndexRecord.insert(cloud->points[i].ring);

            int rowIdx = cloud->points[i].ring;
            int columnIdx = cloud->points[i].column;

            float dist = sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y + cloud->points[i].z * cloud->points[i].z);
            rangeMat[rowIdx][columnIdx] = dist;
            indexMat[rowIdx][columnIdx] = i;
        }

        // 计算点的曲面度
        vector<smoothness_t> cloudSmoothness(row * column);
        vector<int> pointColumnIdx(row * column); // 点的列位置
        vector<int> startColumnIndex;
        vector<int> endColumnIndex;
        vector<float> pointRange(row * column);        // 点的深度
        vector<int> pointIdx(row * column);            // 点的索引
        vector<float> cloudCurvature(row * column);    // 曲面程度
        vector<int> cloudNeighborPicked(row * column); // 点邻居是否被picked
        vector<int> cloudLabel(row * column);          // 点标签

        int count = 0;
        // 每行
        for (auto it = rowIndexRecord.begin(); it != rowIndexRecord.end(); ++it)
        {
            bool flag = false;
            // 行id
            int rowIndex = *it;
            // 这一块的起始id
            startColumnIndex.push_back(count);
            // 每列
            for (int i = 0; i < column; i++)
            {
                // 距离小于0.1视为无效点
                if (rangeMat[rowIndex][i] >= 0.1)
                {
                    // 记录每个点的列id
                    pointColumnIdx[count] = i;
                    // 记录每个点的距离
                    pointRange[count] = rangeMat[rowIndex][i];
                    // 记录每个点在点云中的id
                    pointIdx[count] = indexMat[rowIndex][i];
                    count++;
                    flag = true;
                }
                // 每行分成4段
                if ((i + 1) % (column / 4) == 0 && flag)
                {
                    // 这一块的结束id
                    endColumnIndex.push_back(count);
                    flag = false;
                    if (i + 1 < column)
                    {
                        startColumnIndex.push_back(count + 1);
                    }
                }
            }

            if (startColumnIndex.size() > endColumnIndex.size())
            {
                startColumnIndex.pop_back();
            }
        }

        for (int i = 5; i < count - 5; i++)
        {
            float diffRange = pointRange[i - 5] + pointRange[i - 4] + pointRange[i - 3] + pointRange[i - 2] + pointRange[i - 1] - pointRange[i] * 10 + pointRange[i + 5] + pointRange[i + 4] + pointRange[i + 3] + pointRange[i + 2] + pointRange[i + 1];
            // 记录每个点的曲率
            cloudCurvature[i] = diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;
            // 标识每个点是否被选取
            cloudNeighborPicked[i] = 0;
            // 表示每个点的类别
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
        // cout << "count size: " << count << endl;

        // 标记非法点
        for (int i = 5; i < count - 6; i++)
        {
            // occluded points
            float depth1 = pointRange[i];
            float depth2 = pointRange[i + 1];
            int columnDiff = abs(int(pointColumnIdx[i + 1] - pointColumnIdx[i]));

            // 去除被遮挡的点
            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            // 点云平面与激光射线方向几乎平行的点不能被取，下面在判断是否平行，0.02没看懂是啥
            float diff1 = abs(float(pointRange[i - 1] - pointRange[i]));
            float diff2 = abs(float(pointRange[i + 1] - pointRange[i]));
            if (diff1 > 0.02 * pointRange[i] && diff2 > 0.02 * pointRange[i])
            {
                cloudNeighborPicked[i] = 1;
            }
        }

        pcl::VoxelGrid<FMCWPointType> downSizeFilter;
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        // 遍历每一分块
        for (int i = 0; i < startColumnIndex.size(); i++)
        {
            // cout << "*******************start: " << i << " startColumnIndex: " << startColumnIndex << " endColumnIndex: " << endColumnIndex << endl;
            pcl::PointCloud<FMCWPointType>::Ptr surfaceCloudScan(new pcl::PointCloud<FMCWPointType>());

            // 在分块内将曲率排序，升序
            std::sort(cloudSmoothness.begin() + startColumnIndex[i], cloudSmoothness.begin() + endColumnIndex[i], by_value());

            int largestPickedNum = 0;
            // 将点曲率由高到低处理
            for (int k = endColumnIndex[i]; k > startColumnIndex[i]; k--)
            {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                {
                    largestPickedNum++;
                    // 每个区域只取20个边缘点
                    if (largestPickedNum <= 20)
                    {
                        cloudLabel[ind] = 1;
                        if (pointColumnIdx[ind] > 5 && pointColumnIdx[ind] < (column - 5))
                        {
                            int columnDiff1 = abs(int(pointColumnIdx[ind + 1] - pointColumnIdx[ind]));
                            int columnDiff2 = abs(int(pointColumnIdx[ind] - pointColumnIdx[ind - 1]));
                            if (columnDiff1 < 10 && columnDiff2 < 10)
                            {
                                // 边缘点
                                cloud_corner->push_back(cloud->points[pointIdx[ind]]);
                            }
                        }
                    }
                    else
                    {
                        break;
                    }

                    // 将选取的标记设置为1，同时将周围点的标记也设置为1
                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    {
                        int columnDiff = std::abs(int(pointColumnIdx[ind + l] - pointColumnIdx[ind + l - 1]));
                        if (columnDiff > 5)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        int columnDiff = std::abs(int(pointColumnIdx[ind + l] - pointColumnIdx[ind + l + 1]));
                        if (columnDiff > 5)
                            break;
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = startColumnIndex[i]; k <= endColumnIndex[i]; k++)
            {
                int ind = cloudSmoothness[k].ind;
                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                {
                    // 这里已经判定为平面点了，后面只是做一些周围点的处理
                    cloudLabel[ind] = -1;
                    cloudNeighborPicked[ind] = 1;

                    for (int l = 1; l <= 5; l++)
                    {
                        int columnDiff = abs(pointColumnIdx[ind + l] - pointColumnIdx[ind + l - 1]);
                        if (columnDiff > 5)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        int columnDiff = abs(pointColumnIdx[ind + l] - pointColumnIdx[ind + l + 1]);
                        if (columnDiff > 5)
                        {
                            break;
                        }
                        if (ind + l < 0)
                        {
                            continue;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = startColumnIndex[i]; k <= endColumnIndex[i]; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfaceCloudScan->push_back(cloud->points[pointIdx[k]]);
                }
            }

            pcl::PointCloud<FMCWPointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<FMCWPointType>());
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            // 平面点
            *cloud_surface += *surfaceCloudScanDS;
        }

        // 发布相关数据
        publishClouds(CloudInfoMsg);

        // 重置参数
        resetParameters();
        // double time4 = ros::Time::now().toSec();
        // cout<< "Vel: " << cloudInfo.self_v[0] <<" "<< cloudInfo.self_v[1] <<" "<< cloudInfo.self_v[2] <<endl;
        // cout<< "Total time cost: " << time4-time1 <<" s."<<endl;
    }

    /**
     * 发布相关数据
     * @param CloudInfoMsg 当前帧输入的原始msg
     */
    void publishClouds(const v_losm::cloud_infoConstPtr &CloudInfoMsg)
    {
        // 发布边缘点
        sensor_msgs::PointCloud2 CornerCloudMsg;
        pcl::toROSMsg(*cloud_corner, CornerCloudMsg);
        CornerCloudMsg.header = thisHeader;
        pubCornerCloud.publish(CornerCloudMsg);
        // cout << "output cloud_corner size: " << cloud_corner->points.size() << endl;

        // 发布平面点
        sensor_msgs::PointCloud2 SurfaceCloudMsg;
        pcl::toROSMsg(*cloud_surface, SurfaceCloudMsg);
        SurfaceCloudMsg.header = thisHeader;
        pubSurfaceCloud.publish(SurfaceCloudMsg);
        // cout << "output cloud_surface size: " << cloud_surface->points.size() << endl;

        // 在 cloud_info 中添加边缘点和平面点然后发布
        v_losm::cloud_info cloudInfo;
        cloudInfo = *CloudInfoMsg;
        cloudInfo.cloud_corner = CornerCloudMsg;
        cloudInfo.cloud_surface = SurfaceCloudMsg;
        pubCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "featureExtract");

    FeatureExtract FE;
    
    ROS_INFO("\033[1;32m----> FeatureExtract Started.\033[0m");

    ros::spin();
    return 0;
}
