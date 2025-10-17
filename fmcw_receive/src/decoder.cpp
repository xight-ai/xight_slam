#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>

#include <math.h>
#include <bitset>
#include <fstream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <fmcw_receive/XiLidarPacket.h>

using namespace std;

#define SERV_PORT 62511
#define PACKET_SIZE 1440
#define FRAME_SIZE 170400 // 每帧点云的最大点数x4个数据 42600*4

std::ofstream dataFile;

struct FMCWPointType
{
    PCL_ADD_POINT4D
    float velocity;
    uint ring;
    uint column;
    float column_angle;
    double time;
    float dist;
    float dist_x_y;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(FMCWPointType,
                                  (float, x, x)(float, y, y)(float, z, z)(float, velocity, velocity)(uint, ring, ring)(uint, column, column)(double, time, time)(float, dist, dist)(float, dist_x_y, dist_x_y)(float, column_angle, column_angle))

ros::Publisher pubFMCWPointCloud;
ros::Publisher pubFMCWPointCloud_v;
pcl::PointCloud<FMCWPointType>::Ptr point_cloud(new pcl::PointCloud<FMCWPointType>);
pcl::PointCloud<FMCWPointType>::Ptr point_cloud_v(new pcl::PointCloud<FMCWPointType>);
#define PI 3.1415926
int packet_id = 0;
#define delta_rad 0.0105 // 切波长的幅度变化
//float WAVE_LENGTH[10] = {0, delta_rad, 2*delta_rad, 5*delta_rad, 6*delta_rad, 7*delta_rad, 8*delta_rad, 11*delta_rad, 12*delta_rad, 13*delta_rad};
int point_cnt = 0;          // 每帧的点数
int point_cnt_all = 0;      // 全部帧的点数相加
int point_non_zero_cnt = 0; // 全部帧的非零点点数相加
int point_non_zero_cnt_no_v = 0;
int filter_point_cnt_all = 0; // 全部帧的 做了滤除后剩余的点数 相加
/* 距离滤除算法的几个参数 */
#define DIST_NUM 7
float last_distance[DIST_NUM];
int dist_update_id = 0;
bool dist_flag = false;
int NUM = 0;
int cnt_60 = 0;
int cnt = 0;
bool ini = false;
clock_t startTime, endTime;
int check_pkt_id = 0;
int up_error = 0;
int down_error = 0;
int landa[43] = {0};
int landa_[43] = {0};
float max_freq = 0.0;
double timet=0.0;
double timet_=0.0;
int point_num_=0;
// int point_count = 0;
// double last_time = -1;
// double cur_time = -1;
void packetCallback(const fmcw_receive::XiLidarPacketConstPtr &packet)
{
    //用于dsp
    /***
    auto num_ = int(packet->data[0])*256 + int(packet->data[1]);
    auto up_down = int(packet->data[2]);
    auto raw_cal = int(packet->data[3]);
    dataFile << num_ << " " << up_down << " " << raw_cal ;
    for(int i = 0; i < 121; i++){
        auto data_ = int(packet->data[4*i+8+3]) +int(packet->data[4*i+8+2])*256+int(packet->data[4*i+8+1])*256*256+int(packet->data[4*i+8+0])*256*256*256;
        dataFile << " " << data_;
    }
    dataFile<<endl;
    return;
    ***/


    cnt = cnt + 1;
    // 使用小端存储，低地址为low，高地址为high
    // 每个包的第2.3为包数，第0.1为点数，第4.5为新角度,第8.9为旧角度，第6.7.10.11为空，
    // 第12-15是保留位，数据从第48开始传输数据，第48.49是频率，第50是波长，第51是上/下chir
    // cout<<char(packet->data[0])<<" "<<char(packet->data[1])<<endl;
    //auto time_ = int(packet->data[14])*100+int(packet->data[13])*10+int(packet->data[12])+int(packet->data[15])*1000+int(packet->data[16])*10000+int(packet->data[17])*100000+int(packet->data[18])*1000000+int(packet->data[19])*10000000+int(packet->data[20])*100000000;
    double time_ = packet->header.stamp.toSec();
// cout<< "***********************time: " <<fixed<< time_<<endl;

    if (ini == false)
    {
        startTime = clock();
        ini = true;
        timet_ =time_;
        point_num_ = int(packet->data[1]) * 256 + int(packet->data[0]);
        return;
    }
    // cout<<time_<<endl;

    float angle_old = float(int(packet->data[10]) * 65536 + int(packet->data[9]) * 256 + int(packet->data[8])) * 360.0 / 524288.0;
    float angle_new = float(int(packet->data[6]) * 65536 + int(packet->data[5]) * 256 + int(packet->data[4])) * 360.0 / 524288.0;
    
    // if(angle_old <0 || angle_old >20){
    // return;
    // }
    // if(angle_old>360.1 || angle_new>360.1) {return;}
    // float TimeStamp = float(int(packet->data[15])*16777216+int(packet->data[14])*65536+int(packet->data[13])*256+int(packet->data[12]) + pow(16,8)) / pow(10,6);
    // if(int(packet->data[48+3])/64 != 0){return;}
    packet_id = int(packet->data[3]) * 256 + int(packet->data[2]);
    // cout << packet_id << "  " << check_pkt_id << endl;
    if (packet_id - check_pkt_id != 1)
        // cout << "////////////////////////////////////////////////" << packet_id << "  " << check_pkt_id << endl;
    check_pkt_id = packet_id;

    // 点数
    int points_num = int(packet->data[1]) * 256 + int(packet->data[0]);
    if(abs(points_num - point_num_)<1)
    {time_ = timet_;
    point_num_ = points_num;
    }
    else
    {timet_ = time_;
    point_num_ = points_num;
    }
    // cout<<points_num<<endl;
    points_num = points_num / 2;
    // cout<<points_num<<endl;
    // if(points_num == 257){return;}
    // {
    // 	std::ofstream dataFile;
    // 	dataFile.open("packet.txt", std::ofstream::app);
    // 	std::fstream file("packet.txt", std::ios::app);

    // 	dataFile << packet_id << " " << points_num << " " << angle_old << " " << angle_new << " " << std::endl;
    // 	dataFile.close();
    // }
    // if(angle_new-angle_old<-0.1)
    // std::cout << packet_id << " " << points_num *2<< " " << angle_old << " " << angle_new << " "<<angle_new-angle_old << std::endl;
    // std::cout << std::fixed << packet_id << " " << points_num << " " << time_ << " " << std::endl;
    // std::cout << packet_id << " " << points_num << std::endl;
    if (angle_new - angle_old < -0.1)
        angle_new = angle_new + 360.0;

    // 列角度变换
    float column_delta_angle = (angle_new - angle_old) / points_num * 2.0;
    
    int count = 0;

    for (int n = 48; n < 48 + points_num * 8; n += 8)
    {

        float up_freq, down_freq;
        FMCWPointType point;
        // 确定上下chirp
        // if(int(packet->data[n+3])/64 == 1){continue;}
        // if(int(packet->data[n+3])/64 == 3){continue;}
        // if(int(packet->data[n+3])/64 == 2){continue;}
        // if(int(packet->data[n+3])/64 == 0){continue;}
        point_cnt++;

        int channel = int(packet->data[n + 3]) / 64;
        if (int(packet->data[n + 3]) % 2 == 0 && int(packet->data[n + 7]) % 2 == 1)
        {
            //原来的
            // up_freq = (int(packet->data[n + 1]) * 256 + int(packet->data[n + 0])) / 2048.0 * 500;
            // down_freq = (int(packet->data[n + 5]) * 256 + int(packet->data[n + 4])) / 2048.0 * 500;

            // 改后的
            down_freq = (int(packet->data[n + 1]) * 256 + int(packet->data[n + 0])) / 2048.0 * 500;
            up_freq = (int(packet->data[n + 5]) * 256 + int(packet->data[n + 4])) / 2048.0 * 500;
            // if (up_freq > max_freq)
            // {
            //     max_freq = up_freq;
            // }
            // if (down_freq > max_freq)
            // {
            //     max_freq = down_freq;
            // }
        }
        else if (int(packet->data[n + 3]) % 2 == 1 && int(packet->data[n + 7]) % 2 == 0)
        {
            // 原来的
            // down_freq = (int(packet->data[n + 1]) * 256 + int(packet->data[n + 0])) / 2048.0 * 500;
            // up_freq = (int(packet->data[n + 5]) * 256 + int(packet->data[n + 4])) / 2048.0 * 500;

            // 改后的
            up_freq = (int(packet->data[n + 1]) * 256 + int(packet->data[n + 0])) / 2048.0 * 500;
            down_freq = (int(packet->data[n + 5]) * 256 + int(packet->data[n + 4])) / 2048.0 * 500;
            // if (up_freq > max_freq)
            // {
            //     max_freq = up_freq;
            // }
            // if (down_freq > max_freq)
            // {
            //     max_freq = down_freq;
            // }
        }

        // std::cout<<up_freq<<"  "<<down_freq<<std::endl;
        // dataFile << time_ << " " << down_freq << " " << up_freq << std::endl;

        // if(abs(up_freq - 51.0254) >1.0) {
        //     up_error++;
        // cout<<"up_freq : " <<up_freq<<endl;
        // }
        // if(abs(down_freq - 51.0254) >1.0) {
        //     down_error++;
        // cout<<"down_freq : "<<down_freq<<endl;
        // }
        // if(abs(up_freq -23 )<2.0 && abs(down_freq - 23)<2.0){
        // up_error++;
        // }
        // point_cnt++;
        // up_freq = (int(packet->data[n+1])*256+int(packet->data[n+0])) / 8192.0 * 1000 ;
        // down_freq = (int(packet->data[n+5])*256+int(packet->data[n+4])) / 8192.0 * 1000;
        // if(up_freq>20.0)
        // if(down_freq>75.0 || up_freq>75.0)
        // if(int(packet->data[n+3])/64 != int(packet->data[n+7])/64)
        // cout << int(packet->data[n+3])/64<<"  "<<int(packet->data[n+7])/64<<"  "<<up_freq <<"   "<<down_freq<<"   "<<endl;
        // if(up_freq > 190.0) cnt_60++;
        // if(down_freq > 190.0) cnt_60++;
        // if(int(up_freq) == 0 || int(down_freq) == 0){
        // cnt_60 ++;
        // continue;
        // up_freq = 0;
        // down_freq = 0;
        // }
        float column_angle;
        int c_t ;
        if (n < 48 + points_num * 8 / 2)
            {c_t = (n-48)/8;
            column_angle = (angle_old + column_delta_angle * (n / 8 - 6)) / 180.0 * PI;}
        else
            {column_angle = (angle_old + column_delta_angle * (n / 8 - 6 - points_num / 2)) / 180.0 * PI;
            c_t = (n-48)/8 - points_num / 2;}

        // if(n < 48 + points_num * 8/4) column_angle = (angle_old + column_delta_angle * (n/8-6)) / 180.0 * PI;
        // else if(n >= 48 + points_num * 8/4 && n < 48 + points_num * 8/4*2) column_angle = (angle_old + column_delta_angle * (n/8-6 - points_num/4)) / 180.0 * PI;
        // else if(n >= 48 + points_num * 8/4*2 && n < 48 + points_num * 8/4*3) column_angle = (angle_old + column_delta_angle * (n/8-6 - points_num/4*2)) / 180.0 * PI;
        // else column_angle = (angle_old + column_delta_angle * (n/8-6 - points_num/4*3)) / 180.0 * PI;
        // cout<<int(packet->data[n+3])/64<<"  "<<column_angle<<endl;
        int row;
        int num_ = int(packet->data[n + 2]);
        // cout<<channel<<"  "<<num_<<endl;
        
        // 小白子
        if (channel == 3)
        {
            row = num_;
        }
        else if (channel == 2)
        {
            row = num_ + 8;
        }
        else if (channel == 1)
        {
            row = num_ + 21;
        }
        else if (channel == 0)
        {
            row = num_ + 33;
        }
        landa[row] += 1;

        // 小黑子
        // if (channel == 3)
        // {
        //     row = num_ + 34;
        // }
        // else if (channel == 2)
        // {
        //     row = num_ + 9;
        // }
        // else if (channel == 1)
        // {
        //     row = num_ + 21;
        // }
        // else if (channel == 0)
        // {
        //     row = num_;
        // }

        // cout<<channel<<"  " <<row<<endl;
        // cout<<row<<endl;
        // if(channel == 0){
        //     row = 5;
        // }else if(channel == 1){
        //     row = 15;
        // }else if(channel == 2){
        //     row = 0;
        // }else if(channel == 3){
        //     row = 25;
        // }
        //float row_angle = WAVE_LENGTH[row];
        float row_angle = 0.0025 + (float)(-row + 25) * delta_rad;
        float wavelength = 1550.0; //0.0 + (float)row * delta_wl;

        float distance;

        // 添加过滤
        if(up_freq>120){
            up_freq = 0;
        }
        if(down_freq>120){
            down_freq = 0;
        }
        
        distance = (up_freq + down_freq - 2.6) / 1.33333 / 2.0 / 2.0;

        // cout<<distance<<endl;
        if (up_freq == 0 || down_freq == 0)
            //     // distance = (up_freq + down_freq) / 1.33333 / 2.0; // 这个参数记得改？？
            distance = 0;
        // else
        //     distance = (up_freq + down_freq - 4.0) / 1.33333 / 2.0/2.0; // 这个参数记得改？？
        // std::cout << up_freq << " " << down_freq << " " << distance <<std::endl;
        // float distance = (up_freq + down_freq) / 1.33333 / 2.0; // 这个参数记得改？？
        // if(distance >=50.0){
        //  cout << up_freq <<"   "<<down_freq<<"   "<<distance<<endl;
        // }
        // if(distance < 2) distance = 0;
        point.velocity = (down_freq - up_freq) / 4000.0 * wavelength;
        point.x = distance * cos(row_angle) * cos(-column_angle);
        point.y = distance * cos(row_angle) * sin(-column_angle);
        point.z = distance * sin(row_angle);
        point.ring = row;
        // cout<<fixed<<time_<<endl;
        // cout<<fixed << tt<<endl;
        point.time = time_ + float(c_t)*0.00001;
        // point.time = time_;
        // cout<<fixed<<point.time<<endl;
        // cout<<point.time<<endl;
        point.column = int(column_angle *180 / (column_delta_angle * PI));
        point.column_angle = column_angle;
        point.dist = distance;
        point.dist_x_y = distance * cos(row_angle);
        //if(column_delta_angle > 0.1 || column_delta_angle < 0.001) {std::cout << row <<std::endl;}

        // 距离滤除算法
        // /*
        if (DIST_NUM == 0 && distance != 0)
        {
            point_non_zero_cnt++;
            // if(fabs(point.velocity)<10 ){
            // point_non_zero_cnt_no_v++;
            // }
            // if(distance>0.1){
            // std::cout << packet_id << " " << points_num << " " << up_freq << " " << down_freq << " "<<std::endl;
            // if(fabs(point.velocity)<0.2){
            // if(last_time<0){
            //     last_time = point.time;
            // }
            point_cloud->points.push_back(point);
            ++point_cloud->width;
            // }}
        }
        else if (distance != 0)
        {
            int equal_distance_num = 0;
            for (int j = 0; j < DIST_NUM; ++j)
            {
                if (fabs(distance - last_distance[j]) < 0.2)
                {
                    equal_distance_num++;
                    if (DIST_NUM == 1)
                        equal_distance_num++;
                }
                if (equal_distance_num == 2)
                {
                    dist_flag = true;
                    break;
                }
            }
            if (dist_flag)
            {
                // std::cout << packet_id << " " << points_num << " " << up_freq << " " << down_freq << " "<<std::endl;
                // if(fabs(point.velocity)>0.2 && fabs(point.velocity)<10){
                // if (fabs(point.velocity) > 0.2)
                // {
                //     // if(fabs(point.velocity)>1.0){
                //     //     dataFile<<up_freq<<" "<<down_freq<<endl;
                //     // }

                //     point_cloud_v->points.push_back(point);
                //     ++point_cloud_v->width;
                // }
                // else if (fabs(point.velocity) <= 0.2)
                // {
                //     point_cloud->points.push_back(point);
                //     ++point_cloud->width;
                //     landa_[row] += 1;
                // }
                // if (last_time < 0)
                // {
                //     last_time = point.time;
                // }
                if (abs(point.velocity)<8.0 && point.dist<40.0){
                    point_cloud->points.push_back(point);
                    ++point_cloud->width;
                }
            }
            last_distance[dist_update_id] = distance;
            dist_update_id++;
            if (dist_update_id == DIST_NUM)
                dist_update_id = 0;
            dist_flag = false;
        }
        // */
    }
    // cout<<count<<endl;
    // if(point_cnt>25000000){
    if (angle_new > 360.0)
    { // 当一圈转完，发布该帧
        NUM = NUM + 1;
    }
    if (angle_new > 360.0 && NUM == 2)
    {
        NUM = 0;
        //if(1){
        // std::cout << packet_id << " " << points_num << " " << angle_old << " " << angle_new << " "<<std::endl;
        point_cloud->header.frame_id = "fmcw_lidar";
        // point_cloud->header.stamp = ros::Time::now();
        // // cout<<point_cloud->header.stamp<<endl;
        // cout<< point_cloud->header.stamp<<endl;
        point_cloud->height = 1;
        point_cloud_v->header.frame_id = "fmcw_lidar";
        point_cloud_v->height = 1;
        sensor_msgs::PointCloud2 PointCloudMsg;
        sensor_msgs::PointCloud2 PointCloudMsg_v;
        pcl::toROSMsg(*point_cloud, PointCloudMsg);
        pcl::toROSMsg(*point_cloud_v, PointCloudMsg_v);
        PointCloudMsg.header.stamp =packet->header.stamp;

        // point_count += point_cloud->points.size();
        // cur_time = point_cloud->points.back().time;
        // cout<< "目前平均点数：" << point_count/(cur_time-last_time)<< endl;

        pubFMCWPointCloud.publish(PointCloudMsg);
        {
            point_cnt_all += point_cnt;
            // for(int i =0;i<43;i++){
            // cout<<i<<"  "<<float(landa_[i])/float(landa[i])<<endl;
            // landa_[i]=0;
            // landa[i]=0;
            // }
            // filter_point_cnt_all+=point_cloud->points.size();
            //  std::cout<<point_cloud->points.size()<<std::endl;
            //  std::cout<<point_cnt*2<<std::endl;
            //  std::cout<<cnt_60/(float)point_cnt<<std::endl;
            // std::cout << point_cloud->points.size() / (float)point_cnt << std::endl;
            //cout << max_freq << endl;
            // cout<< "************************************************************************************"<<endl;
            // std::cout<<(point_cloud->points.size() +point_cloud_v->points.size())/(float)point_cnt<<std::endl;
            // std::cout<<point_cloud->points.size() +point_cloud_v->points.size()<<std::endl;

            // std::cout<<point_cnt<<std::endl;
            // std::cout<<cnt<<std::endl;

            //  std::cout<< "non_zero = " <<point_non_zero_cnt/(float)point_cnt<<std::endl;
            //  std::cout<< "non_zero_no_v = " <<point_non_zero_cnt_no_v/(float)point_cnt<<std::endl;
            // std::cout<<"all = "<< (float)filter_point_cnt_all/point_cnt_all<<std::endl;
            // std::cout<<"point_v = "<< (float)point_cloud_v->points.size() <<endl;
            // std::cout<<"point_v / all = "<< (float)point_cloud_v->points.size()/(point_cloud_v->points.size()+point_cloud->points.size())<<std::endl;
            // std::cout<<up_error/(float)point_cnt<<std::endl;
            // std::cout<<down_error/(float)point_cnt<<std::endl;
            max_freq = 0.0;
            cnt = 0;
            point_cnt = 0;
            cnt_60 = 0;
            up_error = 0;
            down_error = 0;
            point_non_zero_cnt = 0;
            point_non_zero_cnt_no_v = 0;
        }
        // pubFMCWPointCloud_v.publish(PointCloudMsg_v);
        point_cloud_v->clear();
        point_cloud_v.reset(new pcl::PointCloud<FMCWPointType>);
        point_cloud->clear();
        point_cloud.reset(new pcl::PointCloud<FMCWPointType>);
    }
    // if(cnt == 10000){
    //     endTime = clock();

    //     cout<<"Totle Time : "<<(double)(endTime - startTime)/10000.0<<"ms"<<endl;
    //     ini = false;
    //     cnt = 0;
    // }
    return;
}

int main(int argc, char *argv[])
{   
    // std::ofstream dataFile;
    dataFile.open("packet_2.txt", std::ofstream::app);
    	// std::fstream file("packet.txt", std::ios::app);
    ros::init(argc, argv, "usb_decoder");
    ros::NodeHandle nh;
    // 订阅雷达包
    ros::Subscriber packet_sub = nh.subscribe<fmcw_receive::XiLidarPacket>("/lidar_packet", 10000, &packetCallback);
    pubFMCWPointCloud = nh.advertise<sensor_msgs::PointCloud2>("/fmcw_pointcloud", 1);
    // pubFMCWPointCloud_v = nh.advertise<sensor_msgs::PointCloud2>("/fmcw_pointcloud_v", 1);
    ROS_INFO("\033[1;32m----> Publish PointCloud.\033[0m");
    ros::spin();
    return 0;
}
