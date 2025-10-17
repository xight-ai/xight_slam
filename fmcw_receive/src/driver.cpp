#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include <ros/ros.h>
#include <fmcw_receive/XiLidarPacket.h>

using namespace std;

#define SERV_PORT 62511
#define PACKET_SIZE 1440
// 每帧点云的最大点数x4个数据 42600*4
int sock_fd = -1;
struct sockaddr_in sender_address;
socklen_t sender_address_len = sizeof(sender_address);
int packet_id_;
ros::Publisher packet_pub;

bool polling()
{ 
    fmcw_receive::XiLidarPacketPtr packet(new fmcw_receive::XiLidarPacket());


    while(ros::ok()){
        // 接收字节数据
        ssize_t nbytes = recvfrom(sock_fd, &packet->data[0], PACKET_SIZE, 0, (struct sockaddr *)&sender_address, (socklen_t *)&sender_address_len);
        
        int packet_id = int(packet->data[3])*256 + int(packet->data[2]);
        int points_num = int(packet->data[1])*256 + int(packet->data[0]);
        // if(nbytes<300){continue;}
        packet_pub.publish(*packet);
        // if(abs(packet_id_ - packet_id)>1)
        std::cout << nbytes << " " << packet_id << " "<<packet_id_ <<" "<<points_num << " " << std::endl;
        packet_id_ = packet_id;

        // if (nbytes < 0)
        // {
        //     if (errno != EWOULDBLOCK)
        //     {
        //         perror("recvfail");
        //         return 1;
        //     }
        // }else if ((size_t) nbytes == PACKET_SIZE)
        // {
        //     // read successful,
        //     // if packet is not from the lidar scanner we selected by IP,
        //     // continue otherwise we are done

        //     break; //done
	    // }
    }

    // packet_pub.publish(*packet);
    return true;
}

int main(int argc, char *argv[])
{
    /* sock_fd --- socket文件描述符 创建udp套接字*/
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd == -1)
    {
      perror("socket");
      return 1;
    }

    /* 将套接字和IP、端口绑定 */
    struct sockaddr_in addr_serv;
    
    memset(&addr_serv, 0, sizeof(struct sockaddr_in));  //每个字节都用0填充
    addr_serv.sin_family = AF_INET; //使用IPV4地址
    addr_serv.sin_port = htons(SERV_PORT); //端口
    /* INADDR_ANY表示不管是哪个网卡接收到数据，只要目的端口是SERV_PORT，就会被该应用程序接收到 */
    addr_serv.sin_addr.s_addr = INADDR_ANY;  //自动获取IP地址

    /* 绑定socket */
    if(bind(sock_fd, (struct sockaddr *)&addr_serv, sizeof(addr_serv)) < 0)
    {
        perror("bind error:");
        exit(1);
    }
    /*v change recv_buffer size */
    int rcv_size = 1500*1024;
    if(setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, (char *)&rcv_size, sizeof(rcv_size)))
    {
        perror("setsockopt error:");
        exit(1);
    }
    if(setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, (char *)&rcv_size, sizeof(rcv_size)))
    {
        perror("setsockopt error:");
        exit(1);
    }

    string devip_str_ = "192.168.1.100";
    struct in_addr devip_;
    if (!devip_str_.empty())
    {
        inet_aton(devip_str_.c_str(), &devip_);
    }

    ros::init(argc, argv, "driver");
    ros::NodeHandle nh;
    packet_pub = nh.advertise<fmcw_receive::XiLidarPacket>("/lidar_packet",1000);
    ROS_INFO("\033[1;32m----> Publish lidar_packet.\033[0m");
    while(ros::ok()) {
        // poll device until end of file
        bool running = polling();
        // ROS_INFO_THROTTLE(30, "polling data successfully");
        if (!running)
            cout << "polling something wrong!" << endl;
    }
    ros::spin();

    close(sock_fd);
    return 0;
}
