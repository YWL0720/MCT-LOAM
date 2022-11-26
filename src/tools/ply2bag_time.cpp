//
// Created by ywl on 22-11-25.
//
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>


void showProgress(float progress)
{
    static int time = 0;
    if (time == 0)
        std::cout << std::endl;
    time++;
    if (progress > 1)
        progress = 1;
    int pa = progress * 50;
    std::cout << "\33[1A";
    std::cout << "[" + std::string(pa, '=') + ">" + std::string(50 - pa, ' ') << "]  " << progress * 100 << "% " << std::endl;
    fflush(stdout);
}

struct PointXYZT
{
    PCL_ADD_POINT4D;
    float timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZT,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, timestamp, timestamp)
)

struct PointXYZTIME
{
    PCL_ADD_POINT4D;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTIME,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, time, time)
)



int main(int argc, char** argv)
{
    ros::init(argc, argv, "ply2bag");
    ros::NodeHandle nh;

    std::string plyPath = "/media/ywl/T7/dataset/00/frames/";
    std::string bagName = "test.bag";
    std::string topicName = "/velodyne_points";
    std::string timePath = "/media/ywl/T7/dataset/kitti/gt/00/times.txt";

    ros::Time timeBegin = ros::Time::now();
    rosbag::Bag bag;
    bag.open(bagName, rosbag::bagmode::Write);

    pcl::PLYReader plyReader;

    std::fstream timeReader;

    timeReader.open(timePath, std::ios::in);

    pcl::PointCloud<PointXYZT>::Ptr cloud(new pcl::PointCloud<PointXYZT>());
    pcl::PointCloud<PointXYZTIME>::Ptr cloudWithTime(new pcl::PointCloud<PointXYZTIME>());


    int cloudSize = 4541;

    for (int i=0; i<cloudSize; i++)
    {
        std::string plyName;
        if (i>=0 && i<10)
            plyName = "frame_000" + std::to_string(i) + ".ply";
        else if (i>=10 && i<100)
            plyName = "frame_00" + std::to_string(i) + ".ply";
        else if (i>=100 && i<1000)
            plyName = "frame_0" + std::to_string(i) + ".ply";
        else
            plyName = "frame_" + std::to_string(i) + ".ply";

        plyReader.read<PointXYZT>(plyPath + plyName, *cloud);

        for (auto point : cloud->points)
        {
            PointXYZTIME pointWithTime;
            pointWithTime.x = point.x;
            pointWithTime.y = point.y;
            pointWithTime.z = point.z;
            pointWithTime.time = point.timestamp;
            cloudWithTime->push_back(pointWithTime);
        }

        std::string cloudTime;
        timeReader >> cloudTime;

        double time = atof(cloudTime.c_str());
        time += timeBegin.toSec();

        sensor_msgs::PointCloud2 pointCloudMsg;
        pcl::toROSMsg<PointXYZTIME>(*cloudWithTime, pointCloudMsg);
        pointCloudMsg.header.stamp = ros::Time(time);
        pointCloudMsg.header.frame_id = "/velodyne";


        bag.write("velodyne_points", pointCloudMsg.header.stamp, pointCloudMsg);

        showProgress(i/float(cloudSize));
        cloudWithTime->clear();
    }
    bag.close();
    timeReader.close();
    ROS_INFO("Finish");
    ros::shutdown();
    return 0;

}