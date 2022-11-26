//
// Created by ywl on 22-11-25.
// 检查点云数据中的field成员
// rosrun mct_loam field_check
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

void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr & msg)
{
    for (auto field : msg->fields)
        ROS_INFO("field name = %s", field.name.c_str());
    ros::shutdown();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "field_check");
    ros::NodeHandle nh;
    std::string topicName;
    if (argc == 2)
        topicName = std::string(argv[1]);
    else
        topicName = "/velodyne_points";
    if (topicName.size() == 0)
    {
        ROS_ERROR("Given a wrong topic name ");
        ros::shutdown();
        return -1;
    }

    ROS_INFO("check the topic %s", topicName.c_str());
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(topicName, 10, pointCloudCallback);

    ros::spin();
}

