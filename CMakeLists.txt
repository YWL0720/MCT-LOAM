cmake_minimum_required(VERSION 3.0.2)
project(mct_loam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

find_package(catkin REQUIRED COMPONENTS
  eigen_conversions
  geometry_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

include(FetchContent)

FetchContent_Declare(
        tessil 
	SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdLibrary/tessil-src)

if (NOT tessil_POPULATED)
    set(BUILD_TESTING OFF)
    FetchContent_Populate(tessil)

    add_library(robin_map INTERFACE)
    add_library(tsl::robin_map ALIAS robin_map)

    target_include_directories(robin_map INTERFACE
            "$<BUILD_INTERFACE:${tessil_SOURCE_DIR}/include>")

    list(APPEND headers "${tessil_SOURCE_DIR}/include/tsl/robin_growth_policy.h"
            "${tessil_SOURCE_DIR}/include/tsl/robin_hash.h"
            "${tessil_SOURCE_DIR}/include/tsl/robin_map.h"
            "${tessil_SOURCE_DIR}/include/tsl/robin_set.h")
    target_sources(robin_map INTERFACE "$<BUILD_INTERFACE:${headers}>")

    if (MSVC)
        target_sources(robin_map INTERFACE
                "$<BUILD_INTERFACE:${tessil_SOURCE_DIR}/tsl-robin-map.natvis>")
    endif ()
endif ()

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  robin_map
  include
)

add_executable(lio_optimization 
		src/lioOptimization.cpp
		src/optimize.cpp
		src/cloudProcessing.cpp 
		src/imuProcessing.cpp 
		src/initialization.cpp 
		src/utility.cpp
		src/parameters.cpp
		src/poseParameterization.cpp 
		src/lidarFactor.cpp
		src/imuFactor.cpp)
target_link_libraries(lio_optimization ${catkin_LIBRARIES} ${CERES_LIBRARIES} robin_map)

# tools
add_executable(trajectory_listener src/tools/mct_listener.cpp)
add_executable(field_check src/tools/field_check.cpp)
add_executable(ply2bag_timestamp src/tools/ply2bag_timestamp.cpp)
add_executable(ply2bag_time src/tools/ply2bag_time.cpp)

target_link_libraries(trajectory_listener ${catkin_LIBRARIES})
target_link_libraries(field_check ${catkin_LIBRARIES})
target_link_libraries(ply2bag_timestamp ${catkin_LIBRARIES})
target_link_libraries(ply2bag_time ${catkin_LIBRARIES})