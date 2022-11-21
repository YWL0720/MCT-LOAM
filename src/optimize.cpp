// c++
#include <iostream>
#include <math.h>
#include <vector>

// eigen 
#include <Eigen/Core>

#include "cloudMap.h"

// utility
#include "utility.h"
#include "parameters.h"

// optimize factor
#include "imuFactor.h"
#include "lidarFactor.h"
#include "poseParameterization.h"
#include "lioOptimization.h"

optimizeSummary lioOptimization::optimizeByAnalyticLidar(const icpOptions &cur_icp_options, const voxelHashMap &voxel_map_temp, std::vector<point3D> &keypoints, cloudFrame *p_frame)
{
    // 参数加载
    // nb_voxels_visited 搜索匹配点时 搜索的体素范围 初始化阶段为2 之后为1
    const short nb_voxels_visited = p_frame->frame_id < cur_icp_options.init_num_frames ? 2 : cur_icp_options.voxel_neighborhood;
    // 最小的Neighbors数量 默认为20
    const int kMinNumNeighbors = cur_icp_options.min_number_neighbors;
    // 默认为1
    const int kThresholdCapacity = p_frame->frame_id < cur_icp_options.init_num_frames ? 1 : cur_icp_options.threshold_voxel_occupancy;

    // 前一状态置为空 单位
    state *previous_state = nullptr;
    Eigen::Vector3d previous_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d previous_velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond previous_orientation = Eigen::Quaterniond::Identity();

    // 对于第0 1 帧以后的情况
    if (p_frame->frame_id > sweep_cut_num)
    {
        // 从历史帧状态中将前一状态直接赋值
        previous_state = all_cloud_frame[p_frame->id - sweep_cut_num]->p_state;
        previous_translation = previous_state->translation_end;
        // 前一状态的位移速度
        previous_velocity = previous_state->translation_end - previous_state->translation_middle;
        previous_orientation = Eigen::Quaterniond(previous_state->rotation_end);
    }
    // 当前状态 当前帧的值为每帧初始化阶段的预测值
    // 优化变量
    state *current_state = p_frame->p_state;
    Eigen::Quaterniond middle_quat = Eigen::Quaterniond(current_state->rotation_middle);
    Eigen::Quaterniond end_quat = Eigen::Quaterniond(current_state->rotation_end);
    Eigen::Vector3d middle_t = current_state->translation_middle;
    Eigen::Vector3d end_t = current_state->translation_end;


    // icp迭代次数 初始化阶段选择15和参数的最大值 其他阶段按照参数默认为5
    int num_iter_icp = p_frame->frame_id < cur_icp_options.init_num_frames ? std::max(15, cur_icp_options.num_iters_icp) :
                       cur_icp_options.num_iters_icp;

    // 根据时间插值计算关键点的世界坐标
    auto transformKeypoints = [&]()
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        for (auto &keypoint: keypoints)
        {
            double alpha_time = keypoint.alpha_time;
            if (alpha_time <= 0.5)
            {
                R = current_state->rotation_last_end.slerp(alpha_time * 2, middle_quat).normalized().toRotationMatrix();
                t = current_state->translation_last_end + alpha_time * 2 * (middle_t - current_state->translation_last_end);
            }
            else
            {
                R = middle_quat.slerp(alpha_time * 2 - 1, end_quat).normalized().toRotationMatrix();
                t = middle_t + (alpha_time * 2 - 1) * (end_t - middle_t);
            }
            keypoint.point = R * (R_imu_lidar * keypoint.raw_point + t_imu_lidar) + t;
        }
    };

    auto estimatePointNeighborhood = [&](std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &vector_neighbors,
                                           Eigen::Vector3d &location, double &planarity_weight)
    {
        // 计算平面法向量
        auto neighborhood = computeNeighborhoodDistribution(vector_neighbors);
        // 权重 a2D的平方
        planarity_weight = std::pow(neighborhood.a2D, cur_icp_options.power_planarity);


        if (neighborhood.normal.dot(p_frame->p_state->translation_middle - location) < 0)
        {
            neighborhood.normal = -1.0 * neighborhood.normal;
        }
        return neighborhood;
    };

    double lambda_weight = std::abs(cur_icp_options.weight_alpha);
    double lambda_neighborhood = std::abs(cur_icp_options.weight_neighborhood);
    const double kMaxPointToPlane = cur_icp_options.max_dist_to_plane_icp;
    const double sum = lambda_weight + lambda_neighborhood;

    lambda_weight /= sum;
    lambda_neighborhood /= sum;

    // 残差数量
    int number_of_residuals = 0;

    // 开始迭代计算
    for (int iter(0); iter < num_iter_icp; iter++)
    {
        // 将当前帧关键点按照最新的位姿还原到世界坐标系下 值保存在point中
        transformKeypoints();
        // 建立ceres问题
        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        // 损失函数
        switch (cur_icp_options.loss_function)
        {
            case LeastSquares::STANDARD:
                break;
            case LeastSquares::CAUCHY:
                loss_function = new ceres::CauchyLoss(cur_icp_options.ls_sigma);
                break;
            case LeastSquares::HUBER:
                loss_function = new ceres::HuberLoss(cur_icp_options.ls_sigma);
                break;
            case LeastSquares::TOLERANT:
                loss_function = new ceres::TolerantLoss(cur_icp_options.ls_tolerant_min_threshold, cur_icp_options.ls_sigma);
                break;
            case LeastSquares::TRUNCATED:
                loss_function = new TruncatedLoss(cur_icp_options.ls_sigma);
                break;
        }
        // 定义参数块
//        ceres::LocalParameterization *parameterization = new RotationParameterization();
        ceres::LocalParameterization *parameterization = new ceres::EigenQuaternionParameterization();

        problem.AddParameterBlock(&middle_quat.x(), 4, parameterization);
        problem.AddParameterBlock(&end_quat.x(), 4, parameterization);
        problem.AddParameterBlock(&middle_t.x(), 3);
        problem.AddParameterBlock(&end_t.x(), 3);

        int num_residuals = 0;

        int num_keypoints = keypoints.size();
        int num_threads = cur_icp_options.ls_num_threads;

        // 开始遍历关键点
        for (int k = 0; k < num_keypoints; ++k)
        {
            auto &keypoint = keypoints[k];
            // 关键点的原始坐标 在雷达坐标系下
            auto &raw_point = keypoint.raw_point;

            std::vector<voxel> voxels;
            // 根据体素地图查找最近点，返回最近点列表
            // 返回的是在相邻的1x1x1范围体素中的点 顺序按照距离从小到大排列 voxels保存了这些点分别属于哪些体素
            auto vector_neighbors = searchNeighbors(voxel_map_temp, keypoint.point,
                                                     nb_voxels_visited, cur_icp_options.size_voxel_map,
                                                     cur_icp_options.max_number_neighbors, kThresholdCapacity,
                                                     cur_icp_options.estimate_normal_from_neighborhood ? nullptr : &voxels);

            // 检查最近点数量是否满足要求 不能小于20
            if (vector_neighbors.size() < kMinNumNeighbors)
                continue;

            double weight;

            // 将该原始数据点位置还原到IMU坐标系下
            Eigen::Vector3d location = R_imu_lidar * raw_point + t_imu_lidar;

            // 计算最近点的法向量
            // 输入最近点列表 IMU坐标系下的原始位置 未定义的权重
            // neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1  weight = std::pow(neighborhood.a2D, cur_icp_options.power_planarity)
            auto neighborhood = estimatePointNeighborhood(vector_neighbors, location/*raw_point*/, weight);

            // TODO ?
            weight = lambda_weight * weight + lambda_neighborhood * std::exp(-(vector_neighbors[0] -
                     keypoint.point).norm() / (kMaxPointToPlane * kMinNumNeighbors));

            double point_to_plane_dist;

            std::set<voxel> neighbor_voxels;
            // 遍历最近点优先队列 默认取队列中的front最近点 计算点到平面距离 其实就一次
            for (int i(0); i < cur_icp_options.num_closest_neighbors; ++i)
            {
                // 点到平面的距离
                point_to_plane_dist = std::abs((keypoint.point - vector_neighbors[i]).transpose() * neighborhood.normal);

                // 距离小于阈值
                if (point_to_plane_dist < cur_icp_options.max_dist_to_plane_icp)
                {

                    num_residuals++;
                    // 平面法向量
                    Eigen::Vector3d norm_vector = neighborhood.normal;
                    norm_vector.normalize();
                    double norm_offset = - norm_vector.dot(vector_neighbors[i]);

                    // (1) 对每个点分别加两段约束 分开写
                    /***********************************************************************/
//                    ceres::CostFunction* cost_function_1 = MCTLidarPlaneNormFactorFirstAutoDiff::Create(keypoints[k].raw_point, current_state->rotation_last_end, current_state->translation_last_end,
//                                                                                                      norm_vector, norm_offset, keypoints[k].alpha_time, weight);
//                    problem.AddResidualBlock(cost_function_1, loss_function, &middle_t.x(), &middle_quat.x());
//
//                    ceres::CostFunction* cost_function_2 = MCTLidarPlaneNormFactorSecondAutoDiff::Create(keypoints[k].raw_point, current_state->rotation_last_end, current_state->translation_last_end,
//                                                                                                       norm_vector, norm_offset, keypoints[k].alpha_time, weight);
//                    problem.AddResidualBlock(cost_function_2, loss_function, &middle_t.x(), &middle_quat.x(), &end_t.x(), &end_quat.x());
                    /***********************************************************************/

                    // (2) 对每个点分别加两段约束 写在一起
                    /***********************************************************************/
                    ceres::CostFunction* cost_function = MCTLidarPlaneNormFactorAD::Create(keypoints[k].raw_point, current_state->rotation_last_end, current_state->translation_last_end,
                                                                                                       norm_vector, norm_offset, keypoints[k].alpha_time, weight);
                    problem.AddResidualBlock(cost_function, loss_function, &middle_t.x(), &middle_quat.x(), &end_t.x(), &end_quat.x());
                    /***********************************************************************/

                    // (3) TODO 对每个点添加三段样条插值约束 数值求导方法
                    /***********************************************************************/
//                    ceres::CostFunction* cost_function = MCTLidarPlaneNormFactorND::Create(keypoints[k].raw_point, current_state->rotation_last_end, current_state->translation_last_end,
//                                                                                                       norm_vector, norm_offset, keypoints[k].alpha_time, weight);
//                    problem.AddResidualBlock(cost_function, loss_function, &middle_t.x(), &middle_quat.x(), &end_t.x(), &end_quat.x());
                    /***********************************************************************/

                }
            }

            // 不会停止 越多越好
            if(num_residuals >= cur_icp_options.max_num_residuals) break;
        }


        // 当前帧添加的残差数量小于20 约束不足
        // 也就是能不能找到 可以匹配到满足要求最近点的 20个当前帧关键点
        if (num_residuals < cur_icp_options.min_number_neighbors)
        {
            std::stringstream ss_out;
            ss_out << "[Optimization] Error : not enough keypoints selected in ct-icp !" << std::endl;
            ss_out << "[Optimization] number_of_residuals : " << num_residuals << std::endl;
            optimizeSummary summary;
            summary.success = false;
            summary.num_residuals_used = num_residuals;
            summary.error_log = ss_out.str();
            if (cur_icp_options.debug_print) {
                std::cout << summary.error_log;
            }
            return summary;
        }

        // 求解
        ceres::Solver::Options ceres_options;
        ceres_options.max_num_iterations = cur_icp_options.ls_max_num_iters;
        ceres_options.num_threads = cur_icp_options.ls_num_threads;
        ceres_options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        ceres_options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary_ceres;
        ceres::Solve(ceres_options, &problem, &summary_ceres);

        if (!summary_ceres.IsSolutionUsable())
        {
            std::cout << summary_ceres.FullReport() << std::endl;
            throw std::runtime_error("Error During Optimization");
        }
        if (cur_icp_options.debug_print) {
            std::cout << summary_ceres.BriefReport() << std::endl;
        }

        // 单位化
        middle_quat.normalize();
        end_quat.normalize();

        double diff_trans = 0, diff_rot = 0;

        // 计算当前优化结束后 四个状态的变化量 当变化量小于阈值时 也认为是ICP收敛
        diff_trans += (current_state->translation_middle - middle_t).norm();
        diff_rot += AngularDistance(current_state->rotation_middle, middle_quat);

        diff_trans += (current_state->translation_end - end_t).norm();
        diff_rot += AngularDistance(current_state->rotation_end, end_quat);

        // 将估计出的值赋给最新的当前帧状态量

        current_state->translation_middle = middle_t;
        current_state->translation_end = end_t;
        current_state->rotation_middle = middle_quat;
        current_state->rotation_end = end_quat;

        if ((p_frame->frame_id > sweep_cut_num) &&
            (diff_rot < cur_icp_options.threshold_orientation_norm/2 &&
             diff_trans < cur_icp_options.threshold_translation_norm/2))
        {

            if (cur_icp_options.debug_print)
            {
                std::cout << "Optimization: Finished with N=" << iter << " ICP iterations" << std::endl;

            }
            break;
        }
    }
    // 用最新的估计出来的状态更新当前帧的点云
    transformKeypoints();

    optimizeSummary summary;
    summary.success = true;
    summary.num_residuals_used = number_of_residuals;
    return summary;
}

optimizeSummary lioOptimization::optimizeByAnalyticLio(const icpOptions &cur_icp_options, const voxelHashMap &voxel_map_temp, std::vector<point3D> &keypoints, cloudFrame *p_frame)
{

    const short nb_voxels_visited = p_frame->frame_id < cur_icp_options.init_num_frames ? 2 : cur_icp_options.voxel_neighborhood;
    const int kMinNumNeighbors = cur_icp_options.min_number_neighbors;
    const int kThresholdCapacity = p_frame->frame_id < cur_icp_options.init_num_frames ? 1 : cur_icp_options.threshold_voxel_occupancy;

    state *previous_state = nullptr;
    Eigen::Vector3d previous_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d previous_velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond previous_orientation = Eigen::Quaterniond::Identity();

    if (p_frame->frame_id > sweep_cut_num) {
        previous_state = all_cloud_frame[p_frame->id - sweep_cut_num]->p_state;
        previous_translation = previous_state->translation_end;
        previous_velocity = previous_state->translation_end - previous_state->translation_middle;
        previous_orientation = Eigen::Quaterniond(previous_state->rotation_end);
    }

    state *current_state = p_frame->p_state;
    Eigen::Quaterniond begin_quat = Eigen::Quaterniond(current_state->rotation_middle);
    Eigen::Quaterniond end_quat = Eigen::Quaterniond(current_state->rotation_end);
    Eigen::Vector3d begin_t = current_state->translation_middle;
    Eigen::Vector3d end_t = current_state->translation_end;

    Eigen::Matrix<double, 9 ,1> begin_velocity_bias;
    begin_velocity_bias.segment<3>(0) = current_state->velocity_begin;
    begin_velocity_bias.segment<3>(3) = current_state->ba_begin;
    begin_velocity_bias.segment<3>(6) = current_state->bg_begin;
    Eigen::Matrix<double, 9 ,1> end_velocity_bias;
    end_velocity_bias.segment<3>(0) = current_state->velocity;
    end_velocity_bias.segment<3>(3) = current_state->ba;
    end_velocity_bias.segment<3>(6) = current_state->bg;

    std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> v_inter_quat;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> v_inter_trans;
    std::vector<Eigen::Matrix<double, 9, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 9, 1>>> v_inter_velocity_bias;
    std::vector<double> v_inter_time;

    v_inter_quat.reserve(sweep_cut_num - 1);
    v_inter_trans.reserve(sweep_cut_num - 1);
    v_inter_velocity_bias.reserve(sweep_cut_num - 1);
    v_inter_time.reserve(sweep_cut_num - 1);

    for (int i = 1; i < sweep_cut_num; i++)
    {
        v_inter_quat.push_back(all_cloud_frame[p_frame->id - i]->p_state->rotation_end);
        v_inter_trans.push_back(all_cloud_frame[p_frame->id - i]->p_state->translation_end);

        Eigen::Matrix<double, 9 ,1> inter_velocity_bias;
        inter_velocity_bias.segment<3>(0) = all_cloud_frame[p_frame->id - i]->p_state->velocity;
        inter_velocity_bias.segment<3>(3) = all_cloud_frame[p_frame->id - i]->p_state->ba;
        inter_velocity_bias.segment<3>(6) = all_cloud_frame[p_frame->id - i]->p_state->bg;

        v_inter_velocity_bias.push_back(inter_velocity_bias);
        v_inter_time.push_back(all_cloud_frame[p_frame->id - i]->time_sweep_end);
    }

    int num_iter_icp = p_frame->frame_id < cur_icp_options.init_num_frames ? std::max(15, cur_icp_options.num_iters_icp) :
                       cur_icp_options.num_iters_icp;

    auto transformKeypoints = [&]()
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        for (auto &keypoint: keypoints) {
            if (cur_icp_options.point_to_plane_with_distortion || cur_icp_options.distance == CT_POINT_TO_PLANE) {
                double alpha_time = keypoint.alpha_time;
                Eigen::Quaterniond q = begin_quat.slerp(alpha_time, end_quat);
                q.normalize();
                R = q.toRotationMatrix();
                t = (1.0 - alpha_time) * begin_t + alpha_time * end_t;
            } else {
                R = end_quat.normalized().toRotationMatrix();
                t = end_t;
            }

            keypoint.point = R * (R_imu_lidar * keypoint.raw_point + t_imu_lidar) + t;
        }
    };

    auto estimatePointNeighborhood = [&](std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &vector_neighbors,
                                           Eigen::Vector3d &location, double &planarity_weight)
    {

        auto neighborhood = computeNeighborhoodDistribution(vector_neighbors);
        planarity_weight = std::pow(neighborhood.a2D, cur_icp_options.power_planarity);

        if (neighborhood.normal.dot(p_frame->p_state->translation_middle - location) < 0) {
            neighborhood.normal = -1.0 * neighborhood.normal;
        }
        return neighborhood;
    };

    double lambda_weight = std::abs(cur_icp_options.weight_alpha);
    double lambda_neighborhood = std::abs(cur_icp_options.weight_neighborhood);
    const double kMaxPointToPlane = cur_icp_options.max_dist_to_plane_icp;
    const double sum = lambda_weight + lambda_neighborhood;

    lambda_weight /= sum;
    lambda_neighborhood /= sum;

    int number_of_residuals = 0;

    for (int iter(0); iter < num_iter_icp; iter++) {
        transformKeypoints();

        ceres::Problem problem;
        ceres::LossFunction *loss_function;

        switch (cur_icp_options.loss_function)
        {
            case LeastSquares::STANDARD:
                break;
            case LeastSquares::CAUCHY:
                loss_function = new ceres::CauchyLoss(cur_icp_options.ls_sigma);
                break;
            case LeastSquares::HUBER:
                loss_function = new ceres::HuberLoss(cur_icp_options.ls_sigma);
                break;
            case LeastSquares::TOLERANT:
                loss_function = new ceres::TolerantLoss(cur_icp_options.ls_tolerant_min_threshold, cur_icp_options.ls_sigma);
                break;
            case LeastSquares::TRUNCATED:
                loss_function = new TruncatedLoss(cur_icp_options.ls_sigma);
                break;
        }

        ceres::LocalParameterization *parameterization = new RotationParameterization();

        switch (cur_icp_options.distance) {
            case CT_POINT_TO_PLANE:
                problem.AddParameterBlock(&begin_quat.x(), 4, parameterization);
                problem.AddParameterBlock(&end_quat.x(), 4, parameterization);
                problem.AddParameterBlock(&begin_t.x(), 3);
                problem.AddParameterBlock(&end_t.x(), 3);
                problem.AddParameterBlock(&begin_velocity_bias[0], 9);
                problem.AddParameterBlock(&end_velocity_bias[0], 9);

                for (int i = 0; i < sweep_cut_num - 1; i++)
                {
                    problem.AddParameterBlock(&v_inter_quat[i].x(), 4, parameterization);
                    problem.AddParameterBlock(&v_inter_trans[i].x(), 3);
                    problem.AddParameterBlock(&v_inter_velocity_bias[i][0], 9);
                }

                break;
            case POINT_TO_PLANE:
                problem.AddParameterBlock(&end_quat.x(), 4, parameterization);
                problem.AddParameterBlock(&end_t.x(), 3);
                problem.AddParameterBlock(&end_velocity_bias[0], 9);
                break;
        }

        int num_residuals = 0;
        int num_keypoints = keypoints.size();
        int num_threads = cur_icp_options.ls_num_threads;

        for (int k = 0; k < num_keypoints; ++k) {
            auto &keypoint = keypoints[k];
            auto &raw_point = keypoint.raw_point;

            std::vector<voxel> voxels;
            auto vector_neighbors = searchNeighbors(voxel_map_temp, keypoint.point,
                                                     nb_voxels_visited, cur_icp_options.size_voxel_map,
                                                     cur_icp_options.max_number_neighbors, kThresholdCapacity,
                                                     cur_icp_options.estimate_normal_from_neighborhood ? nullptr : &voxels);

            if (vector_neighbors.size() < kMinNumNeighbors)
                continue;

            double weight;

            Eigen::Vector3d location = R_imu_lidar * raw_point + t_imu_lidar;

            auto neighborhood = estimatePointNeighborhood(vector_neighbors, location, weight);

            weight = lambda_weight * weight + lambda_neighborhood * std::exp(-(vector_neighbors[0] -
                     keypoint.point).norm() / (kMaxPointToPlane * kMinNumNeighbors));

            double point_to_plane_dist;
            std::set<voxel> neighbor_voxels;
            for (int i(0); i < cur_icp_options.num_closest_neighbors; ++i) {
                point_to_plane_dist = std::abs((keypoint.point - vector_neighbors[i]).transpose() * neighborhood.normal);

                if (point_to_plane_dist < cur_icp_options.max_dist_to_plane_icp) {

                    num_residuals++;

                    Eigen::Vector3d norm_vector = neighborhood.normal;
                    norm_vector.normalize();
                    double norm_offset = - norm_vector.dot(vector_neighbors[i]);

                    switch (cur_icp_options.distance) {
                        case CT_POINT_TO_PLANE:
                        {
                            CTLidarPlaneNormFactor *cost_function = new CTLidarPlaneNormFactor(keypoints[k].raw_point, norm_vector, norm_offset, keypoints[k].alpha_time, weight);
                            problem.AddResidualBlock(cost_function, loss_function, &begin_t.x(), &begin_quat.x(), &end_t.x(), &end_quat.x());

                            if (sweep_cut_num == 3)
                            {
                                if (keypoints[k].timestamp < p_frame->time_sweep_end && keypoints[k].timestamp > v_inter_time[0])
                                {
                                    double alpha = (keypoints[k].timestamp - v_inter_time[0]) / (p_frame->time_sweep_end - v_inter_time[0]);
                                    CTLidarPlaneNormFactor *cost_function = new CTLidarPlaneNormFactor(keypoints[k].raw_point, norm_vector, norm_offset, alpha, weight);
                                    problem.AddResidualBlock(cost_function, loss_function, &v_inter_trans[0].x(), &v_inter_quat[0].x(), &end_t.x(), &end_quat.x());
                                }
                                else if (keypoints[k].timestamp < v_inter_time[0] && keypoints[k].timestamp > v_inter_time[1])
                                {
                                    double alpha = (keypoints[k].timestamp - v_inter_time[1]) / (v_inter_time[0] - v_inter_time[1]);
                                    CTLidarPlaneNormFactor *cost_function = new CTLidarPlaneNormFactor(keypoints[k].raw_point, norm_vector, norm_offset, alpha, weight);
                                    problem.AddResidualBlock(cost_function, loss_function, &v_inter_trans[1].x(), &v_inter_quat[1].x(), &v_inter_trans[0].x(), &v_inter_quat[0].x());
                                }
                                else if (keypoints[k].timestamp < v_inter_time[1] && keypoints[k].timestamp > p_frame->time_sweep_begin)
                                {
                                    double alpha = (keypoints[k].timestamp - p_frame->time_sweep_begin) / (v_inter_time[1] - p_frame->time_sweep_begin);
                                    CTLidarPlaneNormFactor *cost_function = new CTLidarPlaneNormFactor(keypoints[k].raw_point, norm_vector, norm_offset, alpha, weight);
                                    problem.AddResidualBlock(cost_function, loss_function, &begin_t.x(), &begin_quat.x(), &v_inter_trans[1].x(), &v_inter_quat[1].x());
                                }
                            }
                            break;
                        }
                        case POINT_TO_PLANE:
                        {
                            Eigen::Vector3d point_end = end_quat.inverse() * keypoints[k].point - end_quat.inverse() * end_t;
                            LidarPlaneNormFactor *cost_function = new LidarPlaneNormFactor(point_end, norm_vector, norm_offset, weight);
                            problem.AddResidualBlock(cost_function, loss_function, &end_t.x(), &end_quat.x());
                            break;
                        }
                    }
                }
            }

            if(num_residuals >= cur_icp_options.max_num_residuals) break;
        }

        if (p_frame->frame_id > sweep_cut_num) {

            switch (cur_icp_options.distance) {
                case CT_POINT_TO_PLANE:
                {
                    LocationConsistencyFactor *cost_location_consistency = new LocationConsistencyFactor(previous_translation, sqrt(num_residuals * cur_icp_options.beta_location_consistency * laser_point_cov));
                    problem.AddResidualBlock(cost_location_consistency, nullptr, &begin_t.x());

                    RotationConsistencyFactor *cost_rotation_consistency = new RotationConsistencyFactor(previous_orientation, sqrt(num_residuals * cur_icp_options.beta_orientation_consistency * laser_point_cov));
                    problem.AddResidualBlock(cost_rotation_consistency, nullptr, &begin_quat.x());

                    SmallVelocityFactor *cost_small_velocity = new SmallVelocityFactor(sqrt(num_residuals * cur_icp_options.beta_small_velocity * laser_point_cov));
                    problem.AddResidualBlock(cost_small_velocity, nullptr, &begin_t.x(), &end_t.x());

                    if (p_frame->p_state->pre_integration->sum_dt < 10.0) {

                        if (sweep_cut_num > 1)
                        {
                            CTImuFactor* imu_factor_begin = new CTImuFactor(all_cloud_frame[p_frame->id - sweep_cut_num + 1]->p_state->pre_integration, 1);
                            problem.AddResidualBlock(imu_factor_begin, loss_function, &begin_t.x(), &begin_quat.x(), &begin_velocity_bias[0], 
                                                     &v_inter_trans.back().x(), &v_inter_quat.back().x(), &v_inter_velocity_bias.back()[0]);

                            for (int i = 1; i < sweep_cut_num - 1; i++)
                            {
                                CTImuFactor* imu_factor = new CTImuFactor(all_cloud_frame[p_frame->id - i]->p_state->pre_integration, 1);
                                problem.AddResidualBlock(imu_factor, loss_function, &v_inter_trans[i].x(), &v_inter_quat[i].x(), &v_inter_velocity_bias[i][0], 
                                                         &v_inter_trans[i - 1].x(), &v_inter_quat[i - 1].x(), &v_inter_velocity_bias[i - 1][0]);
                            }

                            CTImuFactor* imu_factor_end = new CTImuFactor(p_frame->p_state->pre_integration, 1);
                            problem.AddResidualBlock(imu_factor_end, loss_function, &v_inter_trans[0].x(), &v_inter_quat[0].x(), &v_inter_velocity_bias[0][0], 
                                                     &end_t.x(), &end_quat.x(), &end_velocity_bias[0]);
                        }
                        else
                        {
                            CTImuFactor* imu_factor = new CTImuFactor(all_cloud_frame[p_frame->id - sweep_cut_num + 1]->p_state->pre_integration, 1);
                            problem.AddResidualBlock(imu_factor, loss_function, &begin_t.x(), &begin_quat.x(), &begin_velocity_bias[0],  &end_t.x(), &end_quat.x(), &end_velocity_bias[0]);
                        }

                        VelocityConsistencyFactor *cost_velocity_consistency = new VelocityConsistencyFactor(all_cloud_frame[p_frame->id - sweep_cut_num]->p_state, sqrt(num_residuals * cur_icp_options.beta_constant_velocity * laser_point_cov));
                        problem.AddResidualBlock(cost_velocity_consistency, nullptr, &begin_velocity_bias[0]);
                    }
                    break;
                }
                case POINT_TO_PLANE:
                {
                    ImuFactor* imu_factor = new ImuFactor(p_frame->p_state->pre_integration, all_cloud_frame[p_frame->id - 1]->p_state);
                    problem.AddResidualBlock(imu_factor, nullptr, &end_t.x(), &end_quat.x(), &end_velocity_bias[0]);
                    break;
                }
            }
        }
        if (num_residuals < cur_icp_options.min_number_neighbors)
        {
            std::stringstream ss_out;
            ss_out << "[Optimization] Error : not enough keypoints selected in ct-icp !" << std::endl;
            ss_out << "[Optimization] number_of_residuals : " << num_residuals << std::endl;
            optimizeSummary summary;
            summary.success = false;
            summary.num_residuals_used = num_residuals;
            summary.error_log = ss_out.str();
            if (cur_icp_options.debug_print) {
                std::cout << summary.error_log;
            }
            return summary;
        }

        ceres::Solver::Options ceres_options;
        ceres_options.max_num_iterations = cur_icp_options.ls_max_num_iters;
        ceres_options.num_threads = cur_icp_options.ls_num_threads;
        ceres_options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
        ceres_options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary_ceres;
        ceres::Solve(ceres_options, &problem, &summary_ceres);

        if (!summary_ceres.IsSolutionUsable()) {
            std::cout << summary_ceres.FullReport() << std::endl;
            throw std::runtime_error("Error During Optimization");
        }
        if (cur_icp_options.debug_print) {
            std::cout << summary_ceres.BriefReport() << std::endl;
        }


        begin_quat.normalize();
        end_quat.normalize();

        double diff_trans = 0, diff_rot = 0, diff_velocity = 0;

        for (int i = 1; i < sweep_cut_num - 1; i++)
        {
            v_inter_quat[i].normalize();

            diff_trans += (all_cloud_frame[p_frame->id - i]->p_state->translation_end - v_inter_trans[i - 1]).norm();
            diff_rot += AngularDistance(all_cloud_frame[p_frame->id - i]->p_state->rotation_end, v_inter_quat[i - 1]);
            diff_velocity += (all_cloud_frame[p_frame->id - i]->p_state->velocity - v_inter_velocity_bias[i - 1].segment<3>(0)).norm();
        }

        diff_trans += (current_state->translation_middle - begin_t).norm();
        diff_rot += AngularDistance(current_state->rotation_middle, begin_quat);
        diff_velocity += (current_state->velocity_begin - begin_velocity_bias.segment<3>(0)).norm();

        diff_trans += (current_state->translation_end - end_t).norm();
        diff_rot += AngularDistance(current_state->rotation_end, end_quat);
        diff_velocity += (current_state->velocity - end_velocity_bias.segment<3>(0)).norm();

        switch (cur_icp_options.distance) {
            case CT_POINT_TO_PLANE:
                current_state->translation_middle = begin_t;
                current_state->rotation_middle = begin_quat;
                current_state->velocity_begin = begin_velocity_bias.segment<3>(0);
                current_state->ba_begin = begin_velocity_bias.segment<3>(3);
                current_state->bg_begin = begin_velocity_bias.segment<3>(6);

                current_state->translation_end = end_t;
                current_state->rotation_end = end_quat;
                current_state->velocity = end_velocity_bias.segment<3>(0);
                current_state->ba = end_velocity_bias.segment<3>(3);
                current_state->bg = end_velocity_bias.segment<3>(6);

                for (int i = 1; i < sweep_cut_num; i++)
                {
                    all_cloud_frame[p_frame->id - i]->p_state->translation_end = v_inter_trans[i - 1];
                    all_cloud_frame[p_frame->id - i]->p_state->rotation_end = v_inter_quat[i - 1];
                    all_cloud_frame[p_frame->id - i]->p_state->velocity = v_inter_velocity_bias[i - 1].segment<3>(0);
                    all_cloud_frame[p_frame->id - i]->p_state->ba = v_inter_velocity_bias[i - 1].segment<3>(3);
                    all_cloud_frame[p_frame->id - i]->p_state->bg = v_inter_velocity_bias[i - 1].segment<3>(6);
                }
                break;
            case POINT_TO_PLANE:
                current_state->translation_end = end_t;
                current_state->rotation_end = end_quat;
                current_state->velocity = end_velocity_bias.segment<3>(0);
                current_state->ba = end_velocity_bias.segment<3>(3);
                current_state->bg = end_velocity_bias.segment<3>(6);
                break;
        }

        if ((p_frame->frame_id > sweep_cut_num) &&
            (diff_rot < cur_icp_options.threshold_orientation_norm &&
             diff_trans < cur_icp_options.threshold_translation_norm && 
             diff_velocity < cur_icp_options.threshold_translation_norm)) {

            if (cur_icp_options.debug_print) {
                std::cout << "Optimization: Finished with N=" << iter << " ICP iterations" << std::endl;

            }
            break;
        }
    }
    transformKeypoints();

    optimizeSummary summary;
    summary.success = true;
    summary.num_residuals_used = number_of_residuals;
    return summary;
}

Neighborhood lioOptimization::computeNeighborhoodDistribution(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &points)
{
    // neighborhood是最近点组成的整体表示 内部含有这些点拟合的相关信息
    Neighborhood neighborhood;
    // Compute the normals
    // 计算均值
    Eigen::Vector3d barycenter(Eigen::Vector3d(0, 0, 0));
    for (auto &point: points)
    {
        barycenter += point;
    }

    barycenter /= (double) points.size();
    neighborhood.center = barycenter;

    // 计算协方差矩阵
    Eigen::Matrix3d covariance_Matrix(Eigen::Matrix3d::Zero());
    for (auto &point: points)
    {
        for (int k = 0; k < 3; ++k)
            for (int l = k; l < 3; ++l)
                covariance_Matrix(k, l) += (point(k) - barycenter(k)) *
                                           (point(l) - barycenter(l));
    }
    covariance_Matrix(1, 0) = covariance_Matrix(0, 1);
    covariance_Matrix(2, 0) = covariance_Matrix(0, 2);
    covariance_Matrix(2, 1) = covariance_Matrix(1, 2);
    neighborhood.covariance = covariance_Matrix;
    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_Matrix);
    // 特征向量
    Eigen::Vector3d normal(es.eigenvectors().col(0).normalized());
    neighborhood.normal = normal;
    // 每个特征值的平方根
    double sigma_1 = sqrt(std::abs(es.eigenvalues()[2]));
    double sigma_2 = sqrt(std::abs(es.eigenvalues()[1]));
    double sigma_3 = sqrt(std::abs(es.eigenvalues()[0]));
    neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1;

    if (neighborhood.a2D != neighborhood.a2D) {
        throw std::runtime_error("error");
    }

    return neighborhood;
}

using pair_distance_t = std::tuple<double, Eigen::Vector3d, voxel>;

struct comparator {
    bool operator()(const pair_distance_t &left, const pair_distance_t &right) const {
        return std::get<0>(left) < std::get<0>(right);
    }
};

using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, comparator>;


/*!
 * @brief 根据体素地图查找临近点
 * @param map
 * @param point
 * @param nb_voxels_visited
 * @param size_voxel_map
 * @param max_num_neighbors
 * @param threshold_voxel_capacity
 * @param voxels
 * @return
 */
std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> lioOptimization::searchNeighbors(const voxelHashMap &map, const Eigen::Vector3d &point,
        int nb_voxels_visited, double size_voxel_map, int max_num_neighbors, int threshold_voxel_capacity, std::vector<voxel> *voxels)
{

    if (voxels != nullptr)
        voxels->reserve(max_num_neighbors);
    // 计算该关键点在体素地图中对应的位置
    short kx = static_cast<short>(point[0] / size_voxel_map);
    short ky = static_cast<short>(point[1] / size_voxel_map);
    short kz = static_cast<short>(point[2] / size_voxel_map);
    // 优先队列
    priority_queue_t priority_queue;

    voxel voxel_temp(kx, ky, kz);
    // 遍历当前点所在体素的临近体素 范围为1
    for (short kxx = kx - nb_voxels_visited; kxx < kx + nb_voxels_visited + 1; ++kxx)
    {
        for (short kyy = ky - nb_voxels_visited; kyy < ky + nb_voxels_visited + 1; ++kyy)
        {
            for (short kzz = kz - nb_voxels_visited; kzz < kz + nb_voxels_visited + 1; ++kzz)
            {
                voxel_temp.x = kxx;
                voxel_temp.y = kyy;
                voxel_temp.z = kzz;

                auto search = map.find(voxel_temp);
                // 当前临近体素存在
                if (search != map.end())
                {
                    const auto &voxel_block = search.value();
                    // 检索到的体素中的点数量小于阈值 认为不合格
                    if (voxel_block.NumPoints() < threshold_voxel_capacity)
                        continue;
                    // 遍历当前临近体素中的点
                    for (int i(0); i < voxel_block.NumPoints(); ++i)
                    {
                        auto &neighbor = voxel_block.points[i];
                        // 计算临近体素中点到关键点之间的距离
                        double distance = (neighbor - point).norm();
                        // 优先队列已经达到最大数量了
                        // 判断该距离与队列中最优距离的关系
                        // 如果当前点之间的距离比队列中的最差距离要小 将该点加入队列
                        // 如果队列没有达到最大数量 那么直接加入
                        if (priority_queue.size() == max_num_neighbors)
                        {
                            if (distance < std::get<0>(priority_queue.top()))
                            {
                                priority_queue.pop();
                                priority_queue.emplace(distance, neighbor, voxel_temp);
                            }
                        }
                        else
                            priority_queue.emplace(distance, neighbor, voxel_temp);
                    }
                }
            }
        }
    }

    auto size = priority_queue.size();
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> closest_neighbors(size);
    if (voxels != nullptr)
    {
        voxels->resize(size);
    }
    for (auto i = 0; i < size; ++i)
    {
        // 队列中大小从大到小
        // closest_neighbors从小到大
        closest_neighbors[size - 1 - i] = std::get<1>(priority_queue.top());
        if (voxels != nullptr)
            (*voxels)[size - 1 - i] = std::get<2>(priority_queue.top());
        priority_queue.pop();
    }


    return closest_neighbors;
}

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> lioOptimization::selectClosestNeighbors(
        const std::vector<std::vector<Eigen::Vector3d> const *> &neighbors_ptr, const Eigen::Vector3d &pt_keypoint, int num_neighbors, int max_num_neighbors)
{
    std::vector<std::pair<double, Eigen::Vector3d>> distance_neighbors;
    distance_neighbors.reserve(neighbors_ptr.size());
    for (auto &it_ptr: neighbors_ptr) {
        for (auto &it: *it_ptr) {
            double sq_dist = (pt_keypoint - it).squaredNorm();
            distance_neighbors.emplace_back(sq_dist, it);
        }
    }

    int real_number_neighbors = std::min(max_num_neighbors, (int) distance_neighbors.size());
    std::partial_sort(distance_neighbors.begin(),
                      distance_neighbors.begin() + real_number_neighbors,
                      distance_neighbors.end(),
                      [](const std::pair<double, Eigen::Vector3d> &left,
                         const std::pair<double, Eigen::Vector3d> &right) {
                          return left.first < right.first;
                      });

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> neighbors(real_number_neighbors);
    for (auto i(0); i < real_number_neighbors; ++i)
        neighbors[i] = distance_neighbors[i].second;
    return neighbors;
}

estimationSummary lioOptimization::optimize(cloudFrame *p_frame, const icpOptions &cur_icp_options, estimationSummary &summary, double sample_voxel_size)
{
    std::vector<point3D> keypoints;
    // 网格采样 得到关键点
    gridSampling(p_frame->point_frame, keypoints, sample_voxel_size);

    auto num_keypoints = (int) keypoints.size();
    summary.sample_size = num_keypoints;

    {
        optimizeSummary optimize_summary;

        optimize_summary = optimizeByAnalyticLidar(cur_icp_options, voxel_map, keypoints, p_frame);

        summary.success = optimize_summary.success;
        summary.number_of_residuals = optimize_summary.num_residuals_used;

        if (!summary.success) {
            summary.success = false;
            return summary;
        }
        // 更新当前状态量
        Eigen::Quaterniond q_middle = p_frame->p_state->rotation_middle;
        Eigen::Quaterniond q_end = p_frame->p_state->rotation_end;
        Eigen::Vector3d t_middle = p_frame->p_state->translation_middle;
        Eigen::Vector3d t_end = p_frame->p_state->translation_end;
        for (auto &point_temp: p_frame->point_frame)
        {
            // transformPoint(options.motion_compensation, point_temp, q_begin, q_end, t_begin, t_end, R_imu_lidar, t_imu_lidar);
            transformPoint(options.motion_compensation, point_temp, p_frame->p_state->rotation_last_end, q_middle, q_end, p_frame->p_state->translation_last_end, t_middle, t_end, R_imu_lidar, t_imu_lidar);
        }
    }
    std::vector<point3D>().swap(summary.keypoints);
    summary.keypoints = keypoints;
    summary.state_frame->release();
    summary.state_frame = new state(p_frame->p_state, true);

    return summary;
}