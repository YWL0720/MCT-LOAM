#pragma once
// c++
#include <iostream>

// eigen 
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
// ceres
#include <ceres/ceres.h>

// utility
#include "utility.h"
#include "imuProcessing.h"






/*class CTLidarPlaneNormFactorAutoDiff
{
public:
    CTLidarPlaneNormFactorAutoDiff(const Eigen::Vector3d &raw_keypoint_, const Eigen::Vector3d &norm_vector_,
                                   const double norm_offset_, double alpha_time_, double weight_);
    template <typename T>
    bool operator()(const T* parameters_begin_t, const T* parameters_begin_q, const T* parameters_end_t, const T* parameters_end_q, T* residual) const;

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Vector3d &norm_vector_,
                                       const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<CTLidarPlaneNormFactorAutoDiff, 1, 3, 4, 3, 4>(
                new CTLidarPlaneNormFactorAutoDiff(raw_keypoint_, norm_vector_, norm_offset_, alpha_time_, weight_)
        );
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;

    static Eigen::Vector3d t_il;
    static Eigen::Quaterniond q_il;
    static double sqrt_info;
};*/

class LidarPlaneNormFactor : public ceres::SizedCostFunction<1, 3, 4>
{
public:
    LidarPlaneNormFactor(const Eigen::Vector3d &point_body_, const Eigen::Vector3d &norm_vector_, const double norm_offset_, double weight_ = 1.0);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    void check(double **parameters);

    Eigen::Vector3d point_body;
    Eigen::Vector3d norm_vector;

    double norm_offset;
    double weight;

    static Eigen::Vector3d t_il;
    static Eigen::Quaterniond q_il;
    static double sqrt_info;
};

class CTLidarPlaneNormFactor : public ceres::SizedCostFunction<1, 3, 4, 3, 4>
{
public:
    CTLidarPlaneNormFactor(const Eigen::Vector3d &raw_keypoint_, const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_ = 1.0);

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    void check(double **parameters);

    Eigen::Vector3d raw_keypoint;
    Eigen::Vector3d norm_vector;

    double norm_offset;
    double alpha_time;
    double weight;

    static Eigen::Vector3d t_il;
    static Eigen::Quaterniond q_il;
    static double sqrt_info;
};

class LocationConsistencyFactor : public ceres::SizedCostFunction<3, 3>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    LocationConsistencyFactor(const Eigen::Vector3d &previous_location_, double beta_);

    virtual ~LocationConsistencyFactor() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    Eigen::Vector3d previous_location;
    double beta = 1.0;
};

class RotationConsistencyFactor : public ceres::SizedCostFunction<3, 4>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    RotationConsistencyFactor(const Eigen::Quaterniond &previous_rotation_, double beta_);

    virtual ~RotationConsistencyFactor() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    Eigen::Quaterniond previous_rotation;
    double beta = 1.0;
};

class SmallVelocityFactor : public ceres::SizedCostFunction<3, 3, 3>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SmallVelocityFactor(double beta_);

    virtual ~SmallVelocityFactor() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    double beta;
};

class VelocityConsistencyFactor : public ceres::SizedCostFunction<9, 9>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    VelocityConsistencyFactor(state* previous_state_, double beta_);

    virtual ~VelocityConsistencyFactor() {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

    Eigen::Vector3d previous_velocity;
    Eigen::Vector3d previous_ba;
    Eigen::Vector3d previous_bg;
    double beta = 1.0;
};

// -------------------------------------------------------------------------------------------------------------------------------------------------------------

struct PointToPlaneFunctor {

static constexpr int NumResiduals() { return 1; }

    PointToPlaneFunctor(const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &target,
                        const Eigen::Vector3d &reference_normal,
                        double weight = 1.0) : reference_(reference),
                                               target_(target),
                                               reference_normal_(reference_normal),
                                               weight_(weight) {}

    template<typename T> bool operator()(const T *const rot_params, const T *const trans_params, T *residual) const {
        Eigen::Map<Eigen::Quaternion<T>> quat(const_cast<T *>(rot_params));
        Eigen::Matrix<T, 3, 1> target_temp(T(target_(0, 0)), T(target_(1, 0)), T(target_(2, 0)));
        Eigen::Matrix<T, 3, 1> transformed = quat * target_temp;
        transformed(0, 0) += trans_params[0];
        transformed(1, 0) += trans_params[1];
        transformed(2, 0) += trans_params[2];

        Eigen::Matrix<T, 3, 1> reference_temp(T(reference_(0, 0)), T(reference_(1, 0)), T(reference_(2, 0)));
        Eigen::Matrix<T, 3, 1> reference_normal_temp(T(reference_normal_(0, 0)), T(reference_normal_(1, 0)), T(reference_normal_(2, 0)));

        residual[0] = T(weight_) * (reference_temp - transformed).transpose() * reference_normal_temp;
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d reference_;
    Eigen::Vector3d target_;
    Eigen::Vector3d reference_normal_;
    double weight_ = 1.0;
};

struct CTPointToPlaneFunctor {

    static constexpr int NumResiduals() { return 1; }

    CTPointToPlaneFunctor(const Eigen::Vector3d &reference_point, const Eigen::Vector3d &raw_target,
                          const Eigen::Vector3d &reference_normal, double alpha_timestamp, double weight = 1.0) :
            raw_keypoint_(raw_target),
            reference_point_(reference_point),
            reference_normal_(reference_normal),
            alpha_timestamps_(alpha_timestamp),
            weight_(weight) {}

    template<typename T> bool operator()(const T *const begin_rot_params, const T *begin_trans_params,
                    const T *const end_rot_params, const T *end_trans_params, T *residual) const {
        Eigen::Map<Eigen::Quaternion<T>> quat_begin(const_cast<T *>(begin_rot_params));
        Eigen::Map<Eigen::Quaternion<T>> quat_end(const_cast<T *>(end_rot_params));
        Eigen::Quaternion<T> quat_inter = quat_begin.slerp(T(alpha_timestamps_), quat_end);
        quat_inter.normalize();

        Eigen::Matrix<T, 3, 1> raw_keypoint_temp(T(raw_keypoint_(0, 0)), T(raw_keypoint_(1, 0)), T(raw_keypoint_(2, 0)));

        Eigen::Matrix<T, 3, 1> transformed = quat_inter * raw_keypoint_temp;

        T alpha_m = T(1.0 - alpha_timestamps_);
        transformed(0, 0) += alpha_m * begin_trans_params[0] + alpha_timestamps_ * end_trans_params[0];
        transformed(1, 0) += alpha_m * begin_trans_params[1] + alpha_timestamps_ * end_trans_params[1];
        transformed(2, 0) += alpha_m * begin_trans_params[2] + alpha_timestamps_ * end_trans_params[2];

        Eigen::Matrix<T, 3, 1> reference_point_temp(T(reference_point_(0, 0)), T(reference_point_(1, 0)), T(reference_point_(2, 0)));

        Eigen::Matrix<T, 3, 1> reference_normal_temp(T(reference_normal_(0, 0)), T(reference_normal_(1, 0)), T(reference_normal_(2, 0)));

        residual[0] = T(weight_) * (reference_point_temp - transformed).transpose() * reference_normal_temp;

        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d raw_keypoint_;
    Eigen::Vector3d reference_point_;
    Eigen::Vector3d reference_normal_;
    double alpha_timestamps_;
    double weight_ = 1.0;
};

struct LocationConsistencyFunctor {

    static constexpr int NumResiduals() { return 3; }

    LocationConsistencyFunctor(const Eigen::Vector3d &previous_location,
                               double beta) : beta_(beta), previous_location_(previous_location) {}

    template<typename T> bool operator()(const T *const location_params, T *residual) const {

        Eigen::Matrix<T, 3, 1> previous_location_temp(T(previous_location_(0, 0)), T(previous_location_(1, 0)), T(previous_location_(2, 0)));

        residual[0] = beta_ * (location_params[0] - previous_location_temp(0, 0));
        residual[1] = beta_ * (location_params[1] - previous_location_temp(1, 0));
        residual[2] = beta_ * (location_params[2] - previous_location_temp(2, 0));
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Eigen::Vector3d previous_location_;
    double beta_ = 1.0;
};

// A Functor which enforces frame orientation consistency between two poses
struct OrientationConsistencyFunctor {

    static constexpr int NumResiduals() { return 1; }

    OrientationConsistencyFunctor(const Eigen::Quaterniond &previous_orientation, double beta) : beta_(beta), previous_orientation_(previous_orientation) {}

    template<typename T> bool operator()(const T *const orientation_params, T *residual) const {

        Eigen::Quaternion<T> quat(orientation_params);

        Eigen::Quaternion<T> previous_orientation_temp(T(previous_orientation_.w()), T(previous_orientation_.x()), T(previous_orientation_.y()), T(previous_orientation_.z()));

        T scalar_quat = quat.dot(previous_orientation_temp);

        residual[0] = T(beta_) * (T(1.0) - scalar_quat * scalar_quat);
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Eigen::Quaterniond previous_orientation_;
    double beta_;

};

// A Const Functor which enforces a Constant Velocity constraint on translation
struct ConstantVelocityFunctor {

    static constexpr int NumResiduals() { return 3; }

    ConstantVelocityFunctor(const Eigen::Vector3d &previous_velocity, double beta) : previous_velocity_(previous_velocity), beta_(beta) {}

    template<typename T> bool operator()(const T *const begin_t, const T *const end_t, T *residual) const {

        Eigen::Matrix<T, 3, 1> previous_velocity_temp(T(previous_velocity_(0, 0)), T(previous_velocity_(1, 0)), T(previous_velocity_(2, 0)));

        residual[0] = T(beta_) * (end_t[0] - begin_t[0] - previous_velocity_temp(0, 0));
        residual[1] = T(beta_) * (end_t[1] - begin_t[1] - previous_velocity_temp(1, 0));
        residual[2] = T(beta_) * (end_t[2] - begin_t[2] - previous_velocity_temp(2, 0));
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Eigen::Vector3d previous_velocity_;
    double beta_ = 1.0;
};

// A Const Functor which enforces a Small Velocity constraint
struct SmallVelocityFunctor {

    static constexpr int NumResiduals() { return 3; }

    SmallVelocityFunctor(double beta) : beta_(beta) {};

    template<typename T> bool operator()(const T *const begin_t, const T *const end_t, T *residual) const {
        residual[0] = beta_ * (begin_t[0] - end_t[0]);
        residual[1] = beta_ * (begin_t[1] - end_t[1]);
        residual[2] = beta_ * (begin_t[2] - end_t[2]);
        return true;
    }

    double beta_;
};

class TruncatedLoss : public ceres::LossFunction {
public:
    explicit TruncatedLoss(double sigma) : sigma2_(sigma * sigma) {}

    void Evaluate(double, double *) const override;

private:
    const double sigma2_;
};


struct CTLidarPlaneNormFactorAutoDiff
{
    CTLidarPlaneNormFactorAutoDiff(const Eigen::Vector3d &raw_keypoint_, const Eigen::Vector3d &norm_vector_,
                                   const double norm_offset_, double alpha_time_, double weight_)
            : norm_vector(norm_vector_), norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint_d = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    template <typename T>
    bool operator()(const T* parameters_begin_t, const T* parameters_begin_q, const T* parameters_end_t, const T* parameters_end_q, T* residual) const
    {

        Eigen::Matrix<T, 3, 1> tran_begin{parameters_begin_t[0], parameters_begin_t[1], parameters_begin_t[2]};
        Eigen::Matrix<T, 3, 1> tran_end{parameters_end_t[0], parameters_end_t[1], parameters_end_t[2]};
        Eigen::Quaternion<T> rot_begin{parameters_begin_q[3], parameters_begin_q[0], parameters_begin_q[1], parameters_begin_q[2]};
        Eigen::Quaternion<T> rot_end{parameters_end_q[3], parameters_end_q[0], parameters_end_q[1], parameters_end_q[2]};

        Eigen::Matrix<T, 3, 1> raw_keypoint{T(raw_keypoint_d.x()), T(raw_keypoint_d.y()), T(raw_keypoint_d.z())};
        Eigen::Matrix<T, 3, 1> norm_vector_T{T(norm_vector.x()), T(norm_vector.y()), T(norm_vector.z())};

        Eigen::Quaternion<T> rot_slerp = rot_begin.slerp(T(alpha_time), rot_end);
        rot_slerp.normalize();
        Eigen::Matrix<T, 3, 1> tran_slerp = tran_begin * (T(1.0) - T(alpha_time)) + tran_end * T(alpha_time);
        Eigen::Matrix<T, 3, 1> point_world = rot_slerp * raw_keypoint + tran_slerp;

        T distance = norm_vector_T.dot(point_world) + T(norm_offset);

        residual[0] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Vector3d &norm_vector_,
                                       const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<CTLidarPlaneNormFactorAutoDiff, 1, 3, 4, 3, 4>(
                new CTLidarPlaneNormFactorAutoDiff(raw_keypoint_, norm_vector_, norm_offset_, alpha_time_, weight_)
        );
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;

};

struct MCTLidarPlaneNormFactorAutoDiff
{
    MCTLidarPlaneNormFactorAutoDiff(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                    const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
                                    : raw_keypoint_d(raw_keypoint_), rot_last_end(rot_last_end_), tran_last_end(tran_last_end_), norm_vector(norm_vector_),
                                    norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint_d = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    template <typename T>
    bool operator()(const T* parameters_middle_t, const T* parameters_middle_q, const T* parameters_end_t, const T* parameters_end_q, T* residual) const
    {
        Eigen::Matrix<T, 3, 1> tran_middle{parameters_middle_t[0], parameters_middle_t[1], parameters_middle_t[2]};
        Eigen::Matrix<T, 3, 1> tran_end{parameters_end_t[0], parameters_end_t[1], parameters_end_t[2]};
        Eigen::Quaternion<T> rot_middle{parameters_middle_q[3], parameters_middle_q[0], parameters_middle_q[1], parameters_middle_q[2]};
        Eigen::Quaternion<T> rot_end{parameters_end_q[3], parameters_end_q[0], parameters_end_q[1], parameters_end_q[2]};

        Eigen::Matrix<T, 3, 1> raw_keypoint{T(raw_keypoint_d.x()), T(raw_keypoint_d.y()), T(raw_keypoint_d.z())};
        Eigen::Matrix<T, 3, 1> norm_vector_T{T(norm_vector.x()), T(norm_vector.y()), T(norm_vector.z())};

        Eigen::Quaternion<T> rot_last_end_T{T(rot_last_end.w()), T(rot_last_end.x()), T(rot_last_end.y()), T(rot_last_end.z())};
        Eigen::Matrix<T, 3, 1> tran_last_end_T{T(tran_last_end.x()), T(tran_last_end.y()), T(tran_last_end.z())};

        Eigen::Quaternion<T> rot_slerp;
        Eigen::Matrix<T, 3, 1> tran_slerp;

        if (alpha_time >= 0)
        {
            rot_slerp = rot_middle.slerp(T(alpha_time * 2 - 1), rot_end);
            tran_slerp = tran_middle + T(alpha_time * 2 - 1) * (tran_end - tran_middle);

        }
        else
        {
            rot_slerp = rot_last_end_T.slerp(T(alpha_time * 2), rot_middle);
            tran_slerp = tran_last_end_T + T(alpha_time * 2) * (tran_middle - tran_last_end_T);
        }

        rot_slerp.normalize();
        Eigen::Matrix<T, 3, 1> point_world = rot_slerp * raw_keypoint + tran_slerp;

        T distance = norm_vector_T.dot(point_world) + T(norm_offset);

        residual[0] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                       const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<MCTLidarPlaneNormFactorAutoDiff, 1, 3, 4, 3, 4>(new MCTLidarPlaneNormFactorAutoDiff(
                raw_keypoint_, rot_last_end_, tran_last_end_, norm_vector_, norm_offset_, alpha_time_, weight_
                ));
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Quaterniond rot_last_end;
    Eigen::Vector3d tran_last_end;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;
};



class Spline3D
{
public:
    Spline3D(Eigen::VectorXd vTime, Eigen::Matrix<double, Eigen::Dynamic, 3> vPoints)
    {
        spline_ = Eigen::SplineFitting<Eigen::Spline<double, 3>>::Interpolate(vPoints.transpose(), 2, vTime.transpose());
    };
    Eigen::Vector3d operator()(double x)
    {
        x = x > 1.0 ? 1.0 : x < 0.0 ? 0.0 : x;
        return Eigen::Vector3d(spline_(x)(0), spline_(x)(1), spline_(x)(2));
    }
private:
    Eigen::Spline<double, 3> spline_;
};


struct MCTLidarPlaneNormFactorFirstAutoDiff
{
    MCTLidarPlaneNormFactorFirstAutoDiff(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                    const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
            : raw_keypoint_d(raw_keypoint_), rot_last_end(rot_last_end_), tran_last_end(tran_last_end_), norm_vector(norm_vector_),
              norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint_d = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    template <typename T>
    bool operator()(const T* parameters_middle_t, const T* parameters_middle_q, T* residual) const
    {
        Eigen::Matrix<T, 3, 1> tran_middle{parameters_middle_t[0], parameters_middle_t[1], parameters_middle_t[2]};
        Eigen::Quaternion<T> rot_middle{parameters_middle_q[3], parameters_middle_q[0], parameters_middle_q[1], parameters_middle_q[2]};

        Eigen::Matrix<T, 3, 1> raw_keypoint{T(raw_keypoint_d.x()), T(raw_keypoint_d.y()), T(raw_keypoint_d.z())};
        Eigen::Matrix<T, 3, 1> norm_vector_T{T(norm_vector.x()), T(norm_vector.y()), T(norm_vector.z())};

        Eigen::Quaternion<T> rot_last_end_T{T(rot_last_end.w()), T(rot_last_end.x()), T(rot_last_end.y()), T(rot_last_end.z())};
        Eigen::Matrix<T, 3, 1> tran_last_end_T{T(tran_last_end.x()), T(tran_last_end.y()), T(tran_last_end.z())};

        Eigen::Quaternion<T> rot_slerp;
        Eigen::Matrix<T, 3, 1> tran_slerp;


        rot_slerp = rot_last_end_T.slerp(T(alpha_time * 2), rot_middle);
        tran_slerp = tran_last_end_T + T(alpha_time * 2) * (tran_middle - tran_last_end_T);


        rot_slerp.normalize();
        Eigen::Matrix<T, 3, 1> point_world = rot_slerp * raw_keypoint + tran_slerp;

        T distance = norm_vector_T.dot(point_world) + T(norm_offset);

        residual[0] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                       const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<MCTLidarPlaneNormFactorFirstAutoDiff, 1, 3, 4>(new MCTLidarPlaneNormFactorFirstAutoDiff(
                raw_keypoint_, rot_last_end_, tran_last_end_, norm_vector_, norm_offset_, alpha_time_, weight_
        ));
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Quaterniond rot_last_end;
    Eigen::Vector3d tran_last_end;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;
};

struct MCTLidarPlaneNormFactorSecondAutoDiff
{
    MCTLidarPlaneNormFactorSecondAutoDiff(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                          const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
            : raw_keypoint_d(raw_keypoint_), rot_last_end(rot_last_end_), tran_last_end(tran_last_end_), norm_vector(norm_vector_),
              norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint_d = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    template <typename T>
    bool operator()(const T* parameters_middle_t, const T* parameters_middle_q, const T* parameters_end_t, const T* parameters_end_q, T* residual) const
    {
        Eigen::Matrix<T, 3, 1> tran_middle{parameters_middle_t[0], parameters_middle_t[1], parameters_middle_t[2]};
        Eigen::Matrix<T, 3, 1> tran_end{parameters_end_t[0], parameters_end_t[1], parameters_end_t[2]};
        Eigen::Quaternion<T> rot_middle{parameters_middle_q[3], parameters_middle_q[0], parameters_middle_q[1], parameters_middle_q[2]};
        Eigen::Quaternion<T> rot_end{parameters_end_q[3], parameters_end_q[0], parameters_end_q[1], parameters_end_q[2]};

        Eigen::Matrix<T, 3, 1> raw_keypoint{T(raw_keypoint_d.x()), T(raw_keypoint_d.y()), T(raw_keypoint_d.z())};
        Eigen::Matrix<T, 3, 1> norm_vector_T{T(norm_vector.x()), T(norm_vector.y()), T(norm_vector.z())};

        Eigen::Quaternion<T> rot_last_end_T{T(rot_last_end.w()), T(rot_last_end.x()), T(rot_last_end.y()), T(rot_last_end.z())};
        Eigen::Matrix<T, 3, 1> tran_last_end_T{T(tran_last_end.x()), T(tran_last_end.y()), T(tran_last_end.z())};

        Eigen::Quaternion<T> rot_slerp;
        Eigen::Matrix<T, 3, 1> tran_slerp;

        rot_slerp = rot_middle.slerp(T(alpha_time * 2 - 1), rot_end);
        tran_slerp = tran_middle + T(alpha_time * 2 - 1) * (tran_end - tran_middle);

        rot_slerp.normalize();
        Eigen::Matrix<T, 3, 1> point_world = rot_slerp * raw_keypoint + tran_slerp;

        T distance = norm_vector_T.dot(point_world) + T(norm_offset);

        residual[0] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                       const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<MCTLidarPlaneNormFactorSecondAutoDiff, 1, 3, 4, 3, 4>(new MCTLidarPlaneNormFactorSecondAutoDiff(
                raw_keypoint_, rot_last_end_, tran_last_end_, norm_vector_, norm_offset_, alpha_time_, weight_
        ));
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Quaterniond rot_last_end;
    Eigen::Vector3d tran_last_end;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;
};


/*!
 * @brief MCT约束自动求导方法 只需添加这一段约束 适用于所有点
 *
 */
struct MCTLidarPlaneNormFactorAD
{
    MCTLidarPlaneNormFactorAD(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                          const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
            : raw_keypoint_d(raw_keypoint_), rot_last_end(rot_last_end_), tran_last_end(tran_last_end_), norm_vector(norm_vector_),
              norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint_d = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    template <typename T>
    bool operator()(const T* parameters_middle_t, const T* parameters_middle_q, const T* parameters_end_t, const T* parameters_end_q, T* residual) const
    {
        // 优化变量 中间时刻的状态和结束时刻的状态
        Eigen::Matrix<T, 3, 1> tran_middle{parameters_middle_t[0], parameters_middle_t[1], parameters_middle_t[2]};
        Eigen::Matrix<T, 3, 1> tran_end{parameters_end_t[0], parameters_end_t[1], parameters_end_t[2]};
        Eigen::Quaternion<T> rot_middle{parameters_middle_q[3], parameters_middle_q[0], parameters_middle_q[1], parameters_middle_q[2]};
        Eigen::Quaternion<T> rot_end{parameters_end_q[3], parameters_end_q[0], parameters_end_q[1], parameters_end_q[2]};
        // 原始数据点和相关的计算量
        Eigen::Matrix<T, 3, 1> raw_keypoint{T(raw_keypoint_d.x()), T(raw_keypoint_d.y()), T(raw_keypoint_d.z())};
        Eigen::Matrix<T, 3, 1> norm_vector_T{T(norm_vector.x()), T(norm_vector.y()), T(norm_vector.z())};
        // 上一帧结束时刻的位姿 固定
        Eigen::Quaternion<T> rot_last_end_T{T(rot_last_end.w()), T(rot_last_end.x()), T(rot_last_end.y()), T(rot_last_end.z())};
        Eigen::Matrix<T, 3, 1> tran_last_end_T{T(tran_last_end.x()), T(tran_last_end.y()), T(tran_last_end.z())};

        // 旋转插值
        Eigen::Quaternion<T> rot_squad;
        Eigen::Quaternion<T> rot_tmp_1;
        Eigen::Quaternion<T> rot_tmp_2;

        Eigen::Matrix<T, 3, 1> tran_squad;

//        rot_tmp_1 = rot_last_end_T.slerp(T(alpha_time), rot_end);
//        rot_tmp_2 = rot_middle.slerp(T(alpha_time * 2 - 1), rot_end);
//        rot_squad = rot_tmp_1.slerp(T(2*alpha_time*(2-alpha_time * 2)), rot_tmp_2);
//        rot_squad.normalize();

        rot_tmp_1 = rot_last_end_T.slerp(T(alpha_time * 2), rot_middle);
        rot_tmp_2 = rot_middle.slerp(T(2 * alpha_time - 1), rot_end);

        // 平移插值
        Eigen::Matrix<T, 3, 1> tran_tmp_1;
        Eigen::Matrix<T, 3, 1> tran_tmp_2;
        Eigen::Matrix<T, 3, 1> tran_tmp_3;
//        tran_tmp_1 = tran_last_end_T + T(alpha_time * 2) * (tran_middle - tran_last_end_T);
//        tran_tmp_2 = tran_middle + T(alpha_time * 2 - 1) * (tran_end - tran_middle);
//        tran_squad = tran_tmp_1 + T(4*alpha_time*(2-alpha_time * 2))*(tran_tmp_2-tran_tmp_1);
//        Eigen::Matrix<T, 3, 1> point_world = rot_squad * raw_keypoint + tran_squad;

        tran_tmp_1 = tran_last_end_T + T(2 * alpha_time) * (tran_middle - tran_last_end_T);
        tran_tmp_2 = tran_middle + T(2 * alpha_time - 1) * (tran_end - tran_middle);
        // Eigen::Matrix<T, 3, 1> point_world = rot_squad * raw_keypoint + tran_squad;
//        T distance = norm_vector_T.dot(point_world) + T(norm_offset);

        Eigen::Matrix<T, 3, 1> point_world_1 = rot_tmp_1 * raw_keypoint + tran_tmp_1;
        Eigen::Matrix<T, 3, 1> point_world_2 = rot_tmp_2 * raw_keypoint + tran_tmp_2;

        T distance_1 = norm_vector_T.dot(point_world_1) + T(norm_offset);
        T distance_2 = norm_vector_T.dot(point_world_2) + T(norm_offset);
        residual[0] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance_1);
        residual[1] = T(CTLidarPlaneNormFactor::sqrt_info) * T(weight) * T(distance_2);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                       const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::AutoDiffCostFunction<MCTLidarPlaneNormFactorAD, 2, 3, 4, 3, 4>(new MCTLidarPlaneNormFactorAD(
                raw_keypoint_, rot_last_end_, tran_last_end_, norm_vector_, norm_offset_, alpha_time_, weight_
        ));
    }


    Eigen::Vector3d raw_keypoint_d;
    Eigen::Quaterniond rot_last_end;
    Eigen::Vector3d tran_last_end;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;
};

/*!
 * @brief MCT约束数值求导方法 只需添加这一段约束 适用于所有点
 *
 */
struct MCTLidarPlaneNormFactorND
{
    MCTLidarPlaneNormFactorND(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                              const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
            : raw_keypoint(raw_keypoint_), rot_last_end(rot_last_end_), tran_last_end(tran_last_end_), norm_vector(norm_vector_),
              norm_offset(norm_offset_), alpha_time(alpha_time_), weight(weight_)
    {
        raw_keypoint = CTLidarPlaneNormFactor::q_il * raw_keypoint_ + CTLidarPlaneNormFactor::t_il;
    };

    bool operator()(const double* const parameters_middle_t, const double* const parameters_middle_q,
                    const double* const parameters_end_t,    const double* const parameters_end_q,    double* residual) const
    {
        // 优化变量 中间时刻的状态和结束时刻的状态
        Eigen::Vector3d tran_middle{parameters_middle_t[0], parameters_middle_t[1], parameters_middle_t[2]};
        Eigen::Vector3d tran_end{parameters_end_t[0], parameters_end_t[1], parameters_end_t[2]};
        Eigen::Quaterniond rot_middle{parameters_middle_q[3], parameters_middle_q[0], parameters_middle_q[1], parameters_middle_q[2]};
        Eigen::Quaterniond rot_end{parameters_end_q[3], parameters_end_q[0], parameters_end_q[1], parameters_end_q[2]};

        // 旋转插值
        Eigen::Quaterniond rot_squad;
        Eigen::Quaterniond rot_tmp_1;
        Eigen::Quaterniond rot_tmp_2;

        rot_tmp_1 = rot_last_end.slerp(alpha_time, rot_end);
        rot_tmp_2 = rot_middle.slerp(alpha_time * 2 - 1, rot_end);

        rot_squad = rot_tmp_1.slerp(2*alpha_time*(2-alpha_time * 2), rot_tmp_2);
        rot_squad.normalize();
        // 平移插值
        Eigen::Vector3d tran_squad;
        /**********************************************************************/
        // Eigen样条插值

        Eigen::VectorXd vTimes(3);
        vTimes << 0.0, 0.5, 1.0;

        Eigen::Matrix<double, Eigen::Dynamic, 3> vPoints(3, 3);
        vPoints << tran_last_end.x(), tran_middle.x(), tran_end.z(),
                tran_last_end.y(), tran_middle.y(), tran_end.y(),
                tran_last_end.z(), tran_middle.z(), tran_end.z();

        Spline3D s(vTimes, vPoints);

        tran_squad = s(alpha_time);

//        Eigen::Vector3d tran_tmp_1;
//        Eigen::Vector3d tran_tmp_2;
//        Eigen::Vector3d tran_tmp_3;
//        tran_tmp_1 = tran_last_end + (alpha_time * 2) * (tran_middle - tran_last_end);
//        tran_tmp_2 = tran_middle + (alpha_time * 2 - 1) * (tran_end - tran_middle);
//
//        tran_squad = tran_tmp_1 + (4*alpha_time*(2-alpha_time * 2))*(tran_tmp_2-tran_tmp_1);

        /**********************************************************************/

//        if (alpha_time >= 0.724 && alpha_time <= 0.73)
//        {
//            std::cout << "=========================================" << std::endl;
//            std::cout << "tran_last_end = " << tran_last_end.transpose() << std::endl;
//            std::cout << "tran_middle = " << tran_middle.transpose() << std::endl;
//            std::cout << "tran_end = " << tran_end.transpose() << std::endl;
//            std::cout << "alpha_time = " << alpha_time << std::endl;
//            std::cout << "tran_squad = " << tran_squad.transpose() << std::endl;
//        }

        Eigen::Vector3d point_world = rot_squad * raw_keypoint + tran_squad;
        double distance = norm_vector.dot(point_world) + norm_offset;

        residual[0] = CTLidarPlaneNormFactor::sqrt_info * weight * distance;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d &raw_keypoint_, const Eigen::Quaterniond rot_last_end_, const Eigen::Vector3d tran_last_end_,
                                       const Eigen::Vector3d &norm_vector_, const double norm_offset_, double alpha_time_, double weight_)
    {
        return new ceres::NumericDiffCostFunction<MCTLidarPlaneNormFactorND, ceres::CENTRAL, 1, 3, 4, 3, 4>(new MCTLidarPlaneNormFactorND(
                raw_keypoint_, rot_last_end_, tran_last_end_, norm_vector_, norm_offset_, alpha_time_, weight_
        ));
    }


    Eigen::Vector3d raw_keypoint;
    Eigen::Quaterniond rot_last_end;
    Eigen::Vector3d tran_last_end;
    Eigen::Vector3d norm_vector;
    double norm_offset, alpha_time, weight;
};

