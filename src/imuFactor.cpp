#include "imuFactor.h"

ImuFactor::ImuFactor(imuIntegration* pre_integration_, state* last_state_) : pre_integration(pre_integration_)
{
	rot_last = last_state_->rotation;
	tran_last = last_state_->translation;
	velocity_last = last_state_->velocity;
	ba_last = last_state_->ba;
	bg_last = last_state_->bg;
}

bool ImuFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    Eigen::Vector3d tran_cur(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond rot_cur(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d velocity_cur(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d ba_cur(parameters[2][3], parameters[2][4], parameters[2][5]);
    Eigen::Vector3d bg_cur(parameters[2][6], parameters[2][7], parameters[2][8]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = pre_integration->evaluate(tran_last, rot_last, velocity_last, ba_last, bg_last, tran_cur, rot_cur, velocity_cur, ba_cur, bg_cur);

    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
    residual = sqrt_info * residual;

    if (jacobians)
    {
    	double sum_dt = pre_integration->sum_dt;
    	Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);
        Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

        if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
        {
            ROS_WARN("numerical unstable in preintegration");
        }

        if (jacobians[0])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_tran_cur(jacobians[0]);
        	jacobian_tran_cur.setZero();

            jacobian_tran_cur.block<3, 3>(O_P, O_P) = rot_last.inverse().toRotationMatrix();
            jacobian_tran_cur = sqrt_info * jacobian_tran_cur;
        }
        if (jacobians[1])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_rot_cur(jacobians[1]);
        	jacobian_rot_cur.setZero();

        	Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * numType::deltaQ(dq_dbg * (bg_last - pre_integration->linearized_bg));
        	jacobian_rot_cur.block<3, 3>(O_R, O_R - O_R) = numType::Qleft(corrected_delta_q.inverse() * rot_last.inverse() * rot_cur).bottomRightCorner<3, 3>();
        	jacobian_rot_cur = sqrt_info * jacobian_rot_cur;
        }
        if (jacobians[2])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_velocity_bias_cur(jacobians[2]);
            jacobian_velocity_bias_cur.setZero();

            jacobian_velocity_bias_cur.block<3, 3>(O_V, O_V - O_V) = rot_last.inverse().toRotationMatrix();
            jacobian_velocity_bias_cur.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
            jacobian_velocity_bias_cur.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
            jacobian_velocity_bias_cur = sqrt_info * jacobian_velocity_bias_cur;
        }
    }

    return true;
}

CTImuFactor::CTImuFactor(imuIntegration* pre_integration_, int beta_) : pre_integration(pre_integration_), beta(beta_)
{

}

bool CTImuFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

	Eigen::Vector3d tran_last(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond rot_last(parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d velocity_last(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Vector3d ba_last(parameters[2][3], parameters[2][4], parameters[2][5]);
    Eigen::Vector3d bg_last(parameters[2][6], parameters[2][7], parameters[2][8]);

    Eigen::Vector3d tran_cur(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Quaterniond rot_cur(parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]);
    Eigen::Vector3d velocity_cur(parameters[5][0], parameters[5][1], parameters[5][2]);
    Eigen::Vector3d ba_cur(parameters[5][3], parameters[5][4], parameters[5][5]);
    Eigen::Vector3d bg_cur(parameters[5][6], parameters[5][7], parameters[5][8]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = pre_integration->evaluate(tran_last, rot_last, velocity_last, ba_last, bg_last, tran_cur, rot_cur, velocity_cur, ba_cur, bg_cur);

    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
    residual = sqrt_info * residual * beta;

    if (jacobians)
    {
    	double sum_dt = pre_integration->sum_dt;
    	Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);
        Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

        if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
        {
            ROS_WARN("numerical unstable in preintegration");
        }

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_tran_last(jacobians[0]);
            jacobian_tran_last.setZero();

            jacobian_tran_last.block<3, 3>(O_P, O_P) = - rot_last.inverse().toRotationMatrix();
            jacobian_tran_last = sqrt_info * jacobian_tran_last * beta;
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_rot_last(jacobians[1]);
            jacobian_rot_last.setZero();

            jacobian_rot_last.block<3, 3>(O_P, O_R - O_R) = numType::skewSymmetric(rot_last.inverse() * (0.5 * G * sum_dt * sum_dt + tran_cur - tran_last - velocity_last * sum_dt));
            Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * numType::deltaQ(dq_dbg * (bg_last - pre_integration->linearized_bg));
            jacobian_rot_last.block<3, 3>(O_R, O_R - O_R) = - (numType::Qleft(rot_cur.inverse() * rot_last) * numType::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
            jacobian_rot_last.block<3, 3>(O_V, O_R - O_R) = numType::skewSymmetric(rot_last.inverse() * (G * sum_dt + velocity_cur - velocity_last));
            jacobian_rot_last = sqrt_info * jacobian_rot_last * beta;
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_velocity_bias_last(jacobians[2]);
            jacobian_velocity_bias_last.setZero();
            jacobian_velocity_bias_last.block<3, 3>(O_P, O_V - O_V) = - rot_last.inverse().toRotationMatrix() * sum_dt;
            jacobian_velocity_bias_last.block<3, 3>(O_P, O_BA - O_V) = - dp_dba;
            jacobian_velocity_bias_last.block<3, 3>(O_P, O_BG - O_V) = - dp_dbg;

            jacobian_velocity_bias_last.block<3, 3>(O_R, O_BG - O_V) = - numType::Qleft(rot_cur.inverse() * rot_last * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;

            jacobian_velocity_bias_last.block<3, 3>(O_V, O_V - O_V) = - rot_last.inverse().toRotationMatrix();
            jacobian_velocity_bias_last.block<3, 3>(O_V, O_BA - O_V) = - dv_dba;
            jacobian_velocity_bias_last.block<3, 3>(O_V, O_BG - O_V) = - dv_dbg;

            jacobian_velocity_bias_last.block<3, 3>(O_BA, O_BA - O_V) = - Eigen::Matrix3d::Identity();

            jacobian_velocity_bias_last.block<3, 3>(O_BG, O_BG - O_V) = - Eigen::Matrix3d::Identity();

            jacobian_velocity_bias_last = sqrt_info * jacobian_velocity_bias_last * beta;
        }
        if (jacobians[3])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_tran_cur(jacobians[3]);
        	jacobian_tran_cur.setZero();

            jacobian_tran_cur.block<3, 3>(O_P, O_P) = rot_last.inverse().toRotationMatrix();
            jacobian_tran_cur = sqrt_info * jacobian_tran_cur * beta;
        }
        if (jacobians[4])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> jacobian_rot_cur(jacobians[4]);
        	jacobian_rot_cur.setZero();

        	Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * numType::deltaQ(dq_dbg * (bg_last - pre_integration->linearized_bg));
        	jacobian_rot_cur.block<3, 3>(O_R, O_R - O_R) = numType::Qleft(corrected_delta_q.inverse() * rot_last.inverse() * rot_cur).bottomRightCorner<3, 3>();
        	jacobian_rot_cur = sqrt_info * jacobian_rot_cur * beta;
        }
        if (jacobians[5])
        {
        	Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_velocity_bias_cur(jacobians[5]);
            jacobian_velocity_bias_cur.setZero();

            jacobian_velocity_bias_cur.block<3, 3>(O_V, O_V - O_V) = rot_last.inverse().toRotationMatrix();
            jacobian_velocity_bias_cur.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
            jacobian_velocity_bias_cur.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
            jacobian_velocity_bias_cur = sqrt_info * jacobian_velocity_bias_cur * beta;
        }
    }

    return true;
}

