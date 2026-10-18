/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_LINE_TRIANGULATION_H
#define OV_MSCKF_LINE_TRIANGULATION_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>
#include <vector>

#include "feat/LineFeature.h"
#include "types/LineRepresentation.h"

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Helper for line triangulation and line measurement Jacobians.
 *
 * This mirrors ov_msckf::UpdaterHelper for point features but is specialized to 3D
 * Plücker lines. A line is stored in either a global (GLOBAL_PLUCKER) or an anchored
 * (ANCHORED_PLUCKER) camera frame, and is always represented internally by the 4-DOF
 * orthonormal parametrization (see ov_core::line::boxplus). All Jacobians are with
 * respect to that 4-DOF error state and the OpenVINS left-perturbation pose error state.
 */
class LineTriangulation {

public:
  /**
   * @brief Updater-facing line feature with triangulated mean and all measurements.
   */
  struct LineFeatureTri {

    /// Unique ID of this feature
    size_t featid;

    /// Representation of this line feature
    ov_type::LineRepresentation::Representation representation = ov_type::LineRepresentation::Representation::GLOBAL_PLUCKER;

    /// Anchor camera / clone for ANCHORED_PLUCKER lines
    int anchor_cam_id = -1;
    double anchor_clone_timestamp = -1;

    /// Measurements (mapped by camera id): normalized homogeneous endpoints [us; vs; ue; ve]
    std::unordered_map<size_t, std::vector<ov_core::LineObservation>> observations;

    /// Triangulated Plücker line (global or anchored frame), current estimate
    Eigen::Vector3d n;
    Eigen::Vector3d d;

    /// Triangulated Plücker line first estimate (FEJ linearization point)
    Eigen::Vector3d n_fej;
    Eigen::Vector3d d_fej;
  };

  /**
   * @brief Triangulate a global Plücker line from two or more camera observations.
   *
   * For each observation the measured line l_i = P_s x P_e is back-projected to the
   * plane pi_i = R_GtoC_i^T l_i with offset rho_i = pi_i^T p_CinG_i. The two views with
   * the largest baseline are intersected; if degenerate, the next pair is tried.
   *
   * @param state Filter state (for clones and extrinsics)
   * @param observations Per-camera line observations
   * @param n_G, d_G Output normalized global Plücker line (||n||^2 + ||d||^2 = 1)
   * @return true if a non-degenerate intersection was found
   */
  static bool triangulate(std::shared_ptr<State> state,
                          const std::unordered_map<size_t, std::vector<ov_core::LineObservation>> &observations,
                          Eigen::Vector3d &n_G, Eigen::Vector3d &d_G);

  /**
   * @brief Triangulate a line in the anchor camera frame.
   *
   * Triangulates globally and then applies the inverse anchor transform
   * X_A = R_A^T (X_G - p_AinG) to express the line in the anchor frame.
   *
   * @return true if triangulation succeeded
   */
  static bool triangulate_anchored(std::shared_ptr<State> state,
                                   const std::unordered_map<size_t, std::vector<ov_core::LineObservation>> &observations,
                                   int anchor_cam_id, double anchor_clone_timestamp, Eigen::Vector3d &n_A,
                                   Eigen::Vector3d &d_A);

  /**
   * @brief Construct the stacked measurement Jacobians for a single line feature.
   *
   * The line residual is the (negative) point-to-line distance of each observed endpoint
   * to the predicted line: res = -[P_s^T l/m ; P_e^T l/m] (measurement z = 0 minus the
   * prediction h = line_residual). H_f and H_x are the Jacobians of the prediction h with
   * respect to the 4-DOF line error state and the state (clones + extrinsics), matching the
   * convention of StateHelper::EKFUpdate.
   *
   * @param state Filter state
   * @param feat Line feature (must have triangulated n/d and observations)
   * @param H_f Jacobian wrt the 4-DOF line error state (2k x 4)
   * @param H_x Jacobian wrt the state (2k x total_hx)
   * @param res Residual (2k x 1)
   * @param x_order State variables ordered as the columns of H_x
   */
  static void get_jacobian_full(std::shared_ptr<State> state, LineFeatureTri &feat, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x,
                                Eigen::VectorXd &res, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief Jacobian of the global Plücker line wrt the 4-DOF line error state and anchor states.
   *
   * For GLOBAL_PLUCKER this is simply H_f = boxplus_jacobian(n_G, d_G) (6x4) with no anchor
   * variables. For ANCHORED_PLUCKER it chains through the anchor transform X_G = R_A X_A + p_AinG
   * and additionally returns the Jacobians of the global line wrt the anchor clone and (if
   * calibrated) anchor extrinsic, one 6x6 block per variable in x_order.
   *
   * @param state Filter state
   * @param feat Line feature (must have triangulated n/d and a valid anchor for ANCHORED)
   * @param H_f Jacobian wrt the 4-DOF line error state (6x4)
   * @param H_x Jacobians wrt anchor variables (each 6x6, empty for GLOBAL)
   * @param x_order Anchor variables corresponding to each H_x block
   */
  static void get_jacobian_representation(std::shared_ptr<State> state, LineFeatureTri &feat, Eigen::MatrixXd &H_f,
                                          std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief Jacobian of d[n_dest; d_dest] / d[pose error] for the transform X_dest = R X_src + t.
   *
   * The pose error state is [dtheta; dp] with the OpenVINS JPL convention
   * R <- Exp(-dtheta) R, t <- t + dp (matching PoseJPL::update). For a Plücker
   * line this is the 6x6 matrix
   *   [ +[R n_src]x + [t]x [d_dest]x ,  -[d_dest]x ;
   *     +[d_dest]x                   ,       0     ]
   *
   * @param R, t Rotation/translation of the frame transform
   * @param n_src, d_src Source line
   * @return 6x6 Jacobian
   */
  static Eigen::Matrix<double, 6, 6> transform_jacobian_wrt_pose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                                                                 const Eigen::Vector3d &n_src, const Eigen::Vector3d &d_src);
};

} // namespace ov_msckf

#endif // OV_MSCKF_LINE_TRIANGULATION_H
