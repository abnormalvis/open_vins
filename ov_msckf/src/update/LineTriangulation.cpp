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

#include "LineTriangulation.h"

#include <algorithm>

#include "line/LineMath.h"
#include "state/State.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

/**
 * @brief Camera pose (R_GtoC, p_CinG) from an IMU clone and camera extrinsic.
 *
 * The extrinsic is always read at its current value (matching the point updater, which
 * does not FEJ the calibration). The clone is read at fej or current depending on use_fej.
 */
void camera_pose(const std::shared_ptr<State> &state, size_t cam_id, double timestamp, bool use_fej, Eigen::Matrix3d &R_GtoC,
                 Eigen::Vector3d &p_CinG) {
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(cam_id);
  Eigen::Matrix3d R_ItoC = calibration->Rot();
  Eigen::Vector3d p_IinC = calibration->pos();
  std::shared_ptr<PoseJPL> clone = state->_clones_IMU.at(timestamp);
  Eigen::Matrix3d R_GtoI = (use_fej) ? clone->Rot_fej() : clone->Rot();
  Eigen::Vector3d p_IinG = (use_fej) ? clone->pos_fej() : clone->pos();
  R_GtoC = R_ItoC * R_GtoI;
  p_CinG = p_IinG - R_GtoC.transpose() * p_IinC;
}

/**
 * @brief Anchor camera pose (R_A, p_AinG) expressed as the anchor-to-global transform.
 *
 * R_A = R_GtoI^T R_ItoC^T (anchor camera -> global), p_AinG = p_IinG - R_A p_IinC.
 */
void anchor_pose(const std::shared_ptr<State> &state, int anchor_cam_id, double anchor_timestamp, bool use_fej,
                 Eigen::Matrix3d &R_A, Eigen::Vector3d &p_AinG) {
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(anchor_cam_id);
  Eigen::Matrix3d R_ItoC = calibration->Rot();
  Eigen::Vector3d p_IinC = calibration->pos();
  std::shared_ptr<PoseJPL> clone = state->_clones_IMU.at(anchor_timestamp);
  Eigen::Matrix3d R_GtoI = (use_fej) ? clone->Rot_fej() : clone->Rot();
  Eigen::Vector3d p_IinG = (use_fej) ? clone->pos_fej() : clone->pos();
  R_A = R_GtoI.transpose() * R_ItoC.transpose();
  p_AinG = p_IinG - R_A * p_IinC;
}

/**
 * @brief Express the line in the global frame.
 *
 * For GLOBAL_PLUCKER the line is already global. For ANCHORED_PLUCKER it is transformed
 * through the anchor pose X_G = R_A X_A + p_AinG.
 */
void global_line(const std::shared_ptr<State> &state, const LineTriangulation::LineFeatureTri &feat, bool use_fej,
                 Eigen::Vector3d &n_G, Eigen::Vector3d &d_G) {
  if (feat.representation == LineRepresentation::Representation::GLOBAL_PLUCKER) {
    n_G = (use_fej) ? feat.n_fej : feat.n;
    d_G = (use_fej) ? feat.d_fej : feat.d;
    return;
  }
  Eigen::Matrix3d R_A;
  Eigen::Vector3d p_AinG;
  anchor_pose(state, feat.anchor_cam_id, feat.anchor_clone_timestamp, use_fej, R_A, p_AinG);
  const Eigen::Vector3d n_A = (use_fej) ? feat.n_fej : feat.n;
  const Eigen::Vector3d d_A = (use_fej) ? feat.d_fej : feat.d;
  line::transform_plucker(R_A, p_AinG, n_A, d_A, n_G, d_G);
}

} // namespace

bool LineTriangulation::triangulate(std::shared_ptr<State> state,
                                    const std::unordered_map<size_t, std::vector<ov_core::LineObservation>> &observations,
                                    Eigen::Vector3d &n_G, Eigen::Vector3d &d_G) {

  // Back-project every observation into a global plane pi^T X = rho.
  struct View {
    Eigen::Vector3d pi;
    double rho;
    Eigen::Vector3d p;
  };
  std::vector<View> views;
  views.reserve(8);
  for (auto const &pair : observations) {
    for (auto const &o : pair.second) {
      Eigen::Matrix3d R_GtoC;
      Eigen::Vector3d p_CinG;
      camera_pose(state, pair.first, o.timestamp, false, R_GtoC, p_CinG);
      Eigen::Vector3d P_s(o.endpoints_norm(0), o.endpoints_norm(1), 1.0);
      Eigen::Vector3d P_e(o.endpoints_norm(2), o.endpoints_norm(3), 1.0);
      Eigen::Vector3d l = P_s.cross(P_e);
      if (l.norm() < 1e-8) {
        continue;
      }
      Eigen::Vector3d pi = R_GtoC.transpose() * l;
      views.push_back({pi, pi.dot(p_CinG), p_CinG});
    }
  }
  if (views.size() < 2) {
    return false;
  }

  // Try pairs in descending baseline order (most informative first).
  std::vector<std::pair<double, std::pair<size_t, size_t>>> pairs;
  pairs.reserve(views.size() * views.size());
  for (size_t i = 0; i < views.size(); ++i) {
    for (size_t j = i + 1; j < views.size(); ++j) {
      pairs.push_back({(views[i].p - views[j].p).norm(), {i, j}});
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](auto const &a, auto const &b) { return a.first > b.first; });

  for (auto const &pair : pairs) {
    if (line::triangulate_from_planes(views[pair.second.first].pi, views[pair.second.first].rho, views[pair.second.second].pi,
                                      views[pair.second.second].rho, n_G, d_G)) {
      return true;
    }
  }
  return false;
}

bool LineTriangulation::triangulate_anchored(std::shared_ptr<State> state,
                                             const std::unordered_map<size_t, std::vector<ov_core::LineObservation>> &observations,
                                             int anchor_cam_id, double anchor_clone_timestamp, Eigen::Vector3d &n_A,
                                             Eigen::Vector3d &d_A) {
  Eigen::Vector3d n_G, d_G;
  if (!triangulate(state, observations, n_G, d_G)) {
    return false;
  }
  Eigen::Matrix3d R_A;
  Eigen::Vector3d p_AinG;
  anchor_pose(state, anchor_cam_id, anchor_clone_timestamp, false, R_A, p_AinG);
  // Inverse anchor transform: X_A = R_A^T (X_G - p_AinG).
  d_A = R_A.transpose() * d_G;
  n_A = R_A.transpose() * (n_G - skew_x(p_AinG) * d_G);
  // The moment transform changes ||n||, so re-normalize to the canonical
  // ||n||^2 + ||d||^2 = 1 form used by boxplus / boxplus_jacobian.
  const double scale = std::sqrt(n_A.squaredNorm() + d_A.squaredNorm());
  n_A /= scale;
  d_A /= scale;
  return true;
}

Eigen::Matrix<double, 6, 6> LineTriangulation::transform_jacobian_wrt_pose(const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                                                                           const Eigen::Vector3d &n_src,
                                                                           const Eigen::Vector3d &d_src) {
  const Eigen::Vector3d d_dest = R * d_src;
  Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
  J.block(0, 0, 3, 3) = skew_x(R * n_src) + skew_x(t) * skew_x(d_dest);
  J.block(0, 3, 3, 3) = -skew_x(d_dest);
  J.block(3, 0, 3, 3) = skew_x(d_dest);
  return J;
}

void LineTriangulation::get_jacobian_full(std::shared_ptr<State> state, LineFeatureTri &feat, Eigen::MatrixXd &H_f,
                                          Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                          std::vector<std::shared_ptr<Type>> &x_order) {

  const bool do_fej = state->_options.do_fej;
  const bool anchored = (feat.representation == LineRepresentation::Representation::ANCHORED_PLUCKER);

  // Total number of measurements for this feature.
  int total_meas = 0;
  for (auto const &pair : feat.observations) {
    total_meas += (int)pair.second.size();
  }

  // Compute the size of the states involved with this feature (clones + extrinsics).
  int total_hx = 0;
  std::unordered_map<std::shared_ptr<Type>, size_t> map_hx;
  for (auto const &pair : feat.observations) {
    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    if (state->_options.do_calib_camera_pose) {
      map_hx.insert({calibration, total_hx});
      x_order.push_back(calibration);
      total_hx += calibration->size();
    }
    for (auto const &o : pair.second) {
      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(o.timestamp);
      if (map_hx.find(clone_Ci) == map_hx.end()) {
        map_hx.insert({clone_Ci, total_hx});
        x_order.push_back(clone_Ci);
        total_hx += clone_Ci->size();
      }
    }
  }

  // For anchored lines the anchor clone / extrinsic must also be in the Jacobian.
  std::shared_ptr<PoseJPL> clone_A;
  std::shared_ptr<PoseJPL> calib_A;
  if (anchored) {
    clone_A = state->_clones_IMU.at(feat.anchor_clone_timestamp);
    if (map_hx.find(clone_A) == map_hx.end()) {
      map_hx.insert({clone_A, total_hx});
      x_order.push_back(clone_A);
      total_hx += clone_A->size();
    }
    if (state->_options.do_calib_camera_pose) {
      calib_A = state->_calib_IMUtoCAM.at(feat.anchor_cam_id);
      if (map_hx.find(calib_A) == map_hx.end()) {
        map_hx.insert({calib_A, total_hx});
        x_order.push_back(calib_A);
        total_hx += calib_A->size();
      }
    }
  }

  // Global line at current and FEJ linearization points.
  Eigen::Vector3d n_G, d_G, n_G_fej, d_G_fej;
  global_line(state, feat, false, n_G, d_G);
  global_line(state, feat, true, n_G_fej, d_G_fej);

  // Anchored line + anchor pose at current and FEJ linearization points.
  Eigen::Vector3d n_A = feat.n, d_A = feat.d;
  Eigen::Vector3d n_A_fej = feat.n_fej, d_A_fej = feat.d_fej;
  Eigen::Matrix3d R_A, R_A_fej;
  Eigen::Vector3d p_AinG, p_AinG_fej;
  if (anchored) {
    anchor_pose(state, feat.anchor_cam_id, feat.anchor_clone_timestamp, false, R_A, p_AinG);
    anchor_pose(state, feat.anchor_cam_id, feat.anchor_clone_timestamp, true, R_A_fej, p_AinG_fej);
  }

  // Allocate outputs.
  res = Eigen::VectorXd::Zero(2 * total_meas);
  H_f = Eigen::MatrixXd::Zero(2 * total_meas, 4);
  H_x = Eigen::MatrixXd::Zero(2 * total_meas, total_hx);

  int c = 0;
  for (auto const &pair : feat.observations) {

    std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(pair.first);
    Eigen::Matrix3d R_ItoC = calibration->Rot();
    Eigen::Vector3d p_IinC = calibration->pos();

    for (auto const &o : pair.second) {

      std::shared_ptr<PoseJPL> clone_Ci = state->_clones_IMU.at(o.timestamp);

      // Homogeneous normalized endpoints.
      Eigen::Vector3d P_s(o.endpoints_norm(0), o.endpoints_norm(1), 1.0);
      Eigen::Vector3d P_e(o.endpoints_norm(2), o.endpoints_norm(3), 1.0);

      //====================================================================
      // Residual (always evaluated at the current state and current line).
      //====================================================================
      Eigen::Matrix3d R_GtoC = R_ItoC * clone_Ci->Rot();
      Eigen::Vector3d p_CinG = clone_Ci->pos() - R_GtoC.transpose() * p_IinC;
      Eigen::Vector3d l = line::project_line(R_GtoC, p_CinG, n_G, d_G);
      // z - h with z = 0 and h = line_residual(l).
      res.block(2 * c, 0, 2, 1) = -line::line_residual(l, P_s, P_e);

      //====================================================================
      // Jacobian (evaluated at FEJ clone poses / line when do_fej is set).
      //====================================================================
      Eigen::Matrix3d R_GtoI_j = (do_fej) ? clone_Ci->Rot_fej() : clone_Ci->Rot();
      Eigen::Vector3d p_IinG_j = (do_fej) ? clone_Ci->pos_fej() : clone_Ci->pos();
      Eigen::Matrix3d R_GtoC_j = R_ItoC * R_GtoI_j;
      Eigen::Vector3d p_CinG_j = p_IinG_j - R_GtoC_j.transpose() * p_IinC;
      Eigen::Vector3d n_G_j = (do_fej) ? n_G_fej : n_G;
      Eigen::Vector3d d_G_j = (do_fej) ? d_G_fej : d_G;
      Eigen::Vector3d l_j = line::project_line(R_GtoC_j, p_CinG_j, n_G_j, d_G_j);
      Eigen::Vector3d d_C_j = R_GtoC_j * d_G_j;

      Eigen::Matrix<double, 2, 3> H_line = line::line_residual_jacobian_wrt_l(l_j, P_s, P_e);

      //====================================================================
      // H_f: derivative of the prediction wrt the 4-DOF line error state.
      //====================================================================
      if (!anchored) {
        H_f.block(2 * c, 0, 2, 4) = line::line_jacobian_wrt_global_line(R_GtoC_j, p_CinG_j, n_G_j, d_G_j, P_s, P_e);
      } else {
        Eigen::Matrix3d R_A_j = (do_fej) ? R_A_fej : R_A;
        Eigen::Vector3d p_AinG_j = (do_fej) ? p_AinG_fej : p_AinG;
        Eigen::Vector3d n_A_j = (do_fej) ? n_A_fej : n_A;
        Eigen::Vector3d d_A_j = (do_fej) ? d_A_fej : d_A;
        H_f.block(2 * c, 0, 2, 4) = line::line_jacobian_wrt_anchored_line(R_A_j, p_AinG_j, R_GtoC_j, p_CinG_j, n_A_j, d_A_j, P_s, P_e);
      }

      //====================================================================
      // H_x: derivative wrt the observing clone pose.
      // PoseJPL::update uses R <- Exp(-dtheta) R (JPL convention), so the
      // rotation error enters with the opposite sign from the naive Exp(+dtheta).
      //   dl/ddtheta_I = +[l]x R_ItoC + [d_C]x [p_IinC]x R_ItoC
      //   dl/ddp_I     = R_GtoC [d_G]x
      //====================================================================
      Eigen::Matrix<double, 3, 6> dl_dclone;
      dl_dclone.block(0, 0, 3, 3) = skew_x(l_j) * R_ItoC + skew_x(d_C_j) * skew_x(p_IinC) * R_ItoC;
      dl_dclone.block(0, 3, 3, 3) = R_GtoC_j * skew_x(d_G_j);
      H_x.block(2 * c, map_hx[clone_Ci], 2, clone_Ci->size()).noalias() = H_line * dl_dclone;

      //====================================================================
      // H_x: derivative wrt the observing camera extrinsic.
      //   dl/ddtheta_C = +[l]x + [d_C]x [p_IinC]x
      //   dl/ddp_C     = -[d_C]x
      //====================================================================
      if (state->_options.do_calib_camera_pose) {
        Eigen::Matrix<double, 3, 6> dl_dcalib;
        dl_dcalib.block(0, 0, 3, 3) = skew_x(l_j) + skew_x(d_C_j) * skew_x(p_IinC);
        dl_dcalib.block(0, 3, 3, 3) = -skew_x(d_C_j);
        H_x.block(2 * c, map_hx[calibration], 2, calibration->size()).noalias() = H_line * dl_dcalib;
      }

      //====================================================================
      // H_x: derivative wrt the anchor clone / extrinsic (anchored only).
      // The global line depends on the anchor pose, so chain through
      // dl/d(n_G;d_G) and transform_jacobian_wrt_pose.
      //====================================================================
      if (anchored) {
        Eigen::Matrix3d R_A_j = (do_fej) ? R_A_fej : R_A;
        Eigen::Vector3d p_AinG_j = (do_fej) ? p_AinG_fej : p_AinG;
        Eigen::Vector3d n_A_j = (do_fej) ? n_A_fej : n_A;
        Eigen::Vector3d d_A_j = (do_fej) ? d_A_fej : d_A;

        // dl/d(n_G; d_G) = [R_GtoC, -R_GtoC [p_CinG]x]
        Eigen::Matrix<double, 3, 6> dl_dng;
        dl_dng.block(0, 0, 3, 3) = R_GtoC_j;
        dl_dng.block(0, 3, 3, 3) = -R_GtoC_j * skew_x(p_CinG_j);
        Eigen::Matrix<double, 2, 6> dres_dng = H_line * dl_dng;

        Eigen::Matrix<double, 6, 6> dng_dpose = transform_jacobian_wrt_pose(R_A_j, p_AinG_j, n_A_j, d_A_j);
        Eigen::Vector3d p_IinC_A = calib_A->pos();

        // d[anchor pose error]/d[anchor clone error]
        Eigen::Matrix3d R_GtoI_A = (do_fej) ? clone_A->Rot_fej() : clone_A->Rot();
        Eigen::Matrix<double, 6, 6> dpose_dclone = Eigen::Matrix<double, 6, 6>::Zero();
        dpose_dclone.block(0, 0, 3, 3) = -R_GtoI_A.transpose();
        dpose_dclone.block(3, 0, 3, 3) = skew_x(R_A_j * p_IinC_A) * R_GtoI_A.transpose();
        dpose_dclone.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
        H_x.block(2 * c, map_hx[clone_A], 2, clone_A->size()).noalias() += dres_dng * dng_dpose * dpose_dclone;

        // d[anchor pose error]/d[anchor extrinsic error]
        if (state->_options.do_calib_camera_pose) {
          Eigen::Matrix<double, 6, 6> dpose_dcalib = Eigen::Matrix<double, 6, 6>::Zero();
          dpose_dcalib.block(0, 0, 3, 3) = -R_A_j;
          dpose_dcalib.block(3, 0, 3, 3) = skew_x(R_A_j * p_IinC_A) * R_A_j;
          dpose_dcalib.block(3, 3, 3, 3) = -R_A_j;
          H_x.block(2 * c, map_hx[calib_A], 2, calib_A->size()).noalias() += dres_dng * dng_dpose * dpose_dcalib;
        }
      }

      c++;
    }
  }
}

void LineTriangulation::get_jacobian_representation(std::shared_ptr<State> state, LineFeatureTri &feat, Eigen::MatrixXd &H_f,
                                                    std::vector<Eigen::MatrixXd> &H_x,
                                                    std::vector<std::shared_ptr<Type>> &x_order) {

  const bool do_fej = state->_options.do_fej;
  const bool anchored = (feat.representation == LineRepresentation::Representation::ANCHORED_PLUCKER);

  // Global line (and its FEJ linearization point).
  Eigen::Vector3d n_G, d_G;
  global_line(state, feat, false, n_G, d_G);

  if (!anchored) {
    H_f = line::boxplus_jacobian(n_G, d_G);
    H_x.clear();
    x_order.clear();
    return;
  }

  // Anchor transform: X_G = R_A X_A + p_AinG (R_A = anchor-to-global).
  Eigen::Matrix3d R_A, R_A_fej;
  Eigen::Vector3d p_AinG, p_AinG_fej;
  anchor_pose(state, feat.anchor_cam_id, feat.anchor_clone_timestamp, false, R_A, p_AinG);
  anchor_pose(state, feat.anchor_cam_id, feat.anchor_clone_timestamp, true, R_A_fej, p_AinG_fej);
  const Eigen::Vector3d n_A = feat.n;
  const Eigen::Vector3d d_A = feat.d;
  const Eigen::Vector3d n_A_fej = feat.n_fej;
  const Eigen::Vector3d d_A_fej = feat.d_fej;

  // H_f = d[n_G; d_G] / d[dx_line] = [R_A, [p_AinG]x R_A; 0, R_A] * boxplus_jacobian(n_A, d_A).
  const Eigen::Matrix3d R_A_j = (do_fej) ? R_A_fej : R_A;
  const Eigen::Vector3d p_AinG_j = (do_fej) ? p_AinG_fej : p_AinG;
  const Eigen::Vector3d n_A_j = (do_fej) ? n_A_fej : n_A;
  const Eigen::Vector3d d_A_j = (do_fej) ? d_A_fej : d_A;
  Eigen::Matrix<double, 6, 6> dng_dline;
  dng_dline.block(0, 0, 3, 3) = R_A_j;
  dng_dline.block(0, 3, 3, 3) = skew_x(p_AinG_j) * R_A_j;
  dng_dline.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
  dng_dline.block(3, 3, 3, 3) = R_A_j;
  H_f = dng_dline * line::boxplus_jacobian(n_A_j, d_A_j);

  // d[n_G; d_G] / d[anchor pose error].
  const Eigen::Matrix<double, 6, 6> dng_dpose = transform_jacobian_wrt_pose(R_A_j, p_AinG_j, n_A_j, d_A_j);

  // Anchor clone and extrinsic.
  std::shared_ptr<PoseJPL> clone_A = state->_clones_IMU.at(feat.anchor_clone_timestamp);
  std::shared_ptr<PoseJPL> calib_A = state->_calib_IMUtoCAM.at(feat.anchor_cam_id);
  const Eigen::Vector3d p_IinC_A = calib_A->pos();

  // d[anchor pose error]/d[anchor clone error].
  Eigen::Matrix3d R_GtoI_A = (do_fej) ? clone_A->Rot_fej() : clone_A->Rot();
  Eigen::Matrix<double, 6, 6> dpose_dclone = Eigen::Matrix<double, 6, 6>::Zero();
  dpose_dclone.block(0, 0, 3, 3) = -R_GtoI_A.transpose();
  dpose_dclone.block(3, 0, 3, 3) = skew_x(R_A_j * p_IinC_A) * R_GtoI_A.transpose();
  dpose_dclone.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
  x_order.push_back(clone_A);
  H_x.push_back(dng_dpose * dpose_dclone);

  // d[anchor pose error]/d[anchor extrinsic error].
  if (state->_options.do_calib_camera_pose) {
    Eigen::Matrix<double, 6, 6> dpose_dcalib = Eigen::Matrix<double, 6, 6>::Zero();
    dpose_dcalib.block(0, 0, 3, 3) = -R_A_j;
    dpose_dcalib.block(3, 0, 3, 3) = skew_x(R_A_j * p_IinC_A) * R_A_j;
    dpose_dcalib.block(3, 3, 3, 3) = -R_A_j;
    x_order.push_back(calib_A);
    H_x.push_back(dng_dpose * dpose_dcalib);
  }
}
