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

#include "UpdaterLineSLAM.h"

#include "LineTriangulation.h"
#include "UpdaterHelper.h"

#include "feat/LineFeature.h"
#include "line/LineMath.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkLine.h"
#include "types/LineRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

namespace {

/**
 * @brief Pick the earliest observation across all cameras as the anchor for a line.
 * @return true if at least one observation was found
 */
bool pick_anchor(const std::unordered_map<size_t, std::vector<LineObservation>> &observations, int &anchor_cam_id,
                 double &anchor_timestamp) {
  anchor_cam_id = -1;
  anchor_timestamp = INFINITY;
  for (const auto &pair : observations) {
    for (const auto &o : pair.second) {
      if (o.timestamp < anchor_timestamp) {
        anchor_timestamp = o.timestamp;
        anchor_cam_id = (int)pair.first;
      }
    }
  }
  return anchor_cam_id != -1;
}

/**
 * @brief Check that a 6x4 feature Jacobian has full column rank 4 in the numerical sense.
 */
bool full_rank_4(const Eigen::MatrixXd &H_f) {
  if (H_f.rows() < 4) {
    return false;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H_f, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd sv = svd.singularValues();
  const double s0 = sv(0);
  if (s0 < 1e-12) {
    return false;
  }
  return sv(sv.size() - 1) / s0 > 1e-4;
}

} // namespace

UpdaterLineSLAM::UpdaterLineSLAM(UpdaterOptions &options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Initialize the chi squared test table with confidence level 0.95
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterLineSLAM::delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<LineFeature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  const bool anchored = (state->_options.line_feat_representation == LineRepresentation::Representation::ANCHORED_PLUCKER);
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {
    (*it0)->clean_invalid_timestamps(clonetimes);
    // Persistent SLAM lines need at least three observations to initialize
    if ((*it0)->count_measurements() < 3) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Triangulate each line (globally or in the anchor frame) and reject failures
  std::vector<LineTriangulation::LineFeatureTri> feats;
  feats.reserve(feature_vec.size());
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {
    LineTriangulation::LineFeatureTri feat;
    feat.featid = (*it1)->featid;
    feat.representation = state->_options.line_feat_representation;
    feat.observations = (*it1)->observations;

    bool success = false;
    if (anchored) {
      int anchor_cam_id = -1;
      double anchor_timestamp = INFINITY;
      if (pick_anchor(feat.observations, anchor_cam_id, anchor_timestamp)) {
        feat.anchor_cam_id = anchor_cam_id;
        feat.anchor_clone_timestamp = anchor_timestamp;
        success = LineTriangulation::triangulate_anchored(state, feat.observations, feat.anchor_cam_id,
                                                          feat.anchor_clone_timestamp, feat.n, feat.d);
      }
    } else {
      success = LineTriangulation::triangulate(state, feat.observations, feat.n, feat.d);
    }

    if (!success) {
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    feat.n_fej = feat.n;
    feat.d_fej = feat.d;
    feats.push_back(feat);
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // 3. Compute the linear system and initialize each feature as a persistent line
  for (auto &feat : feats) {

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    LineTriangulation::get_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Reject degenerate lines whose 4-DOF feature Jacobian is rank deficient
    if (!full_rank_4(H_f)) {
      continue;
    }

    // Create the line landmark (4-DOF error state)
    auto landmark = std::make_shared<LandmarkLine>();
    landmark->_featid = feat.featid;
    landmark->_feat_representation = feat.representation;
    landmark->_unique_camera_id = (feat.anchor_cam_id != -1) ? feat.anchor_cam_id : 0;
    if (LineRepresentation::is_relative_representation(feat.representation)) {
      landmark->_anchor_cam_id = feat.anchor_cam_id;
      landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
    }
    landmark->set_from_plucker(feat.n, feat.d, false);
    landmark->set_from_plucker(feat.n_fej, feat.d_fej, true);

    // Measurement noise matrix (isotropic, in normalized coordinate units)
    Eigen::MatrixXd R = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res.rows(), res.rows());

    // Try to initialize, delete new pointer if we failed
    if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, _options.chi2_multipler)) {
      state->_lines_SLAM.insert({feat.featid, landmark});
    }
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Mark all delayed features as consumed
  for (auto &feature : feature_vec) {
    feature->to_delete = true;
  }

  // Debug print timing information
  PRINT_ALL("[LINE-SLAM-DELAY]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-SLAM-DELAY]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-SLAM-DELAY]: %.4f seconds initialize (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)feats.size());
}

void UpdaterLineSLAM::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<LineFeature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {
    (*it0)->clean_invalid_timestamps(clonetimes);
    if ((*it0)->count_measurements() < 1) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else if (state->_lines_SLAM.find((*it0)->featid) == state->_lines_SLAM.end()) {
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (const auto &feature : feature_vec) {
    max_meas_size += 2 * feature->count_measurements();
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size();

  // Large Jacobian, residual, and measurement noise of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size, max_meas_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 2. Compute linear system for each feature and reject outliers
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Ensure we have the landmark and it is the same
    assert(state->_lines_SLAM.find((*it2)->featid) != state->_lines_SLAM.end());
    std::shared_ptr<LandmarkLine> landmark = state->_lines_SLAM.at((*it2)->featid);

    // Convert the state landmark into our current format
    LineTriangulation::LineFeatureTri feat;
    feat.featid = (*it2)->featid;
    feat.representation = landmark->_feat_representation;
    feat.anchor_cam_id = landmark->_anchor_cam_id;
    feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
    feat.observations = (*it2)->observations;
    feat.n = landmark->n(false);
    feat.d = landmark->d(false);
    feat.n_fej = landmark->n(true);
    feat.d_fej = landmark->d(true);

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    LineTriangulation::get_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
    Eigen::MatrixXd H_xf = H_x;
    H_xf.conservativeResize(H_x.rows(), H_x.cols() + H_f.cols());
    H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;

    // Append to our Jacobian order vector
    std::vector<std::shared_ptr<Type>> Hxf_order = Hx_order;
    Hxf_order.push_back(landmark);

    // Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
    Eigen::MatrixXd S = H_xf * P_marg * H_xf.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      landmark->update_fail_count++;
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hxf_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_xf.rows(), var->size()) = H_xf.block(0, ct_hx, H_xf.rows(), var->size());
      ct_hx += var->size();
    }

    // Our isotropic measurement noise
    R_big.block(ct_meas, ct_meas, res.rows(), res.rows()) *= _options.sigma_pix_sq;

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (auto &feature : feature_vec) {
    feature->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);
  R_big.conservativeResize(ct_meas, ct_meas);

  // 3. With all good SLAM features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[LINE-SLAM-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-SLAM-UP]: %.4f seconds creating linear system\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)feature_vec.size(), (int)Hx_big.rows());
}

void UpdaterLineSLAM::change_anchors(std::shared_ptr<State> state) {

  // Return if we do not have enough clones
  if ((int)state->_clones_IMU.size() <= state->_options.max_clone_size) {
    return;
  }

  // Get the marginalization timestep, and change the anchor for any line seen from it
  double marg_timestep = state->margtimestep();
  for (auto &f : state->_lines_SLAM) {
    // Skip any lines that are in the global frame
    if (f.second->_feat_representation != LineRepresentation::Representation::ANCHORED_PLUCKER)
      continue;
    // Else see if it is anchored in the clone that will be marginalized
    assert(marg_timestep <= f.second->_anchor_clone_timestamp);
    if (f.second->_anchor_clone_timestamp == marg_timestep) {
      perform_anchor_change(state, f.second, state->_timestamp, f.second->_anchor_cam_id);
    }
  }
}

void UpdaterLineSLAM::perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<LandmarkLine> landmark,
                                            double new_anchor_timestamp, size_t new_cam_id) {

  // Assert that this is an anchored representation
  assert(LineRepresentation::is_relative_representation(landmark->_feat_representation));
  assert(landmark->_anchor_cam_id != -1);

  // Helper to compute an anchor camera pose (R_A = anchor-to-global, p_AinG).
  auto anchor_pose = [&](int cam_id, double ts, bool use_fej, Eigen::Matrix3d &R_A, Eigen::Vector3d &p_AinG) {
    auto calib = state->_calib_IMUtoCAM.at(cam_id);
    Eigen::Matrix3d R_ItoC = calib->Rot();
    Eigen::Vector3d p_IinC = calib->pos();
    auto clone = state->_clones_IMU.at(ts);
    Eigen::Matrix3d R_GtoI = use_fej ? clone->Rot_fej() : clone->Rot();
    Eigen::Vector3d p_IinG = use_fej ? clone->pos_fej() : clone->pos();
    R_A = R_GtoI.transpose() * R_ItoC.transpose();
    p_AinG = p_IinG - R_A * p_IinC;
  };

  // Helper to express a global line in an anchor frame (inverse anchor transform).
  auto transform_to_anchor = [&](const Eigen::Matrix3d &R_A, const Eigen::Vector3d &p_AinG, const Eigen::Vector3d &n_G,
                                 const Eigen::Vector3d &d_G, Eigen::Vector3d &n_A, Eigen::Vector3d &d_A) {
    d_A = R_A.transpose() * d_G;
    n_A = R_A.transpose() * (n_G - skew_x(p_AinG) * d_G);
    const double scale = std::sqrt(n_A.squaredNorm() + d_A.squaredNorm());
    n_A /= scale;
    d_A /= scale;
  };

  //==========================================================================
  // Current (old) representation and its Jacobians
  //==========================================================================
  LineTriangulation::LineFeatureTri old_feat;
  old_feat.featid = landmark->_featid;
  old_feat.representation = landmark->_feat_representation;
  old_feat.anchor_cam_id = landmark->_anchor_cam_id;
  old_feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
  old_feat.n = landmark->n(false);
  old_feat.d = landmark->d(false);
  old_feat.n_fej = landmark->n(true);
  old_feat.d_fej = landmark->d(true);

  Eigen::MatrixXd H_f_old;
  std::vector<Eigen::MatrixXd> H_x_old;
  std::vector<std::shared_ptr<Type>> x_order_old;
  LineTriangulation::get_jacobian_representation(state, old_feat, H_f_old, H_x_old, x_order_old);

  //==========================================================================
  // Future (new) representation: shift anchor and transform the line
  //==========================================================================
  Eigen::Matrix3d R_A_old, R_A_new, R_A_old_fej, R_A_new_fej;
  Eigen::Vector3d p_Aold_inG, p_Anew_inG, p_Aold_inG_fej, p_Anew_inG_fej;
  anchor_pose(old_feat.anchor_cam_id, old_feat.anchor_clone_timestamp, false, R_A_old, p_Aold_inG);
  anchor_pose((int)new_cam_id, new_anchor_timestamp, false, R_A_new, p_Anew_inG);
  anchor_pose(old_feat.anchor_cam_id, old_feat.anchor_clone_timestamp, true, R_A_old_fej, p_Aold_inG_fej);
  anchor_pose((int)new_cam_id, new_anchor_timestamp, true, R_A_new_fej, p_Anew_inG_fej);

  // Global line from the old anchor (current + FEJ)
  Eigen::Vector3d n_G, d_G, n_G_fej, d_G_fej;
  line::transform_plucker(R_A_old, p_Aold_inG, old_feat.n, old_feat.d, n_G, d_G);
  line::transform_plucker(R_A_old_fej, p_Aold_inG_fej, old_feat.n_fej, old_feat.d_fej, n_G_fej, d_G_fej);

  // New anchored line
  Eigen::Vector3d n_A_new, d_A_new, n_A_new_fej, d_A_new_fej;
  transform_to_anchor(R_A_new, p_Anew_inG, n_G, d_G, n_A_new, d_A_new);
  transform_to_anchor(R_A_new_fej, p_Anew_inG_fej, n_G_fej, d_G_fej, n_A_new_fej, d_A_new_fej);

  LineTriangulation::LineFeatureTri new_feat;
  new_feat.featid = landmark->_featid;
  new_feat.representation = landmark->_feat_representation;
  new_feat.anchor_cam_id = (int)new_cam_id;
  new_feat.anchor_clone_timestamp = new_anchor_timestamp;
  new_feat.n = n_A_new;
  new_feat.d = d_A_new;
  new_feat.n_fej = n_A_new_fej;
  new_feat.d_fej = d_A_new_fej;

  Eigen::MatrixXd H_f_new;
  std::vector<Eigen::MatrixXd> H_x_new;
  std::vector<std::shared_ptr<Type>> x_order_new;
  LineTriangulation::get_jacobian_representation(state, new_feat, H_f_new, H_x_new, x_order_new);

  //==========================================================================
  // Build the anchor-change state transition Jacobian
  //==========================================================================

  // New phi order is just the landmark
  std::vector<std::shared_ptr<Type>> phi_order_NEW;
  phi_order_NEW.push_back(landmark);

  // Loop through all our orders and append them (deduplicated)
  std::vector<std::shared_ptr<Type>> phi_order_OLD;
  int current_it = 0;
  std::map<std::shared_ptr<Type>, int> Phi_id_map;
  for (const auto &var : x_order_old) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  for (const auto &var : x_order_new) {
    if (Phi_id_map.find(var) == Phi_id_map.end()) {
      Phi_id_map.insert({var, current_it});
      phi_order_OLD.push_back(var);
      current_it += var->size();
    }
  }
  Phi_id_map.insert({landmark, current_it});
  phi_order_OLD.push_back(landmark);
  current_it += landmark->size();

  // Anchor change Jacobian (4-DOF line error state)
  const int phisize = 4;
  Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(phisize, current_it);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(phisize, phisize);

  // Left inverse of the new representation Jacobian (H_f_new is 6x4)
  Eigen::MatrixXd H_f_new_inv = H_f_new.colPivHouseholderQr().solve(Eigen::Matrix<double, 6, 6>::Identity());

  // pf_new_error = Hfnew^{-1} * (Hfold * pf_old_error + Hxold * x_old_error - Hxnew * x_new_error)
  for (size_t i = 0; i < H_x_old.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_old[i]), phisize, x_order_old[i]->size()).noalias() += H_f_new_inv * H_x_old[i];
  }
  Phi.block(0, Phi_id_map.at(landmark), phisize, phisize) = H_f_new_inv * H_f_old;
  for (size_t i = 0; i < H_x_new.size(); i++) {
    Phi.block(0, Phi_id_map.at(x_order_new[i]), phisize, x_order_new[i]->size()).noalias() -= H_f_new_inv * H_x_new[i];
  }

  // Perform covariance propagation
  StateHelper::EKFPropagation(state, phi_order_NEW, phi_order_OLD, Phi, Q);

  // Set state from new feature
  landmark->_anchor_cam_id = (int)new_cam_id;
  landmark->_anchor_clone_timestamp = new_anchor_timestamp;
  landmark->set_from_plucker(n_A_new, d_A_new, false);
  landmark->set_from_plucker(n_A_new_fej, d_A_new_fej, true);
  landmark->has_had_anchor_change = true;
}
