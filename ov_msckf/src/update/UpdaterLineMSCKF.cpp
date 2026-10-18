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

#include "UpdaterLineMSCKF.h"

#include "LineTriangulation.h"
#include "UpdaterHelper.h"

#include "feat/LineFeature.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LineRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"

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
 * @brief Check that H_f (2k x 4) has full column rank 4 in the numerical sense.
 *
 * Uses the ratio of the smallest to largest singular value. A line is unobservable if
 * all observations come from (nearly) collinear viewpoints.
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

UpdaterLineMSCKF::UpdaterLineMSCKF(UpdaterOptions &options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Initialize the chi squared test table with confidence level 0.95
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterLineMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<LineFeature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
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
    if ((*it0)->count_measurements() < 2) {
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Triangulate each line (globally or in the anchor frame) and reject failures
  const bool anchored = (state->_options.line_feat_representation == LineRepresentation::Representation::ANCHORED_PLUCKER);
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

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (const auto &feat : feats) {
    for (const auto &pair : feat.observations) {
      max_meas_size += 2 * pair.second.size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }
  for (auto &landmark : state->_lines_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // 3. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feats.begin();
  while (it2 != feats.end()) {

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    LineTriangulation::get_jacobian_full(state, *it2, H_f, H_x, res, Hx_order);

    // Observability guard: a MSCKF line needs at least 3 observations (2k > 4 residual
    // rows) and a full-rank 4-DOF feature Jacobian before nullspace projection.
    if (H_f.rows() <= 4 || !full_rank_4(H_f)) {
      it2++;
      continue;
    }

    // Nullspace project the 4-DOF feature state out of the residual
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
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
      it2++;
      continue;
    }

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // 4. Perform measurement compression
  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  if (Hx_big.rows() < 1) {
    return;
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // 5. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6,
            (int)feats.size());
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[LINE-MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
