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

#ifndef OV_TYPE_TYPE_LANDMARK_LINE_H
#define OV_TYPE_TYPE_LANDMARK_LINE_H

#include "LineRepresentation.h"
#include "Type.h"
#include "utils/print.h"

namespace ov_type {

/**
 * @brief Type that implements a persistent SLAM line feature.
 *
 * The line is stored internally as a 6x1 normalized Plücker vector (n; d) with n^T d = 0 and
 * ||n||^2 + ||d||^2 = 1. The error state has 4 DOF and the update implements the orthonormal
 * boxplus (U <- U Exp(dtheta), W <- W R(dphi)), not additive vector addition.
 */
class LandmarkLine : public Type {

public:
  /// Default constructor (line error state has 4 DOF)
  LandmarkLine() : Type(4) {
    _value = Eigen::Matrix<double, 6, 1>::Zero();
    _fej = Eigen::Matrix<double, 6, 1>::Zero();
  }

  /// Feature ID of this landmark (corresponds to frontend id)
  size_t _featid;

  /// What unique camera stream this slam feature was observed from
  int _unique_camera_id = -1;

  /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.
  int _anchor_cam_id = -1;

  /// Timestamp of anchor clone
  double _anchor_clone_timestamp = -1;

  /// Boolean if this landmark has had at least one anchor change
  bool has_had_anchor_change = false;

  /// Boolean if this landmark should be marginalized out
  bool should_marg = false;

  /// Number of times the update has failed for this feature (we should remove if it fails a couple times!)
  int update_fail_count = 0;

  /// What feature representation this feature currently has
  LineRepresentation::Representation _feat_representation;

  /**
   * @brief Overrides the default update rule to apply the orthonormal boxplus.
   * @param dx 4x1 error state correction [dtheta; dphi]
   */
  void update(const Eigen::VectorXd &dx) override;

  /**
   * @brief Performs all the cloning
   */
  std::shared_ptr<Type> clone() override;

  /**
   * @brief Return the stored normalized Plücker line (n; d).
   * @param getfej Set to true to get the landmark FEJ value
   * @return 6x1 Plücker vector
   */
  Eigen::Matrix<double, 6, 1> get_plucker(bool getfej) const;

  /**
   * @brief Return the stored Plücker moment n.
   * @param getfej Set to true to get the landmark FEJ value
   */
  Eigen::Vector3d n(bool getfej = false) const;

  /**
   * @brief Return the stored Plücker direction d.
   * @param getfej Set to true to get the landmark FEJ value
   */
  Eigen::Vector3d d(bool getfej = false) const;

  /**
   * @brief Set the current value from a Plücker line.
   *
   * The input is orthogonalized (n <- n - (n^T d) d / ||d||^2) and normalized so that
   * ||n||^2 + ||d||^2 = 1.
   *
   * @param n Plücker moment (any scale)
   * @param d Plücker direction (non-zero)
   * @param isfej Set to true to set the landmark FEJ value
   */
  void set_from_plucker(const Eigen::Vector3d &n, const Eigen::Vector3d &d, bool isfej);
};

} // namespace ov_type

#endif // OV_TYPE_TYPE_LANDMARK_LINE_H
