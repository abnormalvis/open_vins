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

#include "LandmarkLine.h"

#include <cstdlib>

#include "line/LineMath.h"

using namespace ov_type;

void LandmarkLine::update(const Eigen::VectorXd &dx) {
  assert(dx.rows() == _size);

  const Eigen::Vector3d n_cur = value().block(0, 0, 3, 1);
  const Eigen::Vector3d d_cur = value().block(3, 0, 3, 1);

  Eigen::Vector3d n_new;
  Eigen::Vector3d d_new;
  ov_core::line::boxplus(n_cur, d_cur, dx, n_new, d_new);

  Eigen::Matrix<double, 6, 1> new_value;
  new_value << n_new, d_new;
  set_value(new_value);
}

std::shared_ptr<Type> LandmarkLine::clone() {
  auto Clone = std::shared_ptr<LandmarkLine>(new LandmarkLine());
  Clone->set_value(value());
  Clone->set_fej(fej());
  Clone->_featid = _featid;
  Clone->_unique_camera_id = _unique_camera_id;
  Clone->_anchor_cam_id = _anchor_cam_id;
  Clone->_anchor_clone_timestamp = _anchor_clone_timestamp;
  Clone->has_had_anchor_change = has_had_anchor_change;
  Clone->should_marg = should_marg;
  Clone->update_fail_count = update_fail_count;
  Clone->_feat_representation = _feat_representation;
  return Clone;
}

Eigen::Matrix<double, 6, 1> LandmarkLine::get_plucker(bool getfej) const {
  return (getfej) ? fej() : value();
}

Eigen::Vector3d LandmarkLine::n(bool getfej) const {
  return (getfej) ? fej().block(0, 0, 3, 1) : value().block(0, 0, 3, 1);
}

Eigen::Vector3d LandmarkLine::d(bool getfej) const {
  return (getfej) ? fej().block(3, 0, 3, 1) : value().block(3, 0, 3, 1);
}

void LandmarkLine::set_from_plucker(const Eigen::Vector3d &n, const Eigen::Vector3d &d, bool isfej) {

  // Orthogonalize the moment against the direction, then normalize both by a common
  // scale so that ||n||^2 + ||d||^2 = 1 while preserving the moment/direction ratio.
  const double d_sq = d.squaredNorm();
  if (d_sq < 1e-12) {
    PRINT_ERROR("LandmarkLine::set_from_plucker called with a zero direction vector!\n");
    std::exit(EXIT_FAILURE);
  }
  const Eigen::Vector3d n_ortho = n - (n.dot(d) / d_sq) * d;
  const double scale = std::sqrt(n_ortho.squaredNorm() + d_sq);

  Eigen::Matrix<double, 6, 1> plucker;
  plucker << n_ortho / scale, d / scale;

  if (isfej)
    set_fej(plucker);
  else
    set_value(plucker);
}
