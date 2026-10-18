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

#ifndef OV_MSCKF_UPDATER_LINE_MSCKF_H
#define OV_MSCKF_UPDATER_LINE_MSCKF_H

#include <Eigen/Eigen>
#include <map>
#include <memory>

#include "UpdaterOptions.h"

namespace ov_core {
class LineFeature;
} // namespace ov_core

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our MSCKF line features and update the filter.
 *
 * This mirrors UpdaterMSCKF but operates on 4-DOF Plücker lines. Each line is
 * triangulated, its 4-DOF feature Jacobian is nullspace projected (so only the clone
 * / extrinsic information remains), and the projected measurements are compressed and
 * applied as a single EKF update. Degenerate lines (fewer than 3 observations or a
 * rank-deficient feature Jacobian) are rejected before projection.
 */
class UpdaterLineMSCKF {

public:
  /**
   * @brief Default constructor for our MSCKF line updater
   * @param options Updater options (include measurement noise value)
   */
  UpdaterLineMSCKF(UpdaterOptions &options);

  /**
   * @brief Given tracked line features, this will try to use them to update the state.
   * @param state State of the filter
   * @param feature_vec Line features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::LineFeature>> &feature_vec);

protected:
  /// Options used during update
  UpdaterOptions _options;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_LINE_MSCKF_H
