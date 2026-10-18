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

#ifndef OV_MSCKF_UPDATER_LINE_SLAM_H
#define OV_MSCKF_UPDATER_LINE_SLAM_H

#include <Eigen/Eigen>
#include <map>
#include <memory>

#include "UpdaterOptions.h"

namespace ov_core {
class LineFeature;
} // namespace ov_core
namespace ov_type {
class LandmarkLine;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our sparse SLAM line features and update the filter.
 *
 * This mirrors UpdaterSLAM but operates on 4-DOF Plücker lines. Lines are initialized
 * from three or more triangulated observations and stored as persistent LandmarkLine
 * state variables (GLOBAL_PLUCKER or ANCHORED_PLUCKER).
 */
class UpdaterLineSLAM {

public:
  /**
   * @brief Default constructor for our SLAM line updater
   * @param options Updater options (include measurement noise value) for line features
   */
  UpdaterLineSLAM(UpdaterOptions &options);

  /**
   * @brief Given tracked line features, this will try to use them to update the state.
   * @param state State of the filter
   * @param feature_vec Line features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::LineFeature>> &feature_vec);

  /**
   * @brief Given tracked line features, try to initialize them as persistent SLAM lines.
   * @param state State of the filter
   * @param feature_vec Line features that can be used for initialization
   */
  void delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::LineFeature>> &feature_vec);

  /**
   * @brief Will change SLAM line anchors if the anchor clone will be marginalized.
   *
   * Only applies to ANCHORED_PLUCKER lines. Shifts the anchor to the newest clone while
   * keeping the camera calibration anchor the same.
   *
   * @param state State of the filter
   */
  void change_anchors(std::shared_ptr<State> state);

protected:
  /**
   * @brief Shifts a line landmark's anchor to a new clone
   * @param state State of filter
   * @param landmark Line landmark whose anchor is being shifted
   * @param new_anchor_timestamp Clone timestamp we want to move to
   * @param new_cam_id Which camera frame we want to move to
   */
  void perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<ov_type::LandmarkLine> landmark,
                             double new_anchor_timestamp, size_t new_cam_id);

  /// Options used during update for line features
  UpdaterOptions _options;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_LINE_SLAM_H
