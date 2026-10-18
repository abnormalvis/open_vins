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

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "line/LineMath.h"
#include "state/State.h"
#include "state/StateOptions.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "update/LineTriangulation.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

// Deterministic RNG so failures are reproducible.
static std::mt19937 g_rng(20261017);

static double rand_uniform(double a, double b) {
  std::uniform_real_distribution<double> dist(a, b);
  return dist(g_rng);
}

static Eigen::Vector3d rand_vec3(double scale = 1.0) {
  return Eigen::Vector3d(rand_uniform(-scale, scale), rand_uniform(-scale, scale), rand_uniform(-scale, scale));
}

static Eigen::Matrix3d rand_rotation() { return ov_core::exp_so3(rand_vec3(1.0)); }

static void set_pose(std::shared_ptr<PoseJPL> pose, const Eigen::Matrix3d &R, const Eigen::Vector3d &p) {
  Eigen::Matrix<double, 7, 1> val;
  val << ov_core::rot_2_quat(R), p;
  pose->set_value(val);
  pose->set_fej(val);
}

static std::shared_ptr<State> make_state(bool do_fej, int num_cameras = 2) {
  StateOptions options;
  options.num_cameras = num_cameras;
  options.do_calib_camera_pose = true;
  options.do_calib_camera_intrinsics = false;
  options.do_calib_camera_timeoffset = false;
  options.do_calib_imu_intrinsics = false;
  options.do_calib_imu_g_sensitivity = false;
  options.do_fej = do_fej;
  auto state = std::make_shared<State>(options);
  for (int i = 0; i < num_cameras; i++) {
    set_pose(state->_calib_IMUtoCAM.at(i), (i == 0) ? Eigen::Matrix3d::Identity() : ov_core::exp_so3(Eigen::Vector3d(0.0, 0.03, 0.0)),
             (i == 0) ? Eigen::Vector3d::Zero() : Eigen::Vector3d(0.06, 0.0, 0.0));
  }
  return state;
}

static void add_clone(std::shared_ptr<State> state, double t, const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG) {
  auto clone = std::make_shared<PoseJPL>();
  set_pose(clone, R_GtoI, p_IinG);
  state->_clones_IMU.insert({t, clone});
}

// Random forward-facing line (closest point in front of the cameras).
static void make_gt_line(Eigen::Vector3d &n, Eigen::Vector3d &d) {
  d = Eigen::Vector3d(rand_uniform(-0.3, 0.3), rand_uniform(-0.3, 0.3), 1.0);
  d.normalize();
  Eigen::Vector3d p0(rand_uniform(-0.5, 0.5), rand_uniform(-0.5, 0.5), rand_uniform(2.0, 4.0));
  n = p0.cross(d);
  double scale = std::sqrt(n.squaredNorm() + d.squaredNorm());
  n /= scale;
  d /= scale;
}

// Populate feat.observations by projecting the ground-truth line into each view.
static void make_observations(std::shared_ptr<State> state, const Eigen::Vector3d &n_gt, const Eigen::Vector3d &d_gt,
                              const std::vector<std::pair<size_t, double>> &views,
                              LineTriangulation::LineFeatureTri &feat, double endpoint_half_len = 0.5) {
  for (auto const &v : views) {
    size_t cam_id = v.first;
    double t = v.second;
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(t)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(t)->pos();
    Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoI;
    Eigen::Vector3d p_CinG = p_IinG - R_GtoC.transpose() * p_IinC;
    Eigen::Vector3d l = line::project_line(R_GtoC, p_CinG, n_gt, d_gt);

    double a = l(0), b = l(1), cc = l(2);
    double denom = a * a + b * b;
    if (denom < 1e-8) {
      continue; // degenerate projection (should not happen for our scene)
    }
    Eigen::Vector3d p0(-cc * a / denom, -cc * b / denom, 1.0);
    Eigen::Vector3d tdir(b, -a, 0.0);
    tdir.normalize();
    Eigen::Vector3d P_s = p0 - endpoint_half_len * tdir;
    Eigen::Vector3d P_e = p0 + endpoint_half_len * tdir;

    ov_core::LineObservation o;
    o.timestamp = t;
    o.camera_id = cam_id;
    o.endpoints_norm << P_s(0), P_s(1), P_e(0), P_e(1);
    feat.observations[cam_id].push_back(o);
  }
}

// Inverse anchor transform: express a global line in the anchor camera frame.
static void global_to_anchored(std::shared_ptr<State> state, int anchor_cam_id, double anchor_timestamp,
                               const Eigen::Vector3d &n_G, const Eigen::Vector3d &d_G, Eigen::Vector3d &n_A,
                               Eigen::Vector3d &d_A) {
  Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(anchor_cam_id)->Rot();
  Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(anchor_cam_id)->pos();
  Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(anchor_timestamp)->Rot();
  Eigen::Vector3d p_IinG = state->_clones_IMU.at(anchor_timestamp)->pos();
  Eigen::Matrix3d R_A = R_GtoI.transpose() * R_ItoC.transpose();
  Eigen::Vector3d p_AinG = p_IinG - R_A * p_IinC;
  d_A = R_A.transpose() * d_G;
  n_A = R_A.transpose() * (n_G - skew_x(p_AinG) * d_G);
  const double scale = std::sqrt(n_A.squaredNorm() + d_A.squaredNorm());
  n_A /= scale;
  d_A /= scale;
}

static Eigen::VectorXd compute_res(std::shared_ptr<State> state, LineTriangulation::LineFeatureTri &feat) {
  Eigen::MatrixXd H_f, H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<Type>> x_order;
  LineTriangulation::get_jacobian_full(state, feat, H_f, H_x, res, x_order);
  return res;
}

// Verify H_f and H_x by central finite differences.
// Assumes (for the FEJ path) that the first-estimate and current values coincide.
static void check_jacobians(std::shared_ptr<State> state, LineTriangulation::LineFeatureTri &feat) {
  const double eps = 1e-6;
  const double tol = 1e-5;

  Eigen::MatrixXd H_f, H_x;
  Eigen::VectorXd res;
  std::vector<std::shared_ptr<Type>> x_order;
  LineTriangulation::get_jacobian_full(state, feat, H_f, H_x, res, x_order);

  // H_f: perturb the line through its 4-DOF boxplus.
  for (int j = 0; j < 4; j++) {
    Eigen::Vector4d e = Eigen::Vector4d::Zero();
    e(j) = eps;
    Eigen::Vector3d np, dp, nm, dm;
    line::boxplus(feat.n, feat.d, e, np, dp);
    line::boxplus(feat.n, feat.d, -e, nm, dm);
    LineTriangulation::LineFeatureTri fp = feat;
    fp.n = np;
    fp.d = dp;
    fp.n_fej = np;
    fp.d_fej = dp;
    LineTriangulation::LineFeatureTri fm = feat;
    fm.n = nm;
    fm.d = dm;
    fm.n_fej = nm;
    fm.d_fej = dm;
    Eigen::VectorXd num = (compute_res(state, fp) - compute_res(state, fm)) / (2 * eps);
    // H_f = d h / d x = - d res / d x
    assert((H_f.col(j) + num).norm() < tol);
  }

  // H_x: perturb each state variable through its own (Lie-group) update.
  int col = 0;
  for (size_t vi = 0; vi < x_order.size(); vi++) {
    auto &var = x_order[vi];
    int size = var->size();
    for (int j = 0; j < size; j++) {
      Eigen::VectorXd e = Eigen::VectorXd::Zero(size);
      e(j) = eps;
      var->update(e);
      Eigen::VectorXd rp = compute_res(state, feat);
      var->update(-2.0 * e);
      Eigen::VectorXd rm = compute_res(state, feat);
      var->update(e); // exact restore (collinear rotations / additive translation)
      Eigen::VectorXd num = (rp - rm) / (2 * eps);
      double err = (H_x.col(col + j) + num).norm();
      if (err >= tol) {
        std::cerr << "H_x mismatch: var idx " << vi << " col " << j << " err " << err << std::endl;
        std::cerr << "  H_x col: " << H_x.col(col + j).transpose() << std::endl;
        std::cerr << "  -num   : " << (-num).transpose() << std::endl;
      }
      assert(err < tol);
    }
    col += size;
  }
}

static void test_triangulation() {
  for (int trial = 0; trial < 200; ++trial) {
    auto state = make_state(false);

    // Camera trajectory: forward translation with small jitter.
    add_clone(state, 0.0, ov_core::exp_so3(Eigen::Vector3d(0.02, 0.01, 0.0)), Eigen::Vector3d(0.0, 0.0, 0.0));
    add_clone(state, 0.1, ov_core::exp_so3(Eigen::Vector3d(0.01, 0.0, 0.0)), Eigen::Vector3d(0.3, 0.0, 0.0));
    add_clone(state, 0.2, ov_core::exp_so3(Eigen::Vector3d(0.0, 0.01, 0.0)), Eigen::Vector3d(0.6, 0.02, 0.0));
    add_clone(state, 0.3, ov_core::exp_so3(Eigen::Vector3d(0.0, -0.01, 0.0)), Eigen::Vector3d(0.9, -0.01, 0.0));

    Eigen::Vector3d n_gt, d_gt;
    make_gt_line(n_gt, d_gt);

    std::vector<std::pair<size_t, double>> views;
    for (double t : {0.0, 0.1, 0.2, 0.3}) {
      views.push_back({0, t});
      views.push_back({1, t});
    }
    LineTriangulation::LineFeatureTri feat;
    make_observations(state, n_gt, d_gt, views, feat);

    // Global triangulation.
    Eigen::Vector3d n, d;
    assert(LineTriangulation::triangulate(state, feat.observations, n, d));
    assert(std::abs(n.dot(d)) < 1e-8);
    assert(std::abs(n.squaredNorm() + d.squaredNorm() - 1.0) < 1e-8);
    assert(std::abs(std::abs(d.normalized().dot(d_gt.normalized())) - 1.0) < 1e-6);
    assert(std::abs(std::abs(n.normalized().dot(n_gt.normalized())) - 1.0) < 1e-6);

    // Anchored triangulation (anchor = first view) must map back to the global line.
    Eigen::Vector3d n_A, d_A;
    assert(LineTriangulation::triangulate_anchored(state, feat.observations, 0, 0.0, n_A, d_A));
    Eigen::Vector3d n_back, d_back;
    line::transform_plucker(state->_clones_IMU.at(0.0)->Rot().transpose() * state->_calib_IMUtoCAM.at(0)->Rot().transpose(),
                            state->_clones_IMU.at(0.0)->pos() -
                                state->_clones_IMU.at(0.0)->Rot().transpose() * state->_calib_IMUtoCAM.at(0)->Rot().transpose() *
                                    state->_calib_IMUtoCAM.at(0)->pos(),
                            n_A, d_A, n_back, d_back);
    assert(std::abs(std::abs(d_back.normalized().dot(d_gt.normalized())) - 1.0) < 1e-6);
    assert(std::abs(std::abs(n_back.normalized().dot(n_gt.normalized())) - 1.0) < 1e-6);
  }
}

static void test_Hx_global() {
  for (int trial = 0; trial < 20; ++trial) {
    auto state = make_state(false);
    add_clone(state, 0.0, rand_rotation(), rand_vec3(0.2));
    add_clone(state, 0.1, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.4, 0.0, 0.0));
    add_clone(state, 0.2, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.8, 0.0, 0.0));

    Eigen::Vector3d n_gt, d_gt;
    make_gt_line(n_gt, d_gt);

    std::vector<std::pair<size_t, double>> views;
    for (double t : {0.0, 0.1, 0.2}) {
      views.push_back({0, t});
      views.push_back({1, t});
    }
    LineTriangulation::LineFeatureTri feat;
    make_observations(state, n_gt, d_gt, views, feat);
    feat.representation = LineRepresentation::Representation::GLOBAL_PLUCKER;
    feat.n = n_gt;
    feat.d = d_gt;
    feat.n_fej = n_gt;
    feat.d_fej = d_gt;
    check_jacobians(state, feat);
  }
}

static void test_Hx_anchored() {
  for (int trial = 0; trial < 20; ++trial) {
    auto state = make_state(false);
    add_clone(state, 0.0, rand_rotation(), rand_vec3(0.2));
    add_clone(state, 0.1, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.4, 0.0, 0.0));
    add_clone(state, 0.2, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.8, 0.0, 0.0));

    Eigen::Vector3d n_gt, d_gt;
    make_gt_line(n_gt, d_gt);

    std::vector<std::pair<size_t, double>> views;
    for (double t : {0.0, 0.1, 0.2}) {
      views.push_back({0, t});
      views.push_back({1, t});
    }
    LineTriangulation::LineFeatureTri feat;
    make_observations(state, n_gt, d_gt, views, feat);
    feat.representation = LineRepresentation::Representation::ANCHORED_PLUCKER;
    feat.anchor_cam_id = 0;
    feat.anchor_clone_timestamp = 0.0;
    global_to_anchored(state, 0, 0.0, n_gt, d_gt, feat.n, feat.d);
    feat.n_fej = feat.n;
    feat.d_fej = feat.d;
    check_jacobians(state, feat);
  }
}

static void test_Hx_fej() {
  for (int trial = 0; trial < 10; ++trial) {
    auto state = make_state(true);
    add_clone(state, 0.0, rand_rotation(), rand_vec3(0.2));
    add_clone(state, 0.1, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.4, 0.0, 0.0));
    add_clone(state, 0.2, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.8, 0.0, 0.0));

    Eigen::Vector3d n_gt, d_gt;
    make_gt_line(n_gt, d_gt);

    std::vector<std::pair<size_t, double>> views;
    for (double t : {0.0, 0.1, 0.2}) {
      views.push_back({0, t});
      views.push_back({1, t});
    }
    LineTriangulation::LineFeatureTri feat;
    make_observations(state, n_gt, d_gt, views, feat);

    // Global (FEJ).
    feat.representation = LineRepresentation::Representation::GLOBAL_PLUCKER;
    feat.n = feat.n_fej = n_gt;
    feat.d = feat.d_fej = d_gt;
    check_jacobians(state, feat);

    // Anchored (FEJ).
    feat.representation = LineRepresentation::Representation::ANCHORED_PLUCKER;
    feat.anchor_cam_id = 0;
    feat.anchor_clone_timestamp = 0.0;
    global_to_anchored(state, 0, 0.0, n_gt, d_gt, feat.n, feat.d);
    feat.n_fej = feat.n;
    feat.d_fej = feat.d;
    check_jacobians(state, feat);
  }
}

// Confirm that when do_fej is on, the Jacobian is pinned to the first estimate: changing
// the current clone value changes the residual but must NOT change H_x.
static void test_fej_actually_used() {
  auto state = make_state(true);
  add_clone(state, 0.0, rand_rotation(), rand_vec3(0.2));
  add_clone(state, 0.1, rand_rotation(), rand_vec3(0.2) + Eigen::Vector3d(0.4, 0.0, 0.0));

  Eigen::Vector3d n_gt, d_gt;
  make_gt_line(n_gt, d_gt);

  std::vector<std::pair<size_t, double>> views;
  for (double t : {0.0, 0.1}) {
    views.push_back({0, t});
    views.push_back({1, t});
  }
  LineTriangulation::LineFeatureTri feat;
  make_observations(state, n_gt, d_gt, views, feat);
  feat.representation = LineRepresentation::Representation::GLOBAL_PLUCKER;
  feat.n = feat.n_fej = n_gt;
  feat.d = feat.d_fej = d_gt;

  Eigen::MatrixXd H_f0, H_x0;
  Eigen::VectorXd res0;
  std::vector<std::shared_ptr<Type>> x_order0;
  LineTriangulation::get_jacobian_full(state, feat, H_f0, H_x0, res0, x_order0);

  // Move only the current value of the first clone, keeping its FEJ fixed.
  auto clone0 = state->_clones_IMU.at(0.0);
  Eigen::Matrix<double, 7, 1> moved;
  moved << ov_core::rot_2_quat(ov_core::exp_so3(Eigen::Vector3d(0.1, 0.0, 0.0)) * clone0->Rot()),
      clone0->pos() + Eigen::Vector3d(0.1, 0.0, 0.0);
  clone0->set_value(moved);

  Eigen::MatrixXd H_f1, H_x1;
  Eigen::VectorXd res1;
  std::vector<std::shared_ptr<Type>> x_order1;
  LineTriangulation::get_jacobian_full(state, feat, H_f1, H_x1, res1, x_order1);

  // Residual changes (current value), Jacobian does not (FEJ value).
  assert((res0 - res1).norm() > 1e-4);
  assert((H_x0 - H_x1).norm() < 1e-10);
  assert((H_f0 - H_f1).norm() < 1e-10);
}

int main() {
  test_triangulation();
  std::cout << "test_triangulation passed" << std::endl;

  test_Hx_global();
  std::cout << "test_Hx_global passed" << std::endl;

  test_Hx_anchored();
  std::cout << "test_Hx_anchored passed" << std::endl;

  test_Hx_fej();
  std::cout << "test_Hx_fej passed" << std::endl;

  test_fej_actually_used();
  std::cout << "test_fej_actually_used passed" << std::endl;

  std::cout << "ALL LINE JACOBIAN TESTS PASSED" << std::endl;
  return 0;
}
