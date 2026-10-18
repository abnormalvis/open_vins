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
#include <random>

#include "line/LineMath.h"
#include "types/LandmarkLine.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_core::line;

// Deterministic RNG so failures are reproducible.
static std::mt19937 g_rng(42);

static double rand_uniform(double a, double b) {
  std::uniform_real_distribution<double> dist(a, b);
  return dist(g_rng);
}

static Eigen::Vector3d rand_vec3(double scale = 1.0) {
  return Eigen::Vector3d(rand_uniform(-scale, scale), rand_uniform(-scale, scale), rand_uniform(-scale, scale));
}

static Eigen::Matrix3d rand_rotation() { return ov_core::exp_so3(rand_vec3(1.0)); }

// Random normalized Plücker line (n^T d = 0, ||n||^2 + ||d||^2 = 1).
// We keep w1 = ||n|| bounded away from 0 and 1 so the orthonormal gauge (sign of u1/u2)
// is stable under small perturbations; the singular case is tested separately.
static void rand_line(Eigen::Vector3d &n, Eigen::Vector3d &d) {
  while (true) {
    d = rand_vec3(1.0);
    if (d.norm() < 1e-6) {
      continue;
    }
    d.normalize();
    n = rand_vec3(1.0);
    n = n - n.dot(d) * d;
    const double scale = std::sqrt(n.squaredNorm() + d.squaredNorm());
    n /= scale;
    d /= scale;
    if (n.norm() > 0.2 && n.norm() < 0.8) {
      return;
    }
  }
}

// Random normalized homogeneous endpoint [u; v; 1].
static Eigen::Vector3d rand_endpoint() {
  return Eigen::Vector3d(rand_uniform(-1.0, 1.0), rand_uniform(-1.0, 1.0), 1.0);
}

static void test_plucker_orthonormal_roundtrip() {
  // round-trip for generic and sign-flipped lines
  for (int trial = 0; trial < 1000; ++trial) {
    Eigen::Vector3d n, d;
    rand_line(n, d);
    Eigen::Matrix3d U;
    Eigen::Matrix2d W;
    plucker_to_orthonormal(n, d, U, W);
    Eigen::Vector3d n_out, d_out;
    orthonormal_to_plucker(U, W, n_out, d_out);
    assert((n_out - n).norm() < 1e-8);
    assert((d_out - d).norm() < 1e-8);
    // (U, W) must be valid orthonormal frame
    assert((U.transpose() * U - Eigen::Matrix3d::Identity()).norm() < 1e-8);
    assert(std::abs(W.determinant() - 1.0) < 1e-8);

    // Sign-flipped line is the same line and must round-trip consistently.
    plucker_to_orthonormal(-n, -d, U, W);
    orthonormal_to_plucker(U, W, n_out, d_out);
    assert((n_out + n).norm() < 1e-8);
    assert((d_out + d).norm() < 1e-8);
  }
}

static void test_triangulation() {
  int tested = 0;
  for (int trial = 0; trial < 500; ++trial) {
    // Random unit direction and a unit moment orthogonal to it (so ||d|| = ||n|| = 1).
    Eigen::Vector3d d_gt = rand_vec3(1.0);
    if (d_gt.norm() < 1e-6) {
      continue;
    }
    d_gt.normalize();
    Eigen::Vector3d n_gt = rand_vec3(1.0);
    n_gt = n_gt - n_gt.dot(d_gt) * d_gt;
    if (n_gt.norm() < 1e-3) {
      continue;
    }
    n_gt.normalize();

    // Closest point of the line to the origin, then a point along the line.
    const Eigen::Vector3d p_closest = d_gt.cross(n_gt);
    const double lambda = rand_uniform(-5.0, 5.0);
    const Eigen::Vector3d p0 = p_closest + lambda * d_gt;

    // Two camera centers placed along two orthonormal directions v1, v2 that are both
    // orthogonal to d_gt. This guarantees the two planes are non-degenerate (orthogonal).
    const Eigen::Vector3d e = (std::abs(d_gt(0)) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    const Eigen::Vector3d v1 = (e - d_gt.dot(e) * d_gt).normalized();
    const Eigen::Vector3d v2 = d_gt.cross(v1);
    const double a1 = rand_uniform(0.5, 5.0);
    const double a2 = rand_uniform(0.5, 5.0);
    const Eigen::Vector3d p1 = p0 + a1 * v1;
    const Eigen::Vector3d p2 = p0 + a2 * v2;

    // Plane i contains p_i and the line: pi_i = d_gt x (p0 - p_i), rho_i = pi_i^T p_i.
    const Eigen::Vector3d pi1 = d_gt.cross(p0 - p1);
    const Eigen::Vector3d pi2 = d_gt.cross(p0 - p2);
    const double rho1 = pi1.dot(p1);
    const double rho2 = pi2.dot(p2);

    Eigen::Vector3d n, d;
    const bool ok = triangulate_from_planes(pi1, rho1, pi2, rho2, n, d);
    assert(ok);

    // Valid Plücker + normalization
    assert(std::abs(n.dot(d)) < 1e-8);
    assert(std::abs(n.squaredNorm() + d.squaredNorm() - 1.0) < 1e-8);
    // Same geometric line as ground truth (direction and moment parallel)
    assert(std::abs(std::abs(d.normalized().dot(d_gt)) - 1.0) < 1e-8);
    assert(std::abs(std::abs(n.normalized().dot(n_gt)) - 1.0) < 1e-8);
    // The closest point (d x n) / ||d||^2 is on both planes: pi_i^T p_rec == rho_i.
    const Eigen::Vector3d p_rec = d.cross(n) / d.squaredNorm();
    assert(std::abs(pi1.dot(p_rec) - rho1) < 1e-6);
    assert(std::abs(pi2.dot(p_rec) - rho2) < 1e-6);
    ++tested;
  }
  assert(tested > 100);

  // Degenerate: near-parallel planes must be rejected.
  const Eigen::Vector3d pi1 = Eigen::Vector3d::UnitX();
  const Eigen::Vector3d pi2 = Eigen::Vector3d(1.0, 1e-6, 0.0).normalized();
  Eigen::Vector3d n, d;
  assert(!triangulate_from_planes(pi1, 1.0, pi2, 2.0, n, d));
}

static void test_boxplus() {
  // Round-trip: update(dx) then update(-dx) returns to the original for small dx.
  const Eigen::Vector4d dx(0.03, -0.02, 0.01, 0.04);
  for (int trial = 0; trial < 500; ++trial) {
    Eigen::Vector3d n, d;
    rand_line(n, d);
    Eigen::Vector3d n1, d1, n2, d2;
    boxplus(n, d, dx, n1, d1);
    boxplus(n1, d1, -dx, n2, d2);
    assert((n2 - n).norm() < 1e-6);
    assert((d2 - d).norm() < 1e-6);
  }

  // boxplus_jacobian vs central finite difference.
  const double eps = 1e-6;
  for (int trial = 0; trial < 500; ++trial) {
    Eigen::Vector3d n, d;
    rand_line(n, d);
    const Eigen::Matrix<double, 6, 4> J = boxplus_jacobian(n, d);
    for (int j = 0; j < 4; ++j) {
      Eigen::Vector4d e = Eigen::Vector4d::Zero();
      e(j) = eps;
      Eigen::Vector3d np, dp, nm, dm;
      boxplus(n, d, e, np, dp);
      boxplus(n, d, -e, nm, dm);
      Eigen::Matrix<double, 6, 1> num;
      num << (np - nm) / (2 * eps), (dp - dm) / (2 * eps);
      assert((num - J.col(j)).norm() < 1e-5);
    }
  }

  // Singular line through the origin (||n|| ~ 0): decomposition must be well-defined and round-trip.
  const Eigen::Vector3d d = Eigen::Vector3d(0.3, -0.4, 0.8660254).normalized();
  const Eigen::Vector3d n = Eigen::Vector3d::Zero();
  Eigen::Matrix3d U;
  Eigen::Matrix2d W;
  plucker_to_orthonormal(n, d, U, W);
  assert((U.transpose() * U - Eigen::Matrix3d::Identity()).norm() < 1e-8);
  Eigen::Vector3d n_out, d_out;
  orthonormal_to_plucker(U, W, n_out, d_out);
  assert(n_out.norm() < 1e-8);
  assert(std::abs(std::abs(d_out.normalized().dot(d)) - 1.0) < 1e-8);

  // Also verify a tiny (but non-zero) moment still behaves.
  Eigen::Vector3d n_small = d.cross(Eigen::Vector3d(1.0, 2.0, 3.0));
  n_small = 1e-7 * n_small.normalized();
  plucker_to_orthonormal(n_small, d, U, W);
  assert((U.transpose() * U - Eigen::Matrix3d::Identity()).norm() < 1e-8);
}

static void test_landmark_line_boxplus() {
  // LandmarkLine::update() must implement the orthonormal boxplus (not addition).
  const double eps = 1e-6;
  for (int trial = 0; trial < 200; ++trial) {
    Eigen::Vector3d n, d;
    rand_line(n, d);
    ov_type::LandmarkLine landmark;
    landmark.set_from_plucker(n, d, false);

    const Eigen::Matrix<double, 6, 4> J = boxplus_jacobian(n, d);
    for (int j = 0; j < 4; ++j) {
      Eigen::Vector4d e = Eigen::Vector4d::Zero();
      e(j) = eps;
      ov_type::LandmarkLine lp, lm;
      lp.set_from_plucker(n, d, false);
      lm.set_from_plucker(n, d, false);
      lp.update(e);
      lm.update(-e);
      const Eigen::VectorXd vp = lp.value();
      const Eigen::VectorXd vm = lm.value();
      Eigen::Matrix<double, 6, 1> num = (vp - vm) / (2 * eps);
      assert((num - J.col(j)).norm() < 1e-5);
    }
  }
}

static void test_Hf_global() {
  const double eps = 1e-6;
  for (int trial = 0; trial < 500; ++trial) {
    const Eigen::Matrix3d R_GtoC = rand_rotation();
    const Eigen::Vector3d p_CinG = rand_vec3(3.0);
    Eigen::Vector3d n_G, d_G;
    rand_line(n_G, d_G);
    const Eigen::Vector3d P_s = rand_endpoint();
    const Eigen::Vector3d P_e = rand_endpoint();

    const Eigen::Matrix<double, 2, 4> Hf = line_jacobian_wrt_global_line(R_GtoC, p_CinG, n_G, d_G, P_s, P_e);
    for (int j = 0; j < 4; ++j) {
      Eigen::Vector4d e = Eigen::Vector4d::Zero();
      e(j) = eps;
      Eigen::Vector3d np, dp, nm, dm;
      boxplus(n_G, d_G, e, np, dp);
      boxplus(n_G, d_G, -e, nm, dm);
      const Eigen::Vector3d lp = project_line(R_GtoC, p_CinG, np, dp);
      const Eigen::Vector3d lm = project_line(R_GtoC, p_CinG, nm, dm);
      const Eigen::Vector2d num = (line_residual(lp, P_s, P_e) - line_residual(lm, P_s, P_e)) / (2 * eps);
      assert((num - Hf.col(j)).norm() < 1e-5);
    }
  }
}

static void test_Hf_anchored() {
  const double eps = 1e-6;
  for (int trial = 0; trial < 500; ++trial) {
    const Eigen::Matrix3d R_A = rand_rotation();
    const Eigen::Vector3d p_AinG = rand_vec3(3.0);
    const Eigen::Matrix3d R_GtoC = rand_rotation();
    const Eigen::Vector3d p_CinG = rand_vec3(3.0);
    Eigen::Vector3d n_A, d_A;
    rand_line(n_A, d_A);
    const Eigen::Vector3d P_s = rand_endpoint();
    const Eigen::Vector3d P_e = rand_endpoint();

    const Eigen::Matrix<double, 2, 4> Hf = line_jacobian_wrt_anchored_line(R_A, p_AinG, R_GtoC, p_CinG, n_A, d_A, P_s, P_e);
    for (int j = 0; j < 4; ++j) {
      Eigen::Vector4d e = Eigen::Vector4d::Zero();
      e(j) = eps;
      Eigen::Vector3d np, dp, nm, dm;
      boxplus(n_A, d_A, e, np, dp);
      boxplus(n_A, d_A, -e, nm, dm);
      Eigen::Vector3d npg, dpg, nmg, dmg;
      transform_plucker(R_A, p_AinG, np, dp, npg, dpg);
      transform_plucker(R_A, p_AinG, nm, dm, nmg, dmg);
      const Eigen::Vector3d lp = project_line(R_GtoC, p_CinG, npg, dpg);
      const Eigen::Vector3d lm = project_line(R_GtoC, p_CinG, nmg, dmg);
      const Eigen::Vector2d num = (line_residual(lp, P_s, P_e) - line_residual(lm, P_s, P_e)) / (2 * eps);
      assert((num - Hf.col(j)).norm() < 1e-5);
    }
  }
}

int main() {
  test_plucker_orthonormal_roundtrip();
  std::cout << "test_plucker_orthonormal_roundtrip passed" << std::endl;

  test_triangulation();
  std::cout << "test_triangulation passed" << std::endl;

  test_boxplus();
  std::cout << "test_boxplus passed" << std::endl;

  test_landmark_line_boxplus();
  std::cout << "test_landmark_line_boxplus passed" << std::endl;

  test_Hf_global();
  std::cout << "test_Hf_global passed" << std::endl;

  test_Hf_anchored();
  std::cout << "test_Hf_anchored passed" << std::endl;

  std::cout << "ALL LINE MATH TESTS PASSED" << std::endl;
  return 0;
}
