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

#include "line/LineMath.h"

#include <cmath>

#include "utils/quat_ops.h"

namespace ov_core {
namespace line {

void plucker_to_orthonormal(const Eigen::Vector3d &n, const Eigen::Vector3d &d, Eigen::Matrix3d &U, Eigen::Matrix2d &W) {

  // Canonical scale so that ||n||^2 + ||d||^2 = 1.
  const double scale = std::sqrt(n.squaredNorm() + d.squaredNorm());
  const Eigen::Vector3d nn = n / scale;
  const Eigen::Vector3d dd = d / scale;

  const double w1 = nn.norm();
  const double w2 = dd.norm();

  // Direction u2 = d / ||d||. Note dd is non-zero by assumption.
  const Eigen::Vector3d u2 = dd / w2;

  // Moment direction u1 = n / ||n|| when well defined, otherwise pick a
  // deterministic unit vector orthogonal to u2 (stable Gram-Schmidt against a
  // fixed axis). This keeps the frame well defined for lines through the origin
  // (||n|| ~ 0), where the result n = w1 u1 is insensitive to u1.
  Eigen::Vector3d u1;
  if (w1 > 1e-8) {
    u1 = nn / w1;
  } else {
    const Eigen::Vector3d e = (std::abs(u2(0)) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    u1 = (e - u2.dot(e) * u2).normalized();
  }

  const Eigen::Vector3d u3 = u1.cross(u2);

  U.col(0) = u1;
  U.col(1) = u2;
  U.col(2) = u3;

  W << w1, -w2, w2, w1;
}

void orthonormal_to_plucker(const Eigen::Matrix3d &U, const Eigen::Matrix2d &W, Eigen::Vector3d &n, Eigen::Vector3d &d) {
  const double w1 = W(0, 0);
  const double w2 = W(1, 0);
  n = w1 * U.col(0);
  d = w2 * U.col(1);
}

void boxplus(const Eigen::Vector3d &n, const Eigen::Vector3d &d, const Eigen::Vector4d &dx, Eigen::Vector3d &n_out,
             Eigen::Vector3d &d_out) {

  Eigen::Matrix3d U;
  Eigen::Matrix2d W;
  plucker_to_orthonormal(n, d, U, W);

  U = U * ov_core::exp_so3(dx.head<3>().eval());

  const double c = std::cos(dx(3));
  const double s = std::sin(dx(3));
  Eigen::Matrix2d R;
  R << c, -s, s, c;
  W = W * R;

  orthonormal_to_plucker(U, W, n_out, d_out);
}

Eigen::Matrix<double, 6, 4> boxplus_jacobian(const Eigen::Vector3d &n, const Eigen::Vector3d &d) {

  Eigen::Matrix3d U;
  Eigen::Matrix2d W;
  plucker_to_orthonormal(n, d, U, W);

  const double w1 = W(0, 0);
  const double w2 = W(1, 0);
  const Eigen::Vector3d u1 = U.col(0);
  const Eigen::Vector3d u2 = U.col(1);
  const Eigen::Vector3d u3 = U.col(2);

  // d[n; d] / d[dtheta; dphi] at dx = 0. Column layout: [dtheta1 dtheta2 dtheta3 dphi].
  Eigen::Matrix<double, 6, 4> J;
  J.col(0) << Eigen::Vector3d::Zero(), w2 * u3;
  J.col(1) << -w1 * u3, Eigen::Vector3d::Zero();
  J.col(2) << w1 * u2, -w2 * u1;
  J.col(3) << -w2 * u1, w1 * u2;
  return J;
}

void transform_plucker(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Vector3d &n_src,
                       const Eigen::Vector3d &d_src, Eigen::Vector3d &n_dest, Eigen::Vector3d &d_dest) {
  d_dest = R * d_src;
  n_dest = R * n_src + ov_core::skew_x(t) * R * d_src;
}

bool triangulate_from_planes(const Eigen::Vector3d &pi1, double rho1, const Eigen::Vector3d &pi2, double rho2,
                             Eigen::Vector3d &n_out, Eigen::Vector3d &d_out) {

  const double norm1 = pi1.norm();
  const double norm2 = pi2.norm();
  if (norm1 < 1e-8 || norm2 < 1e-8) {
    return false;
  }

  const Eigen::Vector3d d = pi1.cross(pi2);

  // Reject near-parallel planes: sin(angle) must be above ~1 degree.
  const double sin_angle = d.norm() / (norm1 * norm2);
  if (sin_angle < 0.017) {
    return false;
  }

  const Eigen::Vector3d n = rho2 * pi1 - rho1 * pi2;

  const double scale = std::sqrt(n.squaredNorm() + d.squaredNorm());
  if (scale < 1e-12) {
    return false;
  }

  n_out = n / scale;
  d_out = d / scale;
  return true;
}

Eigen::Vector3d project_line(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG, const Eigen::Vector3d &n_G,
                             const Eigen::Vector3d &d_G) {
  return R_GtoC * (n_G - ov_core::skew_x(p_CinG) * d_G);
}

Eigen::Vector2d line_residual(const Eigen::Vector3d &l, const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e) {
  const double m = std::sqrt(l(0) * l(0) + l(1) * l(1));
  Eigen::Vector2d r;
  r(0) = P_s.dot(l) / m;
  r(1) = P_e.dot(l) / m;
  return r;
}

Eigen::Matrix<double, 2, 3> line_residual_jacobian_wrt_l(const Eigen::Vector3d &l, const Eigen::Vector3d &P_s,
                                                         const Eigen::Vector3d &P_e) {
  const double m = std::sqrt(l(0) * l(0) + l(1) * l(1));
  const double m3 = m * m * m;
  const Eigen::RowVector3d grad_m(l(0), l(1), 0.0);

  Eigen::Matrix<double, 2, 3> H;
  H.row(0) = P_s.transpose() / m - (P_s.dot(l) / m3) * grad_m;
  H.row(1) = P_e.transpose() / m - (P_e.dot(l) / m3) * grad_m;
  return H;
}

Eigen::Matrix<double, 2, 4> line_jacobian_wrt_global_line(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG,
                                                          const Eigen::Vector3d &n_G, const Eigen::Vector3d &d_G,
                                                          const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e) {

  Eigen::Matrix3d U;
  Eigen::Matrix2d W;
  plucker_to_orthonormal(n_G, d_G, U, W);

  const double w1 = W(0, 0);
  const double w2 = W(1, 0);
  const Eigen::Vector3d u1 = U.col(0);
  const Eigen::Vector3d u2 = U.col(1);
  const Eigen::Vector3d u3 = U.col(2);

  const Eigen::Matrix3d R = R_GtoC;
  const Eigen::Matrix3d px = ov_core::skew_x(p_CinG);

  const Eigen::Vector3d l = project_line(R_GtoC, p_CinG, n_G, d_G);
  const Eigen::Matrix<double, 2, 3> H_line = line_residual_jacobian_wrt_l(l, P_s, P_e);

  Eigen::Matrix<double, 3, 4> dnC_ddx;
  dnC_ddx.col(0) = -w2 * R * px * u3;
  dnC_ddx.col(1) = -w1 * R * u3;
  dnC_ddx.col(2) = w1 * R * u2 + w2 * R * px * u1;
  dnC_ddx.col(3) = -w2 * R * u1 - w1 * R * px * u2;

  return H_line * dnC_ddx;
}

Eigen::Matrix<double, 2, 4> line_jacobian_wrt_anchored_line(const Eigen::Matrix3d &R_A, const Eigen::Vector3d &p_AinG,
                                                            const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG,
                                                            const Eigen::Vector3d &n_A, const Eigen::Vector3d &d_A,
                                                            const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e) {

  const Eigen::Matrix3d R_AC = R_GtoC * R_A;
  const Eigen::Matrix3d M = R_GtoC * ov_core::skew_x(p_AinG - p_CinG) * R_A;

  const Eigen::Matrix<double, 6, 4> J = boxplus_jacobian(n_A, d_A);
  const Eigen::Matrix<double, 3, 4> dnC_ddx = R_AC * J.topRows<3>() + M * J.bottomRows<3>();

  Eigen::Vector3d n_G, d_G;
  transform_plucker(R_A, p_AinG, n_A, d_A, n_G, d_G);
  const Eigen::Vector3d l = project_line(R_GtoC, p_CinG, n_G, d_G);
  const Eigen::Matrix<double, 2, 3> H_line = line_residual_jacobian_wrt_l(l, P_s, P_e);

  return H_line * dnC_ddx;
}

} // namespace line
} // namespace ov_core
