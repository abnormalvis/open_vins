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

#ifndef OV_CORE_LINE_LINE_MATH_H
#define OV_CORE_LINE_LINE_MATH_H

#include <Eigen/Eigen>

namespace ov_core {
namespace line {

/**
 * @brief Convert a Plücker line (n, d) to the orthonormal representation (U, W).
 *
 * A 3D line is represented by Plücker coordinates L = (n; d), n, d in R^3 with n^T d = 0,
 * where n = p x d is the moment and d the direction. The orthonormal representation (see
 * Bartoli & Sturm, and He et al. PL-VIO) stores
 *   U = [u1 u2 u3] in SO(3), with u1 = n/||n||, u2 = d/||d||, u3 = u1 x u2,
 *   W = [[w1, -w2], [w2, w1]] in SO(2), with w1 = ||n||/s, w2 = ||d||/s, s = sqrt(||n||^2+||d||^2).
 * Recovery: n = w1 u1, d = w2 u2.
 *
 * This conversion is a deterministic inverse of orthonormal_to_plucker() (w1, w2 >= 0 and
 * u1, u2 aligned with n, d), so the two round-trip exactly. When ||n|| ~ 0 (line through the
 * origin) u1 is chosen as a stable unit vector orthogonal to u2 (Gram-Schmidt against a fixed
 * axis) so the decomposition is well defined.
 *
 * @param n Plücker moment (any scale, does not need n^T d == 0)
 * @param d Plücker direction (non-zero)
 * @param U Output 3x3 orthonormal frame
 * @param W Output 2x2 SO(2) matrix
 */
void plucker_to_orthonormal(const Eigen::Vector3d &n, const Eigen::Vector3d &d, Eigen::Matrix3d &U, Eigen::Matrix2d &W);

/**
 * @brief Recover the Plücker line (n, d) from the orthonormal representation (U, W).
 * @param U 3x3 orthonormal frame
 * @param W 2x2 SO(2) matrix ([[w1,-w2],[w2,w1]])
 * @param n Output Plücker moment (n = w1 u1)
 * @param d Output Plücker direction (d = w2 u2)
 */
void orthonormal_to_plucker(const Eigen::Matrix3d &U, const Eigen::Matrix2d &W, Eigen::Vector3d &n, Eigen::Vector3d &d);

/**
 * @brief Apply the 4-DOF error-state boxplus to a Plücker line.
 *
 * Error state dx = [dtheta(3); dphi(1)]. The update is U <- U Exp(dtheta), W <- W R(dphi),
 * with R(dphi) = [[cos,-sin],[sin,cos]]. Small dx keeps the line representation continuous.
 *
 * @param n, d Current Plücker line
 * @param dx 4x1 error state
 * @param n_out, d_out Updated Plücker line
 */
void boxplus(const Eigen::Vector3d &n, const Eigen::Vector3d &d, const Eigen::Vector4d &dx, Eigen::Vector3d &n_out,
             Eigen::Vector3d &d_out);

/**
 * @brief Analytic Jacobian of boxplus() at dx = 0: d[n; d] / d[dx] (6x4).
 * @param n, d Current Plücker line
 * @return 6x4 Jacobian
 */
Eigen::Matrix<double, 6, 4> boxplus_jacobian(const Eigen::Vector3d &n, const Eigen::Vector3d &d);

/**
 * @brief Transform a Plücker line between frames where X_dest = R X_src + t.
 *   n_dest = R n_src + [t]x R d_src, d_dest = R d_src.
 * @param R, t Rotation/translation of the frame transform
 * @param n_src, d_src Source line
 * @param n_dest, d_dest Destination line
 */
void transform_plucker(const Eigen::Matrix3d &R, const Eigen::Vector3d &t, const Eigen::Vector3d &n_src,
                       const Eigen::Vector3d &d_src, Eigen::Vector3d &n_dest, Eigen::Vector3d &d_dest);

/**
 * @brief Triangulate a 3D line as the intersection of two back-projected planes.
 *
 * Each plane is given by its normal pi_i and offset rho_i such that pi_i^T X = rho_i.
 * The intersection line is (Bartoli & Sturm):
 *   d = pi1 x pi2,  n = rho2 * pi1 - rho1 * pi2.
 * The result is normalized to ||n||^2 + ||d||^2 = 1.
 *
 * @param pi1, rho1 First plane
 * @param pi2, rho2 Second plane
 * @param n_out, d_out Triangulated line (normalized)
 * @return true if the planes are non-degenerate (angle above ~1 deg), false otherwise
 */
bool triangulate_from_planes(const Eigen::Vector3d &pi1, double rho1, const Eigen::Vector3d &pi2, double rho2,
                             Eigen::Vector3d &n_out, Eigen::Vector3d &d_out);

/**
 * @brief Project a global Plücker line into the normalized camera frame (l = n_C).
 *   n_C = R_GtoC (n_G - [p_CinG]x d_G).
 * @param R_GtoC Global-to-camera rotation
 * @param p_CinG Camera position in global frame
 * @param n_G, d_G Global line
 * @return 3x1 normalized image line (homogeneous)
 */
Eigen::Vector3d project_line(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG, const Eigen::Vector3d &n_G,
                             const Eigen::Vector3d &d_G);

/**
 * @brief Point-to-line reprojection residual (2x1), one component per endpoint.
 *   r_i = P_i^T l / m, m = sqrt(l0^2 + l1^2).
 * @param l Projected normalized line
 * @param P_s, P_e Normalized homogeneous endpoints (3x1, last element 1)
 * @return 2x1 residual
 */
Eigen::Vector2d line_residual(const Eigen::Vector3d &l, const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e);

/**
 * @brief Jacobian of line_residual() wrt the projected line l (2x3).
 * @param l Projected normalized line
 * @param P_s, P_e Normalized homogeneous endpoints
 * @return 2x3 Jacobian d res / d l
 */
Eigen::Matrix<double, 2, 3> line_residual_jacobian_wrt_l(const Eigen::Vector3d &l, const Eigen::Vector3d &P_s,
                                                         const Eigen::Vector3d &P_e);

/**
 * @brief Jacobian of the line residual wrt the 4-DOF global line error state (2x4).
 *
 * Uses the orthonormal parametrization: with n_G = w1 u1, d_G = w2 u2, and n_C = w1 R u1 - w2 R [p]x u2:
 *   dn_C/dtheta = [ -w2 R [p]x u3,  -w1 R u3,  w1 R u2 + w2 R [p]x u1 ]   (3x3)
 *   dn_C/dphi   = -w2 R u1 - w1 R [p]x u2                                  (3x1)
 *   H_f = H_line * [dn_C/dtheta, dn_C/dphi].
 *
 * @param R_GtoC Global-to-camera rotation
 * @param p_CinG Camera position in global frame
 * @param n_G, d_G Global line
 * @param P_s, P_e Normalized homogeneous endpoints
 * @return 2x4 Jacobian d res / d dx_line
 */
Eigen::Matrix<double, 2, 4> line_jacobian_wrt_global_line(const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG,
                                                          const Eigen::Vector3d &n_G, const Eigen::Vector3d &d_G,
                                                          const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e);

/**
 * @brief Jacobian of the line residual wrt the 4-DOF anchored line error state (2x4).
 *
 * The line is stored in an anchor frame A and transformed to global via X_G = R_A X_A + p_AinG
 * (R_A = anchor-to-global), then to camera via B.3. With q = p_AinG - p_CinG:
 *   n_C = R_GtoC R_A n_A + R_GtoC [q]x R_A d_A
 *   dn_C/ddx = [R_GtoC R_A, R_GtoC [q]x R_A] * d[n_A; d_A]/ddx,  d[n_A; d_A]/ddx = boxplus_jacobian(n_A, d_A)
 *
 * @param R_A Anchor-to-global rotation
 * @param p_AinG Anchor position in global frame
 * @param R_GtoC Global-to-camera rotation
 * @param p_CinG Camera position in global frame
 * @param n_A, d_A Anchored line
 * @param P_s, P_e Normalized homogeneous endpoints
 * @return 2x4 Jacobian d res / d dx_line
 */
Eigen::Matrix<double, 2, 4> line_jacobian_wrt_anchored_line(const Eigen::Matrix3d &R_A, const Eigen::Vector3d &p_AinG,
                                                            const Eigen::Matrix3d &R_GtoC, const Eigen::Vector3d &p_CinG,
                                                            const Eigen::Vector3d &n_A, const Eigen::Vector3d &d_A,
                                                            const Eigen::Vector3d &P_s, const Eigen::Vector3d &P_e);

} // namespace line
} // namespace ov_core

#endif // OV_CORE_LINE_LINE_MATH_H
