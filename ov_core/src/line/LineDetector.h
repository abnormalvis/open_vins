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

#ifndef OV_CORE_LINE_DETECTOR_H
#define OV_CORE_LINE_DETECTOR_H

#include "track/TrackLine.h"

namespace ov_core {

/**
 * @brief Line detector backend selection.
 *
 * Only FLD (cv::ximgproc::FastLineDetector) is available at runtime: OpenCV 4.2's
 * contrib build has removed LSD/EDLines (LineSegmentDetector / line_descriptor::LSDDetector /
 * BinaryDescriptor::detect) because of the original code's license, and calling them throws.
 * LBD descriptors (BinaryDescriptor::compute) are a *separate* code path from the removed
 * detect() and are usable on externally supplied line segments, so FLD + LBD is the only
 * runtime-viable descriptor-carrying combination.
 */
enum class DetectorMode {
  FLD_LBD, ///< Fast Line Detector segments + LBD (256-bit) descriptors
  FLD      ///< Fast Line Detector segments only, no descriptor (geometric/optical-flow matching)
};

/**
 * @brief Parameters for the Fast Line Detector backend.
 */
struct LineDetectorOptions {
  int length_threshold = 10;          ///< segments shorter than this are discarded
  float distance_threshold = 1.414213562f; ///< outlier distance for a point off the hypothesis line
  double canny_th1 = 50.0;            ///< first hysteresis threshold for Canny
  double canny_th2 = 50.0;            ///< second hysteresis threshold for Canny
  int canny_aperture_size = 3;        ///< aperture size for the Sobel operator in Canny
  bool do_merge = false;              ///< incremental merging of segments
};

/**
 * @brief Factory producing a TrackLine-compatible line extractor for a given backend.
 */
class LineDetectorFactory {
public:
  /**
   * @brief Build a line extractor for the requested backend.
   *
   * The returned callable has signature `(gray, mask) -> vector<DetectedLine>` and can be
   * handed directly to TrackLine. In FLD_LBD mode each detected segment carries a 256-bit
   * (32-byte, CV_8U) LBD descriptor; in FLD mode the descriptor is left empty so TrackLine's
   * matcher selects the geometric/optical-flow path per line.
   */
  static TrackLine::LineExtractor make_extractor(DetectorMode mode, const LineDetectorOptions &options);
};

} // namespace ov_core

#endif // OV_CORE_LINE_DETECTOR_H
