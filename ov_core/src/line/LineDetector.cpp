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

#include "line/LineDetector.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/ximgproc.hpp>

namespace ov_core {
namespace {

/**
 * @brief Build a fully-populated cv::line_descriptor::KeyLine from a FLD segment.
 *
 * LBD's BinaryDescriptor::compute() walks several KeyLine fields (octave endpoints, length,
 * pixel count, midpoint, response, size); populating only the four original-image endpoints
 * leaves the rest uninitialized and segfaults inside compute(). Every field is therefore set
 * explicitly. FLD reports segments at original resolution only, so octave == 0 and the
 * octave-relative endpoints coincide with the original-image endpoints.
 */
cv::line_descriptor::KeyLine make_keyline(const cv::Vec4f &line, int class_id, const cv::Mat &image) {
  const float dx = line[2] - line[0];
  const float dy = line[3] - line[1];
  const float length = std::sqrt(dx * dx + dy * dy);

  cv::line_descriptor::KeyLine keyline;
  keyline.angle = std::atan2(dy, dx);
  keyline.class_id = class_id;
  keyline.octave = 0;
  keyline.pt = cv::Point2f(0.5f * (line[0] + line[2]), 0.5f * (line[1] + line[3]));
  // Response is the line length normalized by the larger image dimension (per OpenCV docs).
  keyline.response = length / static_cast<float>(std::max(image.cols, image.rows));
  // Area of the smallest rectangle containing the line (a degenerate rectangle of length x 0).
  keyline.size = length;
  keyline.startPointX = line[0];
  keyline.startPointY = line[1];
  keyline.endPointX = line[2];
  keyline.endPointY = line[3];
  keyline.sPointInOctaveX = line[0];
  keyline.sPointInOctaveY = line[1];
  keyline.ePointInOctaveX = line[2];
  keyline.ePointInOctaveY = line[3];
  keyline.lineLength = length;
  keyline.numOfPixels = std::max(1, static_cast<int>(std::lround(length)));
  return keyline;
}

TrackLine::LineExtractor make_fld(const LineDetectorOptions &options) {
  const auto fld = cv::ximgproc::createFastLineDetector(options.length_threshold, options.distance_threshold, options.canny_th1,
                                                        options.canny_th2, options.canny_aperture_size, options.do_merge);
  return [fld](const cv::Mat &image, const cv::Mat &) {
    std::vector<cv::Vec4f> segments;
    fld->detect(image, segments);
    std::vector<TrackLine::DetectedLine> detected_lines;
    detected_lines.reserve(segments.size());
    for (const cv::Vec4f &segment : segments) {
      TrackLine::DetectedLine detected;
      detected.endpoints = segment;
      // Descriptor intentionally left empty: TrackLine's matcher selects the geometric path.
      detected_lines.push_back(std::move(detected));
    }
    return detected_lines;
  };
}

TrackLine::LineExtractor make_fld_lbd(const LineDetectorOptions &options) {
  const auto fld = cv::ximgproc::createFastLineDetector(options.length_threshold, options.distance_threshold, options.canny_th1,
                                                        options.canny_th2, options.canny_aperture_size, options.do_merge);
  const auto descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
  return [fld, descriptor](const cv::Mat &image, const cv::Mat &) {
    std::vector<cv::Vec4f> segments;
    fld->detect(image, segments);
    std::vector<TrackLine::DetectedLine> detected_lines;
    detected_lines.reserve(segments.size());
    if (segments.empty()) {
      return detected_lines;
    }

    // Build fully-populated keylines (octave 0) so compute() does not dereference stale fields.
    std::vector<cv::line_descriptor::KeyLine> keylines;
    keylines.reserve(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
      keylines.push_back(make_keyline(segments[index], static_cast<int>(index), image));
    }

    // compute() updates keylines in place; use the updated keylines for endpoints and the i-th
    // descriptor row for the i-th line's descriptor.
    cv::Mat descriptors;
    descriptor->compute(image, keylines, descriptors);

    detected_lines.resize(segments.size());
    for (size_t index = 0; index < segments.size(); ++index) {
      const cv::line_descriptor::KeyLine &keyline = keylines[index];
      detected_lines[index].endpoints = cv::Vec4f(keyline.startPointX, keyline.startPointY, keyline.endPointX, keyline.endPointY);
      if (!descriptors.empty() && static_cast<int>(index) < descriptors.rows) {
        detected_lines[index].descriptor = descriptors.row(static_cast<int>(index)).clone();
      }
    }
    return detected_lines;
  };
}

} // namespace

TrackLine::LineExtractor LineDetectorFactory::make_extractor(DetectorMode mode, const LineDetectorOptions &options) {
  switch (mode) {
    case DetectorMode::FLD_LBD:
      return make_fld_lbd(options);
    case DetectorMode::FLD:
    default:
      return make_fld(options);
  }
}

} // namespace ov_core
