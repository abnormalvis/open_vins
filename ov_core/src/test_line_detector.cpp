#include <cassert>
#include <iostream>

#include <opencv2/imgproc.hpp>

#include "line/LineDetector.h"

int main() {
  // Synthetic but real pixel image with clearly separated line segments (long, high-contrast,
  // well above the default FLD length/contrast thresholds).
  cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
  cv::line(image, cv::Point(50, 100), cv::Point(590, 100), cv::Scalar(255), 3);
  cv::line(image, cv::Point(50, 200), cv::Point(590, 210), cv::Scalar(255), 3);
  cv::line(image, cv::Point(100, 50), cv::Point(110, 430), cv::Scalar(255), 3);
  cv::line(image, cv::Point(200, 50), cv::Point(190, 430), cv::Scalar(255), 3);
  cv::line(image, cv::Point(80, 420), cv::Point(560, 60), cv::Scalar(255), 3);
  cv::line(image, cv::Point(80, 60), cv::Point(560, 420), cv::Scalar(255), 3);

  ov_core::LineDetectorOptions options;

  // FLD mode: real segments, no descriptors.
  auto fld_extractor = ov_core::LineDetectorFactory::make_extractor(ov_core::DetectorMode::FLD, options);
  const auto fld_lines = fld_extractor(image, cv::Mat());
  assert(!fld_lines.empty());
  for (const auto &line : fld_lines) {
    assert(line.descriptor.empty());
  }
  std::cout << "FLD mode: " << fld_lines.size() << " segments, all without descriptors" << std::endl;

  // FLD_LBD mode: real segments, each carrying a 256-bit (32-byte, CV_8U) LBD descriptor. This is
  // the runtime gate for BinaryDescriptor::compute() (a separate code path from the removed
  // detect()); it segfaults unless KeyLine metadata is fully populated, which the extractor does.
  auto fld_lbd_extractor = ov_core::LineDetectorFactory::make_extractor(ov_core::DetectorMode::FLD_LBD, options);
  const auto fld_lbd_lines = fld_lbd_extractor(image, cv::Mat());
  assert(!fld_lbd_lines.empty());
  size_t with_descriptor = 0;
  for (const auto &line : fld_lbd_lines) {
    if (line.descriptor.empty()) {
      continue;
    }
    assert(line.descriptor.type() == CV_8U);
    assert(line.descriptor.rows == 1);
    assert(line.descriptor.cols == 32); // 256 bits
    ++with_descriptor;
  }
  assert(with_descriptor == fld_lbd_lines.size());
  std::cout << "FLD_LBD mode: " << fld_lbd_lines.size() << " segments, all with 32-byte descriptors" << std::endl;

  std::cout << "ALL LINE DETECTOR TESTS PASSED" << std::endl;
  return 0;
}
