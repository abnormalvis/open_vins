#ifndef OV_CORE_LINE_FEATURE_H
#define OV_CORE_LINE_FEATURE_H

#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

namespace ov_core {

struct LineObservation {
  double timestamp = 0.0;
  size_t camera_id = 0;
  Eigen::Vector4d endpoints_px = Eigen::Vector4d::Zero();
  Eigen::Vector4d endpoints_norm = Eigen::Vector4d::Zero();
  Eigen::Vector3d line_norm = Eigen::Vector3d::Zero();
};

class LineFeature {
public:
  size_t featid = 0;
  bool to_delete = false;
  std::unordered_map<size_t, std::vector<LineObservation>> observations;

  void clean_older_than(double timestamp);
  void clean_invalid_timestamps(const std::vector<double> &valid_timestamps);

  /// Total number of measurements across all cameras
  size_t count_measurements() const;
};

} // namespace ov_core

#endif // OV_CORE_LINE_FEATURE_H