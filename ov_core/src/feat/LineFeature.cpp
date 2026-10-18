#include "LineFeature.h"

#include <algorithm>

namespace ov_core {

void LineFeature::clean_older_than(double timestamp) {
  for (auto &camera_observations : observations) {
    auto &items = camera_observations.second;
    items.erase(std::remove_if(items.begin(), items.end(), [timestamp](const LineObservation &observation) {
                  return observation.timestamp < timestamp;
                }),
                items.end());
  }
}

void LineFeature::clean_invalid_timestamps(const std::vector<double> &valid_timestamps) {
  for (auto &camera_observations : observations) {
    auto &items = camera_observations.second;
    items.erase(std::remove_if(items.begin(), items.end(), [&valid_timestamps](const LineObservation &observation) {
                  return std::find(valid_timestamps.begin(), valid_timestamps.end(), observation.timestamp) == valid_timestamps.end();
                }),
                items.end());
  }
}

size_t LineFeature::count_measurements() const {
  size_t count = 0;
  for (const auto &camera_observations : observations) {
    count += camera_observations.second.size();
  }
  return count;
}

} // namespace ov_core