#include "LineFeatureDatabase.h"

#include <algorithm>

namespace ov_core {

void LineFeatureDatabase::update_feature(size_t id, const LineObservation &observation) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto &feature = features_[id];
  if (!feature) {
    feature = std::make_shared<LineFeature>();
    feature->featid = id;
  }
  auto &camera_observations = feature->observations[observation.camera_id];
  if (camera_observations.empty() || camera_observations.back().timestamp != observation.timestamp) {
    camera_observations.push_back(observation);
  } else {
    camera_observations.back() = observation;
  }
}

bool LineFeatureDatabase::get_feature(size_t id, LineFeature &feature) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = features_.find(id);
  if (found == features_.end()) {
    return false;
  }
  feature = *found->second;
  return true;
}

std::shared_ptr<LineFeature> LineFeatureDatabase::get_feature(size_t id, bool remove) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto found = features_.find(id);
  if (found == features_.end()) {
    return nullptr;
  }
  std::shared_ptr<LineFeature> feature = found->second;
  if (remove) {
    features_.erase(found);
  }
  return feature;
}

std::vector<std::shared_ptr<LineFeature>> LineFeatureDatabase::features_not_containing_newer(double timestamp, bool remove, bool skip_deleted) {
  std::vector<std::shared_ptr<LineFeature>> result;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = features_.begin(); it != features_.end();) {
    if (skip_deleted && it->second->to_delete) {
      ++it;
      continue;
    }
    bool has_newer = false;
    for (const auto &camera_observations : it->second->observations) {
      const auto &items = camera_observations.second;
      has_newer = (!items.empty() && items.back().timestamp >= timestamp);
      if (has_newer) {
        break;
      }
    }
    if (!has_newer) {
      result.push_back(it->second);
      if (remove) {
        it = features_.erase(it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
  return result;
}

std::vector<std::shared_ptr<LineFeature>> LineFeatureDatabase::features_containing_older(double timestamp, bool remove, bool skip_deleted) {
  std::vector<std::shared_ptr<LineFeature>> result;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = features_.begin(); it != features_.end();) {
    if (skip_deleted && it->second->to_delete) {
      ++it;
      continue;
    }
    bool has_older = false;
    for (const auto &camera_observations : it->second->observations) {
      const auto &items = camera_observations.second;
      has_older = (!items.empty() && items.front().timestamp < timestamp);
      if (has_older) {
        break;
      }
    }
    if (has_older) {
      result.push_back(it->second);
      if (remove) {
        it = features_.erase(it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
  return result;
}

std::vector<std::shared_ptr<LineFeature>> LineFeatureDatabase::features_containing(double timestamp, bool remove, bool skip_deleted) {
  std::vector<std::shared_ptr<LineFeature>> result;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = features_.begin(); it != features_.end();) {
    if (skip_deleted && it->second->to_delete) {
      ++it;
      continue;
    }
    bool has_timestamp = false;
    for (const auto &camera_observations : it->second->observations) {
      const auto &items = camera_observations.second;
      has_timestamp = std::any_of(items.begin(), items.end(), [timestamp](const LineObservation &o) { return o.timestamp == timestamp; });
      if (has_timestamp) {
        break;
      }
    }
    if (has_timestamp) {
      result.push_back(it->second);
      if (remove) {
        it = features_.erase(it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
  return result;
}

void LineFeatureDatabase::cleanup() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = features_.begin(); it != features_.end();) {
    if (it->second->to_delete) {
      it = features_.erase(it);
    } else {
      ++it;
    }
  }
}

void LineFeatureDatabase::cleanup_measurements(double timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = features_.begin(); it != features_.end();) {
    it->second->clean_older_than(timestamp);
    if (it->second->count_measurements() == 0) {
      it = features_.erase(it);
    } else {
      ++it;
    }
  }
}

double LineFeatureDatabase::get_oldest_timestamp() {
  std::lock_guard<std::mutex> lock(mutex_);
  double oldest = -1;
  for (const auto &entry : features_) {
    for (const auto &camera_observations : entry.second->observations) {
      if (!camera_observations.second.empty()) {
        const double time = camera_observations.second.front().timestamp;
        if (oldest == -1 || oldest > time) {
          oldest = time;
        }
      }
    }
  }
  return oldest;
}

size_t LineFeatureDatabase::size() {
  std::lock_guard<std::mutex> lock(mutex_);
  return features_.size();
}

std::unordered_map<size_t, std::shared_ptr<LineFeature>> LineFeatureDatabase::get_internal_data() {
  std::lock_guard<std::mutex> lock(mutex_);
  return features_;
}

std::vector<std::shared_ptr<LineFeature>> LineFeatureDatabase::get_features() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::shared_ptr<LineFeature>> result;
  result.reserve(features_.size());
  for (const auto &entry : features_) {
    result.push_back(std::make_shared<LineFeature>(*entry.second));
  }
  return result;
}

void LineFeatureDatabase::erase_features(const std::vector<size_t> &ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const size_t id : ids) {
    features_.erase(id);
  }
}

void LineFeatureDatabase::cleanup_older_than(double timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = features_.begin(); iter != features_.end();) {
    iter->second->clean_older_than(timestamp);
    bool empty = true;
    for (const auto &camera_observations : iter->second->observations) {
      empty = empty && camera_observations.second.empty();
    }
    if (empty) {
      iter = features_.erase(iter);
    } else {
      ++iter;
    }
  }
}

} // namespace ov_core