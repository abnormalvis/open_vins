#ifndef OV_CORE_LINE_FEATURE_DATABASE_H
#define OV_CORE_LINE_FEATURE_DATABASE_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "LineFeature.h"

namespace ov_core {

class LineFeatureDatabase {
public:
  void update_feature(size_t id, const LineObservation &observation);

  /// Copy a feature's data into the passed object (thread safe, returns false if not found)
  bool get_feature(size_t id, LineFeature &feature) const;

  /// Get a pointer to a feature, optionally removing it from the database (not thread safe to keep using)
  std::shared_ptr<LineFeature> get_feature(size_t id, bool remove = false);

  /// Get features that do not have a measurement at a time newer than the specified time
  std::vector<std::shared_ptr<LineFeature>> features_not_containing_newer(double timestamp, bool remove = false, bool skip_deleted = false);

  /// Get features that have at least one measurement older than the specified time
  std::vector<std::shared_ptr<LineFeature>> features_containing_older(double timestamp, bool remove = false, bool skip_deleted = false);

  /// Get features that have a measurement exactly at the specified time
  std::vector<std::shared_ptr<LineFeature>> features_containing(double timestamp, bool remove = false, bool skip_deleted = false);

  /// Deep-copy of all features (thread safe, does not remove)
  std::vector<std::shared_ptr<LineFeature>> get_features() const;

  void erase_features(const std::vector<size_t> &ids);
  void cleanup_older_than(double timestamp);

  /// Delete all features whose to_delete flag is set
  void cleanup();

  /// Remove all measurements older than the specified timestamp, dropping empty features
  void cleanup_measurements(double timestamp);

  /// Oldest measurement timestamp in the database (-1 if empty)
  double get_oldest_timestamp();

  /// Number of features in the database
  size_t size();

  /// Internal map (should not normally be used)
  std::unordered_map<size_t, std::shared_ptr<LineFeature>> get_internal_data();

private:
  mutable std::mutex mutex_;
  std::unordered_map<size_t, std::shared_ptr<LineFeature>> features_;
};

} // namespace ov_core

#endif // OV_CORE_LINE_FEATURE_DATABASE_H