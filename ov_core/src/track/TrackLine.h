#ifndef OV_CORE_TRACK_LINE_H
#define OV_CORE_TRACK_LINE_H

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "cam/CamBase.h"
#include "feat/LineFeatureDatabase.h"
#include "utils/sensor_data.h"

namespace ov_core {

class TrackLine {
public:
  struct Options {
    size_t max_lines;
    double min_line_length;
    double max_endpoint_motion;
    double max_angle_difference_deg;
    double descriptor_ratio;

    Options() : max_lines(150), min_line_length(25.0), max_endpoint_motion(80.0), max_angle_difference_deg(25.0),
                descriptor_ratio(0.8) {}
  };

  struct DetectedLine {
    cv::Vec4f endpoints;
    cv::Mat descriptor;
  };

  using LineExtractor = std::function<std::vector<DetectedLine>(const cv::Mat &, const cv::Mat &)>;

  TrackLine(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, const Options &options, LineExtractor line_extractor);

  void feed_new_camera(const CameraData &message);
  std::shared_ptr<LineFeatureDatabase> get_feature_database() const { return database_; }
  std::unordered_map<size_t, std::vector<cv::Vec4f>> get_last_lines() const;
  std::unordered_map<size_t, std::vector<size_t>> get_last_ids() const;

private:
  struct Candidate {
    cv::Point2f start;
    cv::Point2f end;
    cv::Mat descriptor;
    size_t id = 0;
  };

  struct CameraState {
    cv::Mat previous_image;
    std::vector<Candidate> previous_lines;
  };

  std::vector<Candidate> detect_lines(const cv::Mat &image, const cv::Mat &mask);
  void match_lines(const cv::Mat &previous_image, const cv::Mat &current_image, const std::vector<Candidate> &previous_lines,
                   std::vector<Candidate> &current_lines);
  void match_descriptor(const std::vector<Candidate> &previous_lines, std::vector<Candidate> &current_lines,
                        std::vector<bool> &previous_used, std::vector<bool> &current_used);
  void match_geometric(const cv::Mat &previous_image, const cv::Mat &current_image, const std::vector<Candidate> &previous_lines,
                       std::vector<Candidate> &current_lines, std::vector<bool> &previous_used, std::vector<bool> &current_used);
  bool geometry_compatible(const Candidate &first, const Candidate &second, double max_motion) const;
  LineObservation make_observation(size_t camera_id, double timestamp, const Candidate &line) const;

  std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras_;
  Options options_;
  LineExtractor line_extractor_;
  std::shared_ptr<LineFeatureDatabase> database_;
  std::unordered_map<size_t, CameraState> camera_states_;
  std::unordered_map<size_t, std::vector<cv::Vec4f>> last_lines_;
  std::unordered_map<size_t, std::vector<size_t>> last_ids_;
  std::atomic<size_t> next_id_{1};
  mutable std::mutex mutex_;
};

} // namespace ov_core

#endif // OV_CORE_TRACK_LINE_H