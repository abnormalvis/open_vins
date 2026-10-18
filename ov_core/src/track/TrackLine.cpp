#include "TrackLine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace ov_core {
namespace {

double line_angle(const cv::Point2f &start, const cv::Point2f &end) {
  return std::atan2(static_cast<double>(end.y - start.y), static_cast<double>(end.x - start.x));
}

double angle_difference(double first, double second) {
  double difference = std::abs(first - second);
  while (difference > CV_PI) {
    difference -= CV_PI;
  }
  return std::min(difference, CV_PI - difference);
}

double endpoint_distance(const cv::Point2f &first_start, const cv::Point2f &first_end, const cv::Point2f &second_start,
                         const cv::Point2f &second_end) {
  const double direct = cv::norm(first_start - second_start) + cv::norm(first_end - second_end);
  const double reversed = cv::norm(first_start - second_end) + cv::norm(first_end - second_start);
  return 0.5 * std::min(direct, reversed);
}

} // namespace

TrackLine::TrackLine(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, const Options &options, LineExtractor line_extractor)
    : cameras_(std::move(cameras)), options_(options), line_extractor_(std::move(line_extractor)), database_(std::make_shared<LineFeatureDatabase>()) {
  if (!line_extractor_) {
    throw std::invalid_argument("TrackLine requires a line detector backend (see LineDetectorFactory)");
  }
}

void TrackLine::feed_new_camera(const CameraData &message) {
  if (message.sensor_ids.size() != message.images.size()) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t index = 0; index < message.images.size(); ++index) {
    if (message.images[index].empty()) {
      continue;
    }
    const size_t camera_id = static_cast<size_t>(message.sensor_ids[index]);
    cv::Mat gray;
    if (message.images[index].channels() == 1) {
      gray = message.images[index];
    } else {
      cv::cvtColor(message.images[index], gray, cv::COLOR_BGR2GRAY);
    }
    const cv::Mat mask = index < message.masks.size() ? message.masks[index] : cv::Mat();
    std::vector<Candidate> current_lines = detect_lines(gray, mask);
    auto &state = camera_states_[camera_id];
    if (!state.previous_image.empty()) {
      match_lines(state.previous_image, gray, state.previous_lines, current_lines);
    }

    std::vector<cv::Vec4f> visible_lines;
    std::vector<size_t> visible_ids;
    visible_lines.reserve(current_lines.size());
    visible_ids.reserve(current_lines.size());
    for (Candidate &line : current_lines) {
      if (line.id == 0) {
        line.id = next_id_.fetch_add(1);
      }
      database_->update_feature(line.id, make_observation(camera_id, message.timestamp, line));
      visible_lines.emplace_back(line.start.x, line.start.y, line.end.x, line.end.y);
      visible_ids.push_back(line.id);
    }
    state.previous_image = gray.clone();
    state.previous_lines = std::move(current_lines);
    last_lines_[camera_id] = std::move(visible_lines);
    last_ids_[camera_id] = std::move(visible_ids);
  }
}

std::vector<TrackLine::Candidate> TrackLine::detect_lines(const cv::Mat &image, const cv::Mat &mask) {
  std::vector<Candidate> lines;
  const std::vector<DetectedLine> detected_lines = line_extractor_(image, mask);
  lines.reserve(detected_lines.size());
  for (const DetectedLine &detected : detected_lines) {
    Candidate candidate;
    candidate.start = cv::Point2f(detected.endpoints[0], detected.endpoints[1]);
    candidate.end = cv::Point2f(detected.endpoints[2], detected.endpoints[3]);
    candidate.descriptor = detected.descriptor.clone();
    if (cv::norm(candidate.end - candidate.start) < options_.min_line_length) {
      continue;
    }
    if (!mask.empty()) {
      const cv::Point2f midpoint = 0.5f * (candidate.start + candidate.end);
      const int x = cvRound(midpoint.x);
      const int y = cvRound(midpoint.y);
      if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows || mask.at<unsigned char>(y, x) == 0) {
        continue;
      }
    }
    lines.push_back(std::move(candidate));
  }
  std::sort(lines.begin(), lines.end(), [](const Candidate &first, const Candidate &second) {
    return cv::norm(first.end - first.start) > cv::norm(second.end - second.start);
  });
  if (lines.size() > options_.max_lines) {
    lines.resize(options_.max_lines);
  }
  return lines;
}

void TrackLine::match_lines(const cv::Mat &previous_image, const cv::Mat &current_image, const std::vector<Candidate> &previous_lines,
                            std::vector<Candidate> &current_lines) {
  std::vector<bool> previous_used(previous_lines.size(), false);
  std::vector<bool> current_used(current_lines.size(), false);

  // 1) Descriptor (Hamming) matching. Selected per matching pair: only lines that carry a
  //    descriptor on *both* sides participate; anything else falls through to geometry.
  match_descriptor(previous_lines, current_lines, previous_used, current_used);

  // 2) Geometric (endpoint optical-flow) matching for everything still unmatched. This is the
  //    fallback for descriptor-less backends (FLD) and for the one-sided-descriptor-missing case.
  match_geometric(previous_image, current_image, previous_lines, current_lines, previous_used, current_used);
}

void TrackLine::match_descriptor(const std::vector<Candidate> &previous_lines, std::vector<Candidate> &current_lines,
                                 std::vector<bool> &previous_used, std::vector<bool> &current_used) {
  std::vector<int> previous_indices;
  std::vector<int> current_indices;
  std::vector<cv::Mat> previous_descriptors;
  std::vector<cv::Mat> current_descriptors;
  for (size_t index = 0; index < previous_lines.size(); ++index) {
    if (!previous_lines[index].descriptor.empty()) {
      previous_indices.push_back(static_cast<int>(index));
      previous_descriptors.push_back(previous_lines[index].descriptor);
    }
  }
  for (size_t index = 0; index < current_lines.size(); ++index) {
    if (!current_lines[index].descriptor.empty()) {
      current_indices.push_back(static_cast<int>(index));
      current_descriptors.push_back(current_lines[index].descriptor);
    }
  }
  if (previous_descriptors.empty() || current_descriptors.empty()) {
    return;
  }
  cv::Mat previous_matrix, current_matrix;
  cv::vconcat(previous_descriptors, previous_matrix);
  cv::vconcat(current_descriptors, current_matrix);
  std::vector<std::vector<cv::DMatch>> matches;
  cv::BFMatcher(cv::NORM_HAMMING).knnMatch(previous_matrix, current_matrix, matches, 2);
  for (const auto &neighbors : matches) {
    if (neighbors.empty() || (neighbors.size() > 1 && neighbors[0].distance >= options_.descriptor_ratio * neighbors[1].distance)) {
      continue;
    }
    const size_t previous_index = static_cast<size_t>(previous_indices[neighbors[0].queryIdx]);
    const size_t current_index = static_cast<size_t>(current_indices[neighbors[0].trainIdx]);
    if (!current_used[current_index] && geometry_compatible(previous_lines[previous_index], current_lines[current_index], options_.max_endpoint_motion)) {
      current_lines[current_index].id = previous_lines[previous_index].id;
      previous_used[previous_index] = true;
      current_used[current_index] = true;
    }
  }
}

void TrackLine::match_geometric(const cv::Mat &previous_image, const cv::Mat &current_image, const std::vector<Candidate> &previous_lines,
                                std::vector<Candidate> &current_lines, std::vector<bool> &previous_used, std::vector<bool> &current_used) {
  std::vector<cv::Point2f> previous_endpoints;
  previous_endpoints.reserve(previous_lines.size() * 2);
  for (const Candidate &line : previous_lines) {
    previous_endpoints.push_back(line.start);
    previous_endpoints.push_back(line.end);
  }
  if (previous_endpoints.empty()) {
    return;
  }
  std::vector<cv::Point2f> tracked_endpoints;
  std::vector<unsigned char> status;
  std::vector<float> errors;
  cv::calcOpticalFlowPyrLK(previous_image, current_image, previous_endpoints, tracked_endpoints, status, errors, cv::Size(21, 21), 3);

  struct Match {
    double cost;
    size_t previous_index;
    size_t current_index;
  };
  std::vector<Match> candidates;
  for (size_t previous_index = 0; previous_index < previous_lines.size(); ++previous_index) {
    if (previous_used[previous_index]) {
      continue;
    }
    const size_t endpoint_index = previous_index * 2;
    if (endpoint_index + 1 >= status.size() || !status[endpoint_index] || !status[endpoint_index + 1]) {
      continue;
    }
    Candidate predicted = previous_lines[previous_index];
    predicted.start = tracked_endpoints[endpoint_index];
    predicted.end = tracked_endpoints[endpoint_index + 1];
    for (size_t current_index = 0; current_index < current_lines.size(); ++current_index) {
      if (current_used[current_index]) {
        continue;
      }
      if (geometry_compatible(predicted, current_lines[current_index], options_.max_endpoint_motion)) {
        candidates.push_back({endpoint_distance(predicted.start, predicted.end, current_lines[current_index].start, current_lines[current_index].end),
                              previous_index, current_index});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Match &first, const Match &second) { return first.cost < second.cost; });
  for (const Match &match : candidates) {
    if (!previous_used[match.previous_index] && !current_used[match.current_index]) {
      current_lines[match.current_index].id = previous_lines[match.previous_index].id;
      previous_used[match.previous_index] = true;
      current_used[match.current_index] = true;
    }
  }
}

bool TrackLine::geometry_compatible(const Candidate &first, const Candidate &second, double max_motion) const {
  if (angle_difference(line_angle(first.start, first.end), line_angle(second.start, second.end)) >
      options_.max_angle_difference_deg * CV_PI / 180.0) {
    return false;
  }
  const double length_first = cv::norm(first.end - first.start);
  const double length_second = cv::norm(second.end - second.start);
  if (std::min(length_first, length_second) < 0.5 * std::max(length_first, length_second)) {
    return false;
  }
  return endpoint_distance(first.start, first.end, second.start, second.end) <= max_motion;
}

LineObservation TrackLine::make_observation(size_t camera_id, double timestamp, const Candidate &line) const {
  LineObservation observation;
  observation.timestamp = timestamp;
  observation.camera_id = camera_id;
  observation.endpoints_px << line.start.x, line.start.y, line.end.x, line.end.y;
  Eigen::Vector2d first(line.start.x, line.start.y);
  Eigen::Vector2d second(line.end.x, line.end.y);
  const auto camera = cameras_.find(camera_id);
  if (camera != cameras_.end() && camera->second) {
    first = camera->second->undistort_d(first);
    second = camera->second->undistort_d(second);
  }
  observation.endpoints_norm << first.x(), first.y(), second.x(), second.y();
  observation.line_norm = Eigen::Vector3d(first.x(), first.y(), 1.0).cross(Eigen::Vector3d(second.x(), second.y(), 1.0));
  const double normal_length = observation.line_norm.head<2>().norm();
  if (normal_length > std::numeric_limits<double>::epsilon()) {
    observation.line_norm /= normal_length;
  }
  return observation;
}

std::unordered_map<size_t, std::vector<cv::Vec4f>> TrackLine::get_last_lines() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_lines_;
}

std::unordered_map<size_t, std::vector<size_t>> TrackLine::get_last_ids() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_ids_;
}

} // namespace ov_core