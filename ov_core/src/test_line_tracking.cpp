#include <algorithm>
#include <cassert>
#include <memory>

#include "track/TrackLine.h"

namespace {

// Synthetic frame of 8 horizontal-ish lines. Frame 1 is frame 0 shifted by (4, 2) px, which
// exercises the optical-flow path (and is far below the default 80 px endpoint motion).
cv::Mat make_image(int frame_index) {
  cv::Mat image = cv::Mat::zeros(240, 320, CV_8UC1);
  for (int index = 0; index < 8; ++index) {
    const int y = 25 + index * 25;
    const int shift_x = frame_index == 0 ? 0 : 4;
    const int shift_y = frame_index == 0 ? 0 : 2;
    cv::line(image, cv::Point(25 + shift_x, y + shift_y), cv::Point(260 + shift_x, y + 5 + shift_y), cv::Scalar(255), 2);
  }
  return image;
}

// Extractor yielding 8 lines with a per-frame descriptor policy (both frames, neither frame, or
// only the first frame carry a 256-bit descriptor).
enum class DescriptorPolicy { BOTH, NONE, FIRST_ONLY };

ov_core::TrackLine::LineExtractor make_extractor(DescriptorPolicy policy) {
  auto frame_index = std::make_shared<size_t>(0);
  return [policy, frame_index](const cv::Mat &, const cv::Mat &) {
    std::vector<ov_core::TrackLine::DetectedLine> lines;
    for (int index = 0; index < 8; ++index) {
      ov_core::TrackLine::DetectedLine line;
      const float shift_x = *frame_index == 0 ? 0.0f : 4.0f;
      const float shift_y = *frame_index == 0 ? 0.0f : 2.0f;
      const float y = 25.0f + index * 25.0f + shift_y;
      line.endpoints = cv::Vec4f(25.0f + shift_x, y, 260.0f + shift_x, y + 5.0f);
      const bool first_frame = (*frame_index == 0);
      if (policy == DescriptorPolicy::BOTH || (policy == DescriptorPolicy::FIRST_ONLY && first_frame)) {
        line.descriptor = cv::Mat::zeros(1, 32, CV_8U);
        line.descriptor.setTo(static_cast<unsigned char>(index * 31));
      }
      lines.push_back(line);
    }
    ++(*frame_index);
    return lines;
  };
}

// Feeds two frames and reports whether at least one line keeps its id across both (a persistent
// track with two observations in the database).
bool persistent_track(const cv::Mat &first, const cv::Mat &second, ov_core::TrackLine::LineExtractor extractor) {
  ov_core::TrackLine::Options options;
  options.min_line_length = 30.0;
  ov_core::TrackLine tracker({}, options, extractor);

  ov_core::CameraData message;
  message.sensor_ids = {0};
  message.timestamp = 1.0;
  message.images = {first};
  tracker.feed_new_camera(message);
  const auto first_ids = tracker.get_last_ids().at(0);
  assert(!first_ids.empty());

  message.timestamp = 1.1;
  message.images = {second};
  tracker.feed_new_camera(message);
  const auto ids = tracker.get_last_ids().at(0);
  assert(!ids.empty());

  for (const size_t id : ids) {
    if (std::find(first_ids.begin(), first_ids.end(), id) == first_ids.end()) {
      continue;
    }
    ov_core::LineFeature feature;
    if (tracker.get_feature_database()->get_feature(id, feature) && feature.observations.at(0).size() == 2) {
      return true;
    }
  }
  return false;
}

} // namespace

int main() {
  const cv::Mat first = make_image(0);
  const cv::Mat second = make_image(1);

  // Descriptor matching: both frames carry descriptors, so the Hamming path runs.
  assert(persistent_track(first, second, make_extractor(DescriptorPolicy::BOTH)));

  // Geometric matching: no descriptors anywhere, so the endpoint optical-flow path runs.
  assert(persistent_track(first, second, make_extractor(DescriptorPolicy::NONE)));

  // One-sided descriptor missing: descriptor path has nothing to match, so it must fall back to
  // geometric matching rather than dropping the track.
  assert(persistent_track(first, second, make_extractor(DescriptorPolicy::FIRST_ONLY)));

  return 0;
}
