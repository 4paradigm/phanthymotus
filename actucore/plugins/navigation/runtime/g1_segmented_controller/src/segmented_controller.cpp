#include "g1_segmented_controller/segmented_controller.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "angles/angles.h"
#include "g1_segmented_controller/segment_math.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"

namespace g1_segmented_controller
{

void SegmentedController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer>,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = parent.lock();
  if (!node) {
    throw std::runtime_error("controller parent lifecycle node is unavailable");
  }
  logger_ = node->get_logger();
  clock_ = node->get_clock();
  costmap_ros_ = std::move(costmap_ros);
  collision_checker_ = std::make_unique<
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>(
    costmap_ros_->getCostmap());

  const auto parameter = [&name](const std::string & suffix) {
      return name + "." + suffix;
    };
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("rotate_exit_rad"), rclcpp::ParameterValue(rotate_exit_rad_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("rotate_reengage_rad"), rclcpp::ParameterValue(rotate_reengage_rad_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("rotate_speed_rps"), rclcpp::ParameterValue(rotate_speed_rps_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("approach_distance_m"), rclcpp::ParameterValue(approach_distance_m_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("approach_speed_mps"), rclcpp::ParameterValue(approach_speed_mps_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("cruise_speed_mps"), rclcpp::ParameterValue(cruise_speed_mps_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("segment_tolerance_m"), rclcpp::ParameterValue(segment_tolerance_m_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("final_yaw_tolerance_rad"),
    rclcpp::ParameterValue(final_yaw_tolerance_rad_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("stop_linear_mps"), rclcpp::ParameterValue(stop_linear_mps_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("stop_yaw_rps"), rclcpp::ParameterValue(stop_yaw_rps_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("stop_cycles_required"), rclcpp::ParameterValue(stop_cycles_required_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("preserve_endpoint_m"), rclcpp::ParameterValue(preserve_endpoint_m_));
  nav2_util::declare_parameter_if_not_declared(
    node, parameter("status_topic"),
    rclcpp::ParameterValue("/ubuntu/navigation/nav2/segment_status"));

  node->get_parameter(parameter("rotate_exit_rad"), rotate_exit_rad_);
  node->get_parameter(parameter("rotate_reengage_rad"), rotate_reengage_rad_);
  node->get_parameter(parameter("rotate_speed_rps"), rotate_speed_rps_);
  node->get_parameter(parameter("approach_distance_m"), approach_distance_m_);
  node->get_parameter(parameter("approach_speed_mps"), approach_speed_mps_);
  node->get_parameter(parameter("cruise_speed_mps"), cruise_speed_mps_);
  node->get_parameter(parameter("segment_tolerance_m"), segment_tolerance_m_);
  node->get_parameter(parameter("final_yaw_tolerance_rad"), final_yaw_tolerance_rad_);
  node->get_parameter(parameter("stop_linear_mps"), stop_linear_mps_);
  node->get_parameter(parameter("stop_yaw_rps"), stop_yaw_rps_);
  node->get_parameter(parameter("stop_cycles_required"), stop_cycles_required_);
  node->get_parameter(parameter("preserve_endpoint_m"), preserve_endpoint_m_);
  const auto status_topic = node->get_parameter(parameter("status_topic")).as_string();

  if (!(rotate_exit_rad_ > 0.0 && rotate_reengage_rad_ > rotate_exit_rad_ &&
    rotate_speed_rps_ > 0.0 && approach_distance_m_ > segment_tolerance_m_ &&
    approach_speed_mps_ > 0.0 && cruise_speed_mps_ >= approach_speed_mps_ &&
    segment_tolerance_m_ > 0.0 && final_yaw_tolerance_rad_ > 0.0 &&
    stop_linear_mps_ >= 0.0 && stop_yaw_rps_ >= 0.0 && stop_cycles_required_ > 0 &&
    preserve_endpoint_m_ >= segment_tolerance_m_ && !status_topic.empty()))
  {
    throw std::runtime_error("invalid stop-turn-drive controller parameters");
  }
  speed_limit_mps_ = cruise_speed_mps_;
  status_pub_ = node->create_publisher<std_msgs::msg::String>(
    status_topic, rclcpp::QoS(1).best_effort().durability_volatile());
}

void SegmentedController::cleanup()
{
  std::lock_guard<std::mutex> lock(mutex_);
  status_pub_.reset();
  collision_checker_.reset();
  costmap_ros_.reset();
  plan_.poses.clear();
  has_segment_ = false;
}

void SegmentedController::activate()
{
  if (status_pub_) {
    status_pub_->on_activate();
  }
}

void SegmentedController::deactivate()
{
  if (status_pub_) {
    status_pub_->on_deactivate();
  }
}

void SegmentedController::setPlan(const nav_msgs::msg::Path & path)
{
  std::lock_guard<std::mutex> lock(mutex_);
  plan_ = path;
  if (plan_.poses.empty()) {
    has_segment_ = false;
    phase_ = Phase::BLOCKED;
    stop_reason_ = "empty_plan";
    return;
  }

  bool preserve = false;
  if (phase_ != Phase::BLOCKED && has_segment_) {
    for (std::size_t index = 0; index < plan_.poses.size(); ++index) {
      const auto & pose = plan_.poses[index].pose.position;
      if (std::hypot(pose.x - segment_end_x_, pose.y - segment_end_y_) <= preserve_endpoint_m_) {
        segment_index_ = index;
        segment_is_final_ = index + 1 == plan_.poses.size();
        preserve = true;
        break;
      }
    }
  }
  if (!preserve) {
    has_segment_ = false;
    enterStopCheck(Phase::SELECT, "path_updated");
  }
}

geometry_msgs::msg::TwistStamped SegmentedController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker *)
{
  std::lock_guard<std::mutex> state_lock(mutex_);
  geometry_msgs::msg::TwistStamped command;
  command.header.stamp = clock_->now();
  command.header.frame_id = costmap_ros_->getBaseFrameID();

  const double x = pose.pose.position.x;
  const double y = pose.pose.position.y;
  const double yaw = tf2::getYaw(pose.pose.orientation);
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(yaw)) {
    phase_ = Phase::BLOCKED;
    stop_reason_ = "pose_non_finite";
    publishStatus(pose, 0.0, 0.0, true);
    return command;
  }

  auto * costmap = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> costmap_lock(
    *costmap->getMutex());

  double remaining = has_segment_ ? std::hypot(segment_end_x_ - x, segment_end_y_ - y) : 0.0;
  double heading_error = has_segment_ ? angles::shortest_angular_distance(yaw, segment_yaw_) : 0.0;

  for (int transitions = 0; transitions < 3; ++transitions) {
    if (phase_ == Phase::STOP_CHECK) {
      if (std::hypot(velocity.linear.x, velocity.linear.y) <= stop_linear_mps_ &&
        std::abs(velocity.angular.z) <= stop_yaw_rps_)
      {
        ++stopped_cycles_;
      } else {
        stopped_cycles_ = 0;
      }
      if (stopped_cycles_ >= stop_cycles_required_) {
        phase_ = after_stop_;
        stopped_cycles_ = 0;
      }
      publishStatus(pose, remaining, heading_error);
      return command;
    }

    if (phase_ == Phase::SELECT) {
      const auto & goal = plan_.poses.back().pose;
      if (std::hypot(goal.position.x - x, goal.position.y - y) <= segment_tolerance_m_) {
        has_segment_ = false;
        phase_ = Phase::FINAL_ROTATE;
        continue;
      }
      if (!selectSegment(pose)) {
        phase_ = Phase::BLOCKED;
        stop_reason_ = "no_collision_free_segment";
        publishStatus(pose, 0.0, 0.0, true);
        return command;
      }
      remaining = std::hypot(segment_end_x_ - x, segment_end_y_ - y);
      heading_error = angles::shortest_angular_distance(yaw, segment_yaw_);
      phase_ = Phase::ROTATE;
      continue;
    }

    if (phase_ == Phase::ROTATE) {
      heading_error = angles::shortest_angular_distance(yaw, segment_yaw_);
      if (std::abs(heading_error) <= rotate_exit_rad_) {
        enterStopCheck(Phase::DRIVE, "heading_aligned");
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      if (!rotationSafe(x, y, yaw, segment_yaw_)) {
        phase_ = Phase::BLOCKED;
        stop_reason_ = "rotation_in_collision";
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      command.twist.angular.z = std::copysign(rotate_speed_rps_, heading_error);
      publishStatus(pose, remaining, heading_error);
      return command;
    }

    if (phase_ == Phase::DRIVE) {
      const auto progress = segment_progress(
        segment_start_x_, segment_start_y_, segment_end_x_, segment_end_y_, x, y);
      remaining = std::hypot(segment_end_x_ - x, segment_end_y_ - y);
      heading_error = angles::shortest_angular_distance(yaw, segment_yaw_);
      if (segment_crossed(progress)) {
        has_segment_ = false;
        enterStopCheck(Phase::SELECT, "segment_crossed");
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      if (segment_reached(progress, remaining, segment_tolerance_m_)) {
        has_segment_ = false;
        enterStopCheck(segment_is_final_ ? Phase::FINAL_ROTATE : Phase::SELECT, "segment_reached");
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      if (!straightPathSafe(x, y, segment_yaw_, segment_end_x_, segment_end_y_)) {
        phase_ = Phase::BLOCKED;
        stop_reason_ = "segment_blocked";
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      if (std::abs(heading_error) > rotate_reengage_rad_) {
        enterStopCheck(Phase::ROTATE, "heading_drifted");
        publishStatus(pose, remaining, heading_error, true);
        return command;
      }
      const double desired = remaining <= approach_distance_m_ ? approach_speed_mps_ : cruise_speed_mps_;
      command.twist.linear.x = std::min(desired, speed_limit_mps_);
      publishStatus(pose, remaining, heading_error);
      return command;
    }

    if (phase_ == Phase::FINAL_ROTATE) {
      const double goal_yaw = tf2::getYaw(plan_.poses.back().pose.orientation);
      heading_error = angles::shortest_angular_distance(yaw, goal_yaw);
      if (std::abs(heading_error) <= final_yaw_tolerance_rad_) {
        phase_ = Phase::ARRIVED;
        stop_reason_ = "goal_tolerance_reached";
        publishStatus(pose, 0.0, heading_error, true);
        return command;
      }
      if (!rotationSafe(x, y, yaw, goal_yaw)) {
        phase_ = Phase::BLOCKED;
        stop_reason_ = "final_rotation_in_collision";
        publishStatus(pose, 0.0, heading_error, true);
        return command;
      }
      command.twist.angular.z = std::copysign(rotate_speed_rps_, heading_error);
      publishStatus(pose, 0.0, heading_error);
      return command;
    }

    publishStatus(pose, remaining, heading_error);
    return command;
  }

  phase_ = Phase::BLOCKED;
  stop_reason_ = "transition_limit";
  publishStatus(pose, remaining, heading_error, true);
  return command;
}

void SegmentedController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (!std::isfinite(speed_limit) || speed_limit <= 0.0) {
    speed_limit_mps_ = cruise_speed_mps_;
    return;
  }
  const double absolute = percentage ? cruise_speed_mps_ * speed_limit / 100.0 : speed_limit;
  speed_limit_mps_ = std::clamp(absolute, 0.0, cruise_speed_mps_);
}

bool SegmentedController::selectSegment(const geometry_msgs::msg::PoseStamped & pose)
{
  if (plan_.poses.size() < 2) {
    return false;
  }
  const double x = pose.pose.position.x;
  const double y = pose.pose.position.y;
  std::size_t nearest = 0;
  double nearest_distance = std::numeric_limits<double>::max();
  for (std::size_t index = 0; index < plan_.poses.size(); ++index) {
    const auto & point = plan_.poses[index].pose.position;
    const double distance = std::hypot(point.x - x, point.y - y);
    if (distance < nearest_distance) {
      nearest = index;
      nearest_distance = distance;
    }
  }

  bool found = false;
  std::size_t selected = nearest;
  for (std::size_t index = nearest + 1; index < plan_.poses.size(); ++index) {
    const auto & point = plan_.poses[index].pose.position;
    const double distance = std::hypot(point.x - x, point.y - y);
    if (distance <= segment_tolerance_m_) {
      continue;
    }
    const double segment_yaw = std::atan2(point.y - y, point.x - x);
    if (!straightPathSafe(x, y, segment_yaw, point.x, point.y)) {
      break;
    }
    found = true;
    selected = index;
  }
  if (!found) {
    return false;
  }

  const auto & endpoint = plan_.poses[selected].pose.position;
  segment_start_x_ = x;
  segment_start_y_ = y;
  segment_end_x_ = endpoint.x;
  segment_end_y_ = endpoint.y;
  segment_yaw_ = std::atan2(segment_end_y_ - y, segment_end_x_ - x);
  segment_index_ = selected;
  segment_is_final_ = selected + 1 == plan_.poses.size();
  has_segment_ = true;
  stop_reason_.clear();
  return true;
}

bool SegmentedController::straightPathSafe(
  double x, double y, double yaw, double end_x, double end_y)
{
  const double distance = std::hypot(end_x - x, end_y - y);
  const double step = std::max(0.025, costmap_ros_->getCostmap()->getResolution() * 0.5);
  const int samples = std::max(1, static_cast<int>(std::ceil(distance / step)));
  const auto footprint = costmap_ros_->getRobotFootprint();
  for (int sample = 0; sample <= samples; ++sample) {
    const double ratio = static_cast<double>(sample) / samples;
    const double cost = collision_checker_->footprintCostAtPose(
      x + (end_x - x) * ratio, y + (end_y - y) * ratio, yaw, footprint);
    if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
      return false;
    }
  }
  return true;
}

bool SegmentedController::rotationSafe(
  double x, double y, double from_yaw, double to_yaw)
{
  const double delta = angles::shortest_angular_distance(from_yaw, to_yaw);
  const int samples = std::max(1, static_cast<int>(std::ceil(std::abs(delta) / 0.05)));
  const auto footprint = costmap_ros_->getRobotFootprint();
  for (int sample = 0; sample <= samples; ++sample) {
    const double yaw = from_yaw + delta * static_cast<double>(sample) / samples;
    const double cost = collision_checker_->footprintCostAtPose(x, y, yaw, footprint);
    if (cost >= nav2_costmap_2d::LETHAL_OBSTACLE) {
      return false;
    }
  }
  return true;
}

void SegmentedController::enterStopCheck(Phase next, const std::string & reason)
{
  phase_ = Phase::STOP_CHECK;
  after_stop_ = next;
  stopped_cycles_ = 0;
  stop_reason_ = reason;
}

void SegmentedController::publishStatus(
  const geometry_msgs::msg::PoseStamped & pose,
  double remaining_distance,
  double heading_error,
  bool force)
{
  if (!status_pub_ || !status_pub_->is_activated()) {
    return;
  }
  const auto now = clock_->now();
  if (!force && phase_ == last_status_phase_ && (now - last_status_time_).seconds() < 0.2) {
    return;
  }
  const auto & stamp = pose.header.stamp;
  const std::int64_t pose_stamp_ns =
    static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
  std::ostringstream json;
  json << "{\"phase\":\"" << phaseName(phase_) << "\",\"segment_index\":"
       << (has_segment_ ? static_cast<long long>(segment_index_) : -1)
       << ",\"remaining_distance_m\":" << std::max(0.0, remaining_distance)
       << ",\"heading_error_rad\":" << heading_error
       << ",\"stop_reason\":\"" << stop_reason_
       << "\",\"pose_stamp_ns\":" << pose_stamp_ns << "}";
  std_msgs::msg::String message;
  message.data = json.str();
  status_pub_->publish(message);
  last_status_phase_ = phase_;
  last_status_time_ = now;
}

const char * SegmentedController::phaseName(Phase phase)
{
  switch (phase) {
    case Phase::SELECT: return "SELECT";
    case Phase::STOP_CHECK: return "STOP_CHECK";
    case Phase::ROTATE: return "ROTATE";
    case Phase::DRIVE: return "DRIVE";
    case Phase::FINAL_ROTATE: return "FINAL_ROTATE";
    case Phase::ARRIVED: return "ARRIVED";
    case Phase::BLOCKED: return "BLOCKED";
  }
  return "BLOCKED";
}

}  // namespace g1_segmented_controller

PLUGINLIB_EXPORT_CLASS(g1_segmented_controller::SegmentedController, nav2_core::Controller)
