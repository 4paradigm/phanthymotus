#ifndef SEGMENTED_CONTROLLER__SEGMENTED_CONTROLLER_HPP_
#define SEGMENTED_CONTROLLER__SEGMENTED_CONTROLLER_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "std_msgs/msg/string.hpp"

namespace segmented_controller
{

class SegmentedController : public nav2_core::Controller
{
public:
  SegmentedController() = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;
  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void setPlan(const nav_msgs::msg::Path & path) override;
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  enum class Phase
  {
    SELECT,
    STOP_CHECK,
    ROTATE,
    DRIVE,
    FINAL_ROTATE,
    ARRIVED,
    BLOCKED,
  };

  bool selectSegment(const geometry_msgs::msg::PoseStamped & pose);
  bool straightPathSafe(double x, double y, double yaw, double end_x, double end_y);
  bool commandedForwardSafe(double x, double y, double yaw, double remaining);
  bool rotationSafe(double x, double y, double from_yaw, double to_yaw);
  void enterStopCheck(Phase next, const std::string & reason);
  struct OdometryState;
  static void receiveOdometry(
    const std::shared_ptr<OdometryState> & state,
    const nav_msgs::msg::Odometry::SharedPtr message);
  void publishStatus(
    const geometry_msgs::msg::PoseStamped & pose,
    double remaining_distance,
    double heading_error,
    bool force = false);
  static const char * phaseName(Phase phase);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Logger logger_{rclcpp::get_logger("segmented_controller")};
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::unique_ptr<
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>>
  collision_checker_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  // Callbacks own this state independently of the controller's lifetime.
  struct OdometryState
  {
    std::mutex mutex;
    geometry_msgs::msg::Twist velocity;
    std::chrono::steady_clock::time_point received{};
    std::int64_t stamp_ns{0};
    std::uint64_t motion_generation{0};
    bool valid{false};
    bool active{false};
    rclcpp::Clock::SharedPtr clock;
    std::string base_frame;
    double max_age_sec{0.6};
    double stop_linear_mps{0.05};
    double stop_yaw_rps{0.10};
  };
  std::shared_ptr<OdometryState> odom_state_;
  double odom_max_age_sec_{0.6};

  std::mutex mutex_;
  nav_msgs::msg::Path plan_;
  std::size_t segment_index_{0};
  bool has_segment_{false};
  bool segment_is_final_{false};
  double segment_start_x_{0.0};
  double segment_start_y_{0.0};
  double segment_end_x_{0.0};
  double segment_end_y_{0.0};
  double segment_yaw_{0.0};
  Phase phase_{Phase::STOP_CHECK};
  Phase after_stop_{Phase::SELECT};
  Phase last_status_phase_{Phase::BLOCKED};
  int stopped_cycles_{0};
  std::int64_t stopped_sample_stamp_ns_{0};
  std::int64_t first_stopped_stamp_ns_{0};
  std::uint64_t stopped_motion_generation_{0};
  std::string stop_reason_{"waiting_for_plan"};
  rclcpp::Time last_status_time_{0, 0, RCL_ROS_TIME};

  double rotate_exit_rad_{0.15};
  double rotate_reengage_rad_{0.35};
  double rotate_speed_rps_{1.0};
  double approach_distance_m_{0.50};
  double approach_speed_mps_{0.30};
  double cruise_speed_mps_{1.0};
  double segment_tolerance_m_{0.18};
  double final_yaw_tolerance_rad_{0.45};
  double stop_linear_mps_{0.05};
  double stop_yaw_rps_{0.10};
  int stop_cycles_required_{2};
  double stop_min_duration_sec_{0.3};
  double preserve_endpoint_m_{0.25};
  // A preview floor, not a validated stopping distance.
  double forward_preview_min_m_{0.50};
  double speed_limit_mps_{1.0};
};

}  // namespace segmented_controller

#endif  // SEGMENTED_CONTROLLER__SEGMENTED_CONTROLLER_HPP_
