#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <moveit_msgs/srv/get_position_ik.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include "ttt_interfaces/msg/game_snapshot.hpp"
#include "ttt_interfaces/msg/turn_plan.hpp"
#include "ttt_interfaces/msg/workspace_layout.hpp"
#include "ttt_interfaces/srv/plan_turn.hpp"
#include "ttt_interfaces/srv/register_player.hpp"

using namespace std::chrono_literals;

namespace {

std::vector<std::string> panda_joint_names() {
  return {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"};
}

const std::vector<double> kHomePositions = {
    0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};

constexpr uint8_t kEmpty = ttt_interfaces::msg::GameSnapshot::EMPTY;

const std::array<std::array<int, 3>, 8> kWinningLines = {{
    {{0, 1, 2}},
    {{3, 4, 5}},
    {{6, 7, 8}},
    {{0, 3, 6}},
    {{1, 4, 7}},
    {{2, 5, 8}},
    {{0, 4, 8}},
    {{2, 4, 6}},
}};

const std::array<uint8_t, 4> kCorners = {{0, 2, 6, 8}};
const std::array<uint8_t, 4> kSides = {{1, 3, 5, 7}};
const std::array<uint8_t, 9> kMinimaxTieBreakOrder = {{
    4, 0, 2, 6, 8, 1, 3, 5, 7,
}};

using Board = std::array<uint8_t, 9>;
using LegalMask = std::array<uint8_t, 9>;

uint8_t check_winner(const Board &board) {
  for (const auto &line : kWinningLines) {
    const uint8_t first = board[static_cast<size_t>(line[0])];
    if (first != kEmpty &&
        first == board[static_cast<size_t>(line[1])] &&
        first == board[static_cast<size_t>(line[2])]) {
      return first;
    }
  }
  return kEmpty;
}

bool board_full(const Board &board) {
  return std::all_of(board.begin(), board.end(), [](uint8_t value) {
    return value != kEmpty;
  });
}

bool board_empty(const Board &board) {
  return std::all_of(board.begin(), board.end(), [](uint8_t value) {
    return value == kEmpty;
  });
}

bool is_legal_cell(const LegalMask &legal_actions, uint8_t cell_id) {
  return cell_id < legal_actions.size() && legal_actions[cell_id] == 1;
}

std::vector<uint8_t> legal_cells_from_snapshot(
    const ttt_interfaces::msg::GameSnapshot &snapshot) {
  std::vector<uint8_t> legal_cells;
  legal_cells.reserve(snapshot.legal_actions.size());
  for (size_t index = 0; index < snapshot.legal_actions.size(); ++index) {
    if (snapshot.legal_actions[index] == 1 &&
        snapshot.board[index] == kEmpty) {
      legal_cells.push_back(static_cast<uint8_t>(index));
    }
  }
  return legal_cells;
}

bool contains_cell(const std::vector<uint8_t> &cells, uint8_t cell_id) {
  return std::find(cells.begin(), cells.end(), cell_id) != cells.end();
}

std::vector<uint8_t> prioritized_legal_cells(
    const std::vector<uint8_t> &legal_cells) {
  std::vector<uint8_t> ordered;
  ordered.reserve(legal_cells.size());
  for (uint8_t cell_id : kMinimaxTieBreakOrder) {
    if (contains_cell(legal_cells, cell_id)) {
      ordered.push_back(cell_id);
    }
  }
  for (uint8_t cell_id : legal_cells) {
    if (!contains_cell(ordered, cell_id)) {
      ordered.push_back(cell_id);
    }
  }
  return ordered;
}

std::optional<uint8_t> fallback_cell(const std::vector<uint8_t> &legal_cells) {
  if (contains_cell(legal_cells, 4)) {
    return 4;
  }
  for (uint8_t cell_id : kCorners) {
    if (contains_cell(legal_cells, cell_id)) {
      return cell_id;
    }
  }
  for (uint8_t cell_id : kSides) {
    if (contains_cell(legal_cells, cell_id)) {
      return cell_id;
    }
  }
  if (!legal_cells.empty()) {
    return legal_cells.front();
  }
  return std::nullopt;
}

std::optional<uint8_t> find_immediate_winning_move(
    const Board &board,
    const std::vector<uint8_t> &legal_cells,
    uint8_t mark) {
  for (uint8_t cell_id : prioritized_legal_cells(legal_cells)) {
    Board candidate = board;
    candidate[cell_id] = mark;
    if (check_winner(candidate) == mark) {
      return cell_id;
    }
  }
  return std::nullopt;
}

int minimax(
    Board &board,
    int depth,
    bool my_turn,
    uint8_t my_mark,
    uint8_t opponent_mark) {
  const uint8_t winner = check_winner(board);
  if (winner == my_mark) {
    return 10 - depth;
  }
  if (winner == opponent_mark) {
    return depth - 10;
  }
  if (board_full(board)) {
    return 0;
  }

  if (my_turn) {
    int best_score = -1000;
    for (uint8_t cell_id : kMinimaxTieBreakOrder) {
      if (board[cell_id] != kEmpty) {
        continue;
      }
      board[cell_id] = my_mark;
      best_score = std::max(
          best_score,
          minimax(board, depth + 1, false, my_mark, opponent_mark));
      board[cell_id] = kEmpty;
    }
    return best_score;
  }

  int best_score = 1000;
  for (uint8_t cell_id : kMinimaxTieBreakOrder) {
    if (board[cell_id] != kEmpty) {
      continue;
    }
    board[cell_id] = opponent_mark;
    best_score = std::min(
        best_score,
        minimax(board, depth + 1, true, my_mark, opponent_mark));
    board[cell_id] = kEmpty;
  }
  return best_score;
}

std::optional<uint8_t> choose_best_cell(
    const ttt_interfaces::msg::GameSnapshot &snapshot,
    uint8_t my_mark,
    uint8_t opponent_mark) {
  Board board = snapshot.board;
  const auto legal_cells = legal_cells_from_snapshot(snapshot);
  if (legal_cells.empty()) {
    return std::nullopt;
  }

  if (auto winning_cell = find_immediate_winning_move(board, legal_cells, my_mark)) {
    return winning_cell;
  }
  if (auto blocking_cell =
          find_immediate_winning_move(board, legal_cells, opponent_mark)) {
    return blocking_cell;
  }
  if (board_empty(board) && contains_cell(legal_cells, 4)) {
    return 4;
  }

  int best_score = -1000;
  std::optional<uint8_t> best_cell;
  for (uint8_t cell_id : prioritized_legal_cells(legal_cells)) {
    if (!is_legal_cell(snapshot.legal_actions, cell_id) ||
        board[cell_id] != kEmpty) {
      continue;
    }
    board[cell_id] = my_mark;
    const int score = minimax(board, 1, false, my_mark, opponent_mark);
    board[cell_id] = kEmpty;

    if (!best_cell || score > best_score) {
      best_score = score;
      best_cell = cell_id;
    }
  }

  if (best_cell) {
    return best_cell;
  }
  return fallback_cell(legal_cells);
}

std::optional<uint8_t> choose_available_piece(
    const ttt_interfaces::msg::GameSnapshot &snapshot,
    uint8_t player_id) {
  std::optional<uint8_t> selected_piece;
  for (const auto &piece : snapshot.pieces) {
    if (!piece.available || piece.owner != player_id) {
      continue;
    }
    if (!selected_piece || piece.piece_id < *selected_piece) {
      selected_piece = piece.piece_id;
    }
  }
  return selected_piece;
}

std::vector<double> home_positions_from_layout(
    const ttt_interfaces::msg::WorkspaceLayout &layout) {
  const auto joint_names = panda_joint_names();
  std::vector<double> home_positions;
  home_positions.reserve(joint_names.size());

  for (const auto &joint_name : joint_names) {
    auto name_iter = std::find(
        layout.home_joint_state.name.begin(),
        layout.home_joint_state.name.end(),
        joint_name);
    if (name_iter == layout.home_joint_state.name.end()) {
      return kHomePositions;
    }

    const auto index = static_cast<size_t>(
        std::distance(layout.home_joint_state.name.begin(), name_iter));
    if (index >= layout.home_joint_state.position.size()) {
      return kHomePositions;
    }
    home_positions.push_back(layout.home_joint_state.position[index]);
  }

  return home_positions;
}

// link8 orientation quaternion (x, y, z, w) for gripper pointing down
constexpr double kLink8QuatX = 0.9238795325112867;
constexpr double kLink8QuatY = -0.3826834323650898;
constexpr double kLink8QuatZ = 0.0;
constexpr double kLink8QuatW = 0.0;
// Fixed Z offset from panda_link8 to panda_hand_tcp
constexpr double kLink8TcpZOffset = 0.1034;

geometry_msgs::msg::Pose link8_pose_from_tcp_target(double x, double y, double z) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z + kLink8TcpZOffset;
  pose.orientation.x = kLink8QuatX;
  pose.orientation.y = kLink8QuatY;
  pose.orientation.z = kLink8QuatZ;
  pose.orientation.w = kLink8QuatW;
  return pose;
}

trajectory_msgs::msg::JointTrajectoryPoint make_point(
    const std::vector<double> &positions,
    double time_sec) {
  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions = positions;
  const auto whole_seconds = static_cast<int32_t>(std::floor(time_sec));
  point.time_from_start.sec = whole_seconds;
  point.time_from_start.nanosec =
      static_cast<uint32_t>((time_sec - static_cast<double>(whole_seconds)) * 1e9);
  return point;
}

moveit_msgs::msg::RobotTrajectory make_three_point_trajectory(
    const std::vector<double> &start_positions,
    const std::vector<double> &end_positions,
    double end_time_sec) {
  moveit_msgs::msg::RobotTrajectory trajectory;
  trajectory.joint_trajectory.joint_names = panda_joint_names();

  const std::vector<double> midpoint = [&]() {
    std::vector<double> result;
    result.reserve(start_positions.size());
    for (size_t index = 0; index < start_positions.size(); ++index) {
      result.push_back((start_positions[index] + end_positions[index]) * 0.5);
    }
    return result;
  }();

  trajectory.joint_trajectory.points.push_back(make_point(start_positions, 0.0));
  trajectory.joint_trajectory.points.push_back(make_point(midpoint, end_time_sec * 0.5));
  trajectory.joint_trajectory.points.push_back(make_point(end_positions, end_time_sec));
  return trajectory;
}

}  // namespace

class StudentPlayerNode : public rclcpp::Node {
 public:
  StudentPlayerNode() : Node("student_player") {
    this->declare_parameter<std::string>("player_name", this->get_name());
    this->declare_parameter<std::string>("plan_turn_service", "/student_player/plan_turn");

    player_name_ = this->get_parameter("player_name").as_string();
    plan_turn_service_ = this->get_parameter("plan_turn_service").as_string();

    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    register_client_ =
        this->create_client<ttt_interfaces::srv::RegisterPlayer>("/ttt/register_player");

    ik_client_ = this->create_client<moveit_msgs::srv::GetPositionIK>(
        "/compute_ik",
        rmw_qos_profile_services_default,
        cb_group_);

    plan_turn_service_server_ = this->create_service<ttt_interfaces::srv::PlanTurn>(
        plan_turn_service_,
        std::bind(&StudentPlayerNode::handle_plan_turn, this, std::placeholders::_1,
                  std::placeholders::_2),
        rmw_qos_profile_services_default,
        cb_group_);

    register_timer_ =
        this->create_wall_timer(500ms, std::bind(&StudentPlayerNode::try_register, this));
  }

 private:
  void try_register() {
    if (registered_ || registration_in_flight_) {
      return;
    }
    if (!register_client_->wait_for_service(100ms)) {
      return;
    }

    auto request = std::make_shared<ttt_interfaces::srv::RegisterPlayer::Request>();
    request->player_name = player_name_;
    request->service_name = plan_turn_service_;
    registration_in_flight_ = true;

    register_client_->async_send_request(
        request,
        [this](rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedFuture future) {
          registration_in_flight_ = false;
          try {
            const auto response = future.get();
            if (!response->success) {
              RCLCPP_WARN(this->get_logger(), "Registration rejected: %s",
                          response->message.c_str());
              return;
            }
            registered_ = true;
            player_id_ = response->assigned_player_id;
            RCLCPP_INFO(this->get_logger(), "Registered as player_%u.", player_id_);
            register_timer_->cancel();
          } catch (const std::exception &exc) {
            RCLCPP_ERROR(this->get_logger(), "Registration failed: %s", exc.what());
          }
        });
  }

  // ----------------------------------------------------------------
  // IK helpers
  // ----------------------------------------------------------------

  std::optional<std::vector<double>> compute_ik(
      const geometry_msgs::msg::Pose &target_pose,
      const std::vector<double> &seed_positions) {
    const auto joint_names = panda_joint_names();
    if (seed_positions.size() != joint_names.size()) {
      RCLCPP_ERROR(this->get_logger(), "IK seed has %zu joints, expected %zu.",
                   seed_positions.size(), joint_names.size());
      return std::nullopt;
    }

    if (!ik_client_->wait_for_service(1s)) {
      RCLCPP_ERROR(this->get_logger(), "MoveIt /compute_ik service is unavailable.");
      return std::nullopt;
    }

    sensor_msgs::msg::JointState seed_state;
    seed_state.name = joint_names;
    seed_state.position = seed_positions;

    auto request = std::make_shared<moveit_msgs::srv::GetPositionIK::Request>();
    request->ik_request.group_name = "panda_arm";
    request->ik_request.robot_state.joint_state = seed_state;
    request->ik_request.pose_stamped.header.frame_id = "panda_link0";
    request->ik_request.pose_stamped.pose = target_pose;
    request->ik_request.timeout.sec = 5;
    request->ik_request.timeout.nanosec = 0;
    request->ik_request.avoid_collisions = false;

    using IkResponse = moveit_msgs::srv::GetPositionIK::Response::SharedPtr;
    auto promise = std::make_shared<std::promise<IkResponse>>();
    auto future = promise->get_future();

    try {
      ik_client_->async_send_request(
          request,
          [promise](rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedFuture result) {
            try {
              promise->set_value(result.get());
            } catch (...) {
              promise->set_exception(std::current_exception());
            }
          });
    } catch (const std::exception &exc) {
      RCLCPP_ERROR(this->get_logger(), "Failed to send IK request: %s", exc.what());
      return std::nullopt;
    }

    if (future.wait_for(6s) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "IK request timed out.");
      return std::nullopt;
    }

    IkResponse result;
    try {
      result = future.get();
    } catch (const std::exception &exc) {
      RCLCPP_ERROR(this->get_logger(), "IK request failed: %s", exc.what());
      return std::nullopt;
    }

    if (!result ||
        result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "IK failed with MoveIt error code %d.",
                   result ? result->error_code.val : 0);
      return std::nullopt;
    }

    std::vector<double> solution;
    solution.reserve(joint_names.size());
    const auto &solution_names = result->solution.joint_state.name;
    const auto &solution_positions = result->solution.joint_state.position;

    for (const auto &joint_name : joint_names) {
      auto name_iter = std::find(solution_names.begin(), solution_names.end(), joint_name);
      if (name_iter == solution_names.end()) {
        RCLCPP_ERROR(this->get_logger(), "IK solution is missing joint '%s'.",
                     joint_name.c_str());
        return std::nullopt;
      }

      const auto index = static_cast<size_t>(
          std::distance(solution_names.begin(), name_iter));
      if (index >= solution_positions.size()) {
        RCLCPP_ERROR(this->get_logger(), "IK solution has no position for joint '%s'.",
                     joint_name.c_str());
        return std::nullopt;
      }
      solution.push_back(solution_positions[index]);
    }

    return solution;
  }

  // ----------------------------------------------------------------
  // Turn planning
  // ----------------------------------------------------------------

  void handle_plan_turn(
      const std::shared_ptr<ttt_interfaces::srv::PlanTurn::Request> request,
      std::shared_ptr<ttt_interfaces::srv::PlanTurn::Response> response) {
    if (request->player_id != player_id_) {
      response->accepted = false;
      response->message = "Plan request does not match registered player id.";
      return;
    }

    const uint8_t my_mark = static_cast<uint8_t>(request->player_id + 1);
    const uint8_t opponent_mark = request->player_id == 0 ? 2u : 1u;

    auto cell_id = choose_best_cell(request->snapshot, my_mark, opponent_mark);
    if (!cell_id) {
      response->accepted = false;
      response->message = "No legal tic-tac-toe move available.";
      return;
    }
    if (!is_legal_cell(request->snapshot.legal_actions, *cell_id)) {
      response->accepted = false;
      response->message = "Selected cell is not legal in the current snapshot.";
      return;
    }

    auto piece_id = choose_available_piece(request->snapshot, request->player_id);
    if (!piece_id) {
      response->accepted = false;
      response->message = "No available piece owned by this player.";
      return;
    }

    auto piece_pose = find_piece_pose(request->snapshot, *piece_id);
    if (!piece_pose) {
      response->accepted = false;
      response->message = "Selected piece pose was not found in the snapshot.";
      return;
    }

    const auto &cell_pose = request->layout.cell_poses[*cell_id];
    const auto pick_target = link8_pose_from_tcp_target(
        piece_pose->position.x,
        piece_pose->position.y,
        piece_pose->position.z);
    const auto place_target = link8_pose_from_tcp_target(
        cell_pose.position.x,
        cell_pose.position.y,
        cell_pose.position.z);

    const auto home_positions = home_positions_from_layout(request->layout);
    const auto pick_goal = compute_ik(pick_target, home_positions);
    if (!pick_goal) {
      response->accepted = false;
      response->message = "IK failed for pick target.";
      return;
    }

    const auto place_goal = compute_ik(place_target, home_positions);
    if (!place_goal) {
      response->accepted = false;
      response->message = "IK failed for place target.";
      return;
    }

    ttt_interfaces::msg::TurnPlan plan;
    plan.match_id = request->match_id;
    plan.turn_index = request->turn_index;
    plan.player_id = request->player_id;
    plan.piece_id = *piece_id;
    plan.cell_id = *cell_id;
    plan.home_to_pick = make_three_point_trajectory(home_positions, *pick_goal, 1.5);
    plan.pick_to_home = make_three_point_trajectory(*pick_goal, home_positions, 1.5);
    plan.home_to_place = make_three_point_trajectory(home_positions, *place_goal, 1.5);
    plan.place_to_home = make_three_point_trajectory(*place_goal, home_positions, 1.5);

    response->accepted = true;
    response->message = "Minimax IK-based plan generated.";
    response->plan = plan;

    RCLCPP_INFO(this->get_logger(), "Planned piece %u -> cell %u as player_%u.",
                static_cast<unsigned>(*piece_id),
                static_cast<unsigned>(*cell_id),
                static_cast<unsigned>(request->player_id));
  }

  static std::optional<geometry_msgs::msg::Pose> find_piece_pose(
      const ttt_interfaces::msg::GameSnapshot &snapshot,
      uint8_t piece_id) {
    for (const auto &piece : snapshot.pieces) {
      if (piece.piece_id == piece_id) {
        return piece.pose;
      }
    }
    return std::nullopt;
  }

  std::string player_name_;
  std::string plan_turn_service_;
  bool registered_{false};
  bool registration_in_flight_{false};
  uint8_t player_id_{255};

  rclcpp::CallbackGroup::SharedPtr cb_group_;
  rclcpp::Client<ttt_interfaces::srv::RegisterPlayer>::SharedPtr register_client_;
  rclcpp::Client<moveit_msgs::srv::GetPositionIK>::SharedPtr ik_client_;
  rclcpp::Service<ttt_interfaces::srv::PlanTurn>::SharedPtr plan_turn_service_server_;
  rclcpp::TimerBase::SharedPtr register_timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StudentPlayerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
