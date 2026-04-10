// Copyright (c) 2026, Marc Bloechlinger
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_COSTMAP_2D__ASYMMETRIC_INFLATION_LAYER_HPP_
#define NAV2_COSTMAP_2D__ASYMMETRIC_INFLATION_LAYER_HPP_

#include <map>
#include <vector>
#include <mutex>
#include <memory>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/inflation_layer_interface.hpp"
#include "nav2_costmap_2d/legacy_inflation_layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "nav_msgs/msg/path.hpp"

namespace nav2_costmap_2d
{

/**
 * @class AsymmetricInflationLayer
 * @brief A costmap layer that inflates obstacles asymmetrically relative to the global path.
 *
 * This layer extends the standard BFS inflation algorithm with path-aware side
 * classification.  By evaluating whether each obstacle lies to the left or right
 * of the global path's tangent, it modifies the effective distance used for BFS
 * queue priority and cost lookup.  Left-side obstacles receive a compressed
 * effective distance (higher cost, wider spread) while right-side obstacles
 * receive a stretched effective distance (lower cost, narrower spread).  The
 * result is a cost "valley" shifted toward the right side of corridors,
 * encouraging the local planner to keep right without any planner changes.
 *
 * Safety guarantee: the algorithm runs a symmetric BFS pass first, then an
 * asymmetric pass that can only increase costs.  Within the inscribed radius
 * the effective distance always equals the physical distance.
 *
 * When no path is available, the goal is nearby, or asymmetry_factor is zero,
 * the layer degrades gracefully to standard symmetric inflation identical to
 * LegacyInflationLayer.
 */
class AsymmetricInflationLayer : public InflationLayerInterface
{
public:
  /**
   * @brief Constructor
   */
  AsymmetricInflationLayer();

  /**
   * @brief Destructor
   */
  ~AsymmetricInflationLayer();

  /**
   * @brief Initialization process of layer on startup
   */
  void onInitialize() override;

  /**
   * @brief Activate the layer and register dynamic parameter callbacks
   */
  void activate() override;

  /**
   * @brief Deactivate the layer and unregister dynamic parameter callbacks
   */
  void deactivate() override;

  /**
   * @brief Update the bounds of the master costmap by this layer's update dimensions.
   *
   * Always requests full-map reinflation because asymmetric costs are
   * path-relative in world frame; when a rolling-window costmap shifts,
   * shifted cells carry stale asymmetric values.
   * @param robot_x X pose of robot
   * @param robot_y Y pose of robot
   * @param robot_yaw Robot orientation
   * @param min_x X min map coord of the window to update
   * @param min_y Y min map coord of the window to update
   * @param max_x X max map coord of the window to update
   * @param max_y Y max map coord of the window to update
   */
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw, double * min_x,
    double * min_y,
    double * max_x,
    double * max_y) override;

  /**
   * @brief Update the costs in the master costmap using two-pass BFS.
   *
   * Pass 1 writes a symmetric baseline (identical to LegacyInflationLayer).
   * Pass 2, when a valid path is available, re-runs BFS with asymmetric
   * effective distances and writes max(old, new) so costs can only increase.
   * @param master_grid The master costmap grid to update
   * @param min_i X min map coord of the window to update
   * @param min_j Y min map coord of the window to update
   * @param max_i X max map coord of the window to update
   * @param max_j Y max map coord of the window to update
   */
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  /**
   * @brief Match the size of the master costmap
   */
  void matchSize() override;

  /**
   * @brief If clearing operations should be processed on this layer or not
   */
  bool isClearable() override {return false;}

  /**
   * @brief Reset this costmap
   */
  void reset() override
  {
    matchSize();
    setCurrent(false);
    need_reinflation_ = true;
  }

  /**
   * @brief Given a distance, compute a cost.
   *
   * Uses pure exponential decay without an inscribed-radius plateau,
   * which produces smoother cost gradients for the asymmetric algorithm.
   * @param distance The distance from an obstacle in cells
   * @return A cost value for the distance
   */
  inline unsigned char computeCost(double distance) const override
  {
    unsigned char cost = 0;
    if (distance == 0) {
      cost = LETHAL_OBSTACLE;
    } else {
      double factor = exp(-1.0 * cost_scaling_factor_ * distance * resolution_);
      cost = static_cast<unsigned char>((INSCRIBED_INFLATED_OBSTACLE - 1) * factor);
    }
    return cost;
  }

  /**
   * @brief Get the mutex of the inflation information
   * @return Pointer to the mutex
   */
  mutex_t * getMutex() override
  {
    return access_;
  }

  /**
   * @brief Get the cost scaling factor
   * @return The cost scaling factor
   */
  double getCostScalingFactor() override
  {
    return cost_scaling_factor_;
  }

  /**
   * @brief Get the inflation radius
   * @return The inflation radius in meters
   */
  double getInflationRadius() override
  {
    return inflation_radius_;
  }

protected:
  /**
   * @brief Process updates on footprint changes to the inflation layer
   */
  void onFootprintChanged() override;

  /**
   * @brief Callback for incoming global path messages
   * @param msg The received path message
   */
  void globalPathCallback(const nav_msgs::msg::Path::SharedPtr msg);

  /**
   * @brief Extract the global path into the costmap's coordinate frame.
   *
   * Transforms the stored global path into the costmap frame via TF2,
   * filters to points inside the local costmap window, and returns an
   * empty vector when the robot is within goal_distance_threshold_ of
   * the goal (disabling asymmetry near the goal).
   * @param master_grid The master costmap for coordinate conversion
   * @return Path points in costmap frame, or empty if asymmetry should be disabled
   */
  std::vector<std::pair<double, double>> extractLocalPath(
    nav2_costmap_2d::Costmap2D & master_grid);

  /**
   * @brief Classify an obstacle cell as left (+1), right (-1), or neutral (0)
   * relative to the closest path segment.
   *
   * Projects the cell onto each path segment and uses the cross product
   * of the segment direction and the cell offset to determine the side.
   * Obstacles beyond neutral_threshold_ from the path are classified as neutral.
   * @param i X coordinate of the cell
   * @param j Y coordinate of the cell
   * @param local_path_pts Path points in costmap frame
   * @param master_grid The master costmap for coordinate conversion
   * @return +1 (left), -1 (right), or 0 (neutral)
   */
  int8_t computeObstacleSide(
    int i, int j,
    const std::vector<std::pair<double, double>> & local_path_pts,
    nav2_costmap_2d::Costmap2D & master_grid);

  /**
   * @brief Compute the asymmetric effective distance for BFS priority.
   *
   * Within the inscribed radius, returns the physical distance unchanged
   * to guarantee safety.  Beyond it, scales the excess distance by
   * (1 - asymmetry_factor * path_side), compressing left-side distances
   * and stretching right-side distances.
   * @param physical_dist Physical distance in cells
   * @param path_side Side classification: +1 (left), -1 (right), 0 (neutral)
   * @return Effective distance in cells
   */
  inline double getEffectiveDistance(double physical_dist, int8_t path_side) const
  {
    double inscribed_cells = inscribed_radius_ / resolution_;
    if (physical_dist <= inscribed_cells) {
      return physical_dist;
    }
    double excess = physical_dist - inscribed_cells;
    return inscribed_cells + excess * (1.0 - asymmetry_factor_ * path_side);
  }

  /**
   * @brief Lookup pre-computed Euclidean distance between two cells
   * @param mx The x coordinate of the current cell
   * @param my The y coordinate of the current cell
   * @param src_x The x coordinate of the source cell
   * @param src_y The y coordinate of the source cell
   * @return Euclidean distance in cells
   */
  inline double distanceLookup(
    unsigned int mx, unsigned int my,
    unsigned int src_x, unsigned int src_y)
  {
    unsigned int dx = (mx > src_x) ? mx - src_x : src_x - mx;
    unsigned int dy = (my > src_y) ? my - src_y : src_y - my;
    return cached_distances_[dx * cache_length_ + dy];
  }

  /**
   * @brief Convert a world distance (meters) to a cell distance
   * @param world_dist Distance in meters
   * @return Distance in cells
   */
  unsigned int cellDistance(double world_dist)
  {
    return layered_costmap_->getCostmap()->cellDistance(world_dist);
  }

  /**
   * @brief Pre-compute distance and cost caches and size the BFS priority queue
   */
  void computeCaches();

  /**
   * @brief Enqueue a cell into the BFS priority queue at the appropriate bin
   * @param index Flat index of the cell in the costmap
   * @param mx X coordinate of the cell
   * @param my Y coordinate of the cell
   * @param src_x X coordinate of the source obstacle
   * @param src_y Y coordinate of the source obstacle
   * @param path_side Side classification of the source obstacle
   */
  inline void enqueue(
    unsigned int index, unsigned int mx, unsigned int my,
    unsigned int src_x, unsigned int src_y, int8_t path_side);

  /**
   * @brief Validate incoming parameter updates before applying them.
   * @param parameters List of parameters being updated
   * @return Result indicating whether the update is accepted
   */
  rcl_interfaces::msg::SetParametersResult validateParameterUpdatesCallback(
    const std::vector<rclcpp::Parameter> & parameters);

  /**
   * @brief Apply parameter updates after validation.
   * @param parameters List of parameters that have been updated
   */
  void updateParametersCallback(const std::vector<rclcpp::Parameter> & parameters);

  // --- Parameters ---
  double inflation_radius_;
  double inscribed_radius_;
  double cost_scaling_factor_;
  double asymmetry_factor_;
  bool inflate_unknown_;
  bool inflate_around_unknown_;
  double goal_distance_threshold_;
  double neutral_threshold_;
  std::string plan_topic_;

  // --- State ---
  double current_robot_x_{0.0};
  double current_robot_y_{0.0};
  unsigned int cell_inflation_radius_;
  unsigned int cached_cell_inflation_radius_{0};
  double resolution_;
  unsigned int cache_length_;
  bool need_reinflation_;

  // --- BFS data structures ---
  std::vector<std::vector<CellData>> inflation_cells_;
  std::vector<bool> seen_;
  std::vector<double> cached_distances_;
  std::vector<unsigned char> cached_costs_;
  std::vector<int8_t> obstacle_side_grid_;

  /// Number of priority-queue bins per cell of effective distance.
  /// Higher values give finer ordering but increase memory use.
  static constexpr double kEffDistPrecision = 20.0;

  // --- Path subscription ---
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  nav_msgs::msg::Path::SharedPtr latest_global_path_;
  std::mutex path_mutex_;

  // --- Synchronization ---
  mutex_t * access_;

  // --- Dynamic parameter handlers ---
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr post_set_params_handler_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_set_params_handler_;
};

}  // namespace nav2_costmap_2d

#endif  // NAV2_COSTMAP_2D__ASYMMETRIC_INFLATION_LAYER_HPP_
