// Copyright (c) 2026 Marc Blöchlinger
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

// Unit tests for AsymmetricInflationLayer — pure-algorithm coverage.
//
// Tests exercise cost_lut_disfavored_ and seedDistanceMap via a test-subclass
// that exposes protected members and methods without requiring a full
// LayeredCostmap / LifecycleNode stack.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <cmath>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/asymmetric_inflation_layer.hpp"
#include "nav2_costmap_2d/distance_transform.hpp"

namespace nav2_costmap_2d
{

class TestableAsymmetricInflationLayer : public AsymmetricInflationLayer
{
public:
  // Expose protected methods and members for testing
  using AsymmetricInflationLayer::seedDistanceMap;
  using AsymmetricInflationLayer::cost_lut_disfavored_;

  // Plain setters so tests can configure the math without going through ROS init
  void setResolution(double r) {resolution_ = r;}
  void setInscribedRadius(double r) {inscribed_radius_ = r;}
  void setInflationRadius(double r) {inflation_radius_ = r;}
  void setCostScalingFactorLeft(double c)
  {
    cost_scaling_factor_left_ = c;
    updateCostScalingFactor();
  }
  void setCostScalingFactorRight(double c)
  {
    cost_scaling_factor_right_ = c;
    updateCostScalingFactor();
  }
  void setCellInflationRadius(unsigned int r) {cell_inflation_radius_ = r;}
  void rebuildCaches() {computeAsymmetricCaches();}

private:
  void updateCostScalingFactor()
  {
    cost_scaling_factor_ = std::max(cost_scaling_factor_left_, cost_scaling_factor_right_);
  }
};

}  // namespace nav2_costmap_2d

using nav2_costmap_2d::TestableAsymmetricInflationLayer;

// Test 1: cost_lut_disfavored_ is built with c_side (the smaller scaling factor).
// Verifies the mathematical equivalence: using c_max on effective distance equals
// using c_side on physical distance, so the LUT must reflect c_side.
TEST(DisfavoredLutTest, lut_uses_c_side_scaling_factor)
{
  auto layer = std::make_unique<TestableAsymmetricInflationLayer>();
  // resolution=0.1m, inscribed_radius=0.3m (3 cells), cell_inflation_radius=20 cells
  layer->setResolution(0.1);
  layer->setInscribedRadius(0.3);
  layer->setInflationRadius(2.0);
  layer->setCellInflationRadius(20);

  // c_left=1, c_right=7 → c_side=1 (left is disfavored), c_max=7
  layer->setCostScalingFactorLeft(1.0);
  layer->setCostScalingFactorRight(7.0);
  layer->rebuildCaches();

  ASSERT_FALSE(layer->cost_lut_disfavored_.empty());

  const double resolution = 0.1;
  const double inscribed_radius = 0.3;
  const double c_side = 1.0;
  const int lut_precision = 100;  // COST_LUT_PRECISION

  // Spot-check several distances beyond the inscribed radius
  for (int d_scaled = lut_precision * 4; d_scaled <= lut_precision * 19;
    d_scaled += lut_precision)
  {
    double distance = static_cast<double>(d_scaled) / lut_precision;  // cells
    double factor = exp(-c_side * (distance * resolution - inscribed_radius));
    unsigned char expected =
      static_cast<unsigned char>((nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE - 1) * factor);

    EXPECT_EQ(layer->cost_lut_disfavored_[d_scaled], expected)
      << "LUT mismatch at d_scaled=" << d_scaled;
  }

  // Inside the inscribed radius: costs must equal INSCRIBED_INFLATED_OBSTACLE
  for (int d_scaled = 1; d_scaled < lut_precision * 3; ++d_scaled) {
    EXPECT_EQ(
      layer->cost_lut_disfavored_[d_scaled],
      nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
      << "Expected INSCRIBED at d_scaled=" << d_scaled;
  }

  // At distance=0: must be LETHAL_OBSTACLE
  EXPECT_EQ(layer->cost_lut_disfavored_[0], nav2_costmap_2d::LETHAL_OBSTACLE);
}

// Test 2: swapping side factors rebuilds the LUT with the new c_side.
TEST(DisfavoredLutTest, lut_reflects_updated_c_side)
{
  auto layer = std::make_unique<TestableAsymmetricInflationLayer>();
  layer->setResolution(0.1);
  layer->setInscribedRadius(0.3);
  layer->setCellInflationRadius(10);
  layer->setCostScalingFactorLeft(1.0);
  layer->setCostScalingFactorRight(7.0);
  layer->rebuildCaches();

  std::vector<unsigned char> lut_c1 = layer->cost_lut_disfavored_;

  // Swap sides: now c_right=1 is disfavored, c_left=7 is favored
  layer->setCostScalingFactorLeft(7.0);
  layer->setCostScalingFactorRight(1.0);
  layer->rebuildCaches();

  // With both cases having c_side=1, LUTs should be identical
  EXPECT_EQ(lut_c1, layer->cost_lut_disfavored_);
}

// Fixture for seedDistanceMap tests.
//
// Setup: 20x20 costmap (0.1m/cell, 2m×2m), horizontal path y=1.0m going east.
// c_left=2.0 < c_right=8.0 → disfavored_side = +1 (LEFT = north for eastward path).
// inflation_radius=1.0m → cell_inflation_radius=10 cells.
class SeedDistanceMapTest : public ::testing::Test
{
protected:
  static constexpr int kSize = 20;
  static constexpr double kRes = 0.1;
  static constexpr float kInf = nav2_costmap_2d::DistanceTransform::DT_INF;

  void SetUp() override
  {
    costmap_ = std::make_unique<nav2_costmap_2d::Costmap2D>(kSize, kSize, kRes, 0.0, 0.0);
    layer_ = std::make_unique<TestableAsymmetricInflationLayer>();
    layer_->setResolution(kRes);
    layer_->setInscribedRadius(0.1);
    layer_->setInflationRadius(1.0);
    layer_->setCellInflationRadius(10);
    layer_->setCostScalingFactorLeft(2.0);
    layer_->setCostScalingFactorRight(8.0);
  }

  // Horizontal path going east at world y=1.0m
  std::vector<std::pair<double, double>> eastwardPath() const
  {
    return {{0.5, 1.0}, {1.5, 1.0}};
  }

  std::unique_ptr<nav2_costmap_2d::Costmap2D> costmap_;
  std::unique_ptr<TestableAsymmetricInflationLayer> layer_;
};

// Test 3: A LETHAL boundary cell on the disfavored (left/north) side of an eastward
// path is seeded as 0.0f; a cell on the favored (right/south) side is not.
TEST_F(SeedDistanceMapTest, disfavored_side_seeded_favored_side_not)
{
  // North of path at world (1.05, 1.35) → cell (mx=10, my=13): disfavored (LEFT)
  costmap_->setCost(10, 13, nav2_costmap_2d::LETHAL_OBSTACLE);
  // South of path at world (1.05, 0.75) → cell (mx=10, my=7): favored (RIGHT)
  costmap_->setCost(10, 7, nav2_costmap_2d::LETHAL_OBSTACLE);

  auto dist_map = layer_->seedDistanceMap(*costmap_, 0, 0, kSize, kSize, eastwardPath());

  // dist_map indexing: (row = my - roi_min_j, col = mx - roi_min_i)
  EXPECT_FLOAT_EQ(dist_map(13, 10), 0.0f)
    << "North (disfavored) obstacle should be seeded";
  EXPECT_FLOAT_EQ(dist_map(7, 10), kInf)
    << "South (favored) obstacle should not be seeded";
}

// Test 4: Swapping the scaling factors flips which side is disfavored — the south
// (right) obstacle is now seeded and the north (left) is not.
TEST_F(SeedDistanceMapTest, swapped_factors_flip_disfavored_side)
{
  // c_left=8.0 > c_right=2.0 → disfavored_side = -1 (RIGHT = south for eastward path)
  layer_->setCostScalingFactorLeft(8.0);
  layer_->setCostScalingFactorRight(2.0);

  costmap_->setCost(10, 13, nav2_costmap_2d::LETHAL_OBSTACLE);  // north
  costmap_->setCost(10, 7, nav2_costmap_2d::LETHAL_OBSTACLE);   // south

  auto dist_map = layer_->seedDistanceMap(*costmap_, 0, 0, kSize, kSize, eastwardPath());

  EXPECT_FLOAT_EQ(dist_map(13, 10), kInf)
    << "North (now favored) obstacle should not be seeded";
  EXPECT_FLOAT_EQ(dist_map(7, 10), 0.0f)
    << "South (now disfavored) obstacle should be seeded";
}

// Test 5: An obstacle beyond the inflation radius (>10 cells from any path cell)
// must not be seeded, regardless of which side it is on.
TEST_F(SeedDistanceMapTest, obstacle_beyond_inflation_radius_not_seeded)
{
  // Cell (mx=0, my=19) = world (0.05, 1.95).
  // Nearest path cell is (mx=5, my=10): Euclidean distance = sqrt(25+81) ≈ 10.3 > 10 cells.
  costmap_->setCost(0, 19, nav2_costmap_2d::LETHAL_OBSTACLE);

  auto dist_map = layer_->seedDistanceMap(*costmap_, 0, 0, kSize, kSize, eastwardPath());

  EXPECT_FLOAT_EQ(dist_map(19, 0), kInf)
    << "Obstacle beyond inflation radius should not be seeded";
}

// Test 6: With no LETHAL cells in the costmap, every dist_map entry must remain DT_INF.
TEST_F(SeedDistanceMapTest, no_obstacles_all_inf)
{
  auto dist_map = layer_->seedDistanceMap(*costmap_, 0, 0, kSize, kSize, eastwardPath());

  for (int r = 0; r < kSize; ++r) {
    for (int c = 0; c < kSize; ++c) {
      EXPECT_FLOAT_EQ(dist_map(r, c), kInf)
        << "Cell (" << r << "," << c << ") should be DT_INF with no obstacles";
    }
  }
}

// Test 7: When the path lies entirely outside the ROI, the BFS queue is empty
// and every dist_map entry stays DT_INF.
TEST_F(SeedDistanceMapTest, path_outside_roi_nothing_seeded)
{
  // Obstacle inside the ROI window [0,10)×[0,10), but the path is outside it
  costmap_->setCost(5, 5, nav2_costmap_2d::LETHAL_OBSTACLE);

  // ROI = upper-left quadrant [0,10)×[0,10); path cells are around mx=5..15,my=10 — outside
  auto dist_map = layer_->seedDistanceMap(*costmap_, 0, 0, 10, 10, eastwardPath());

  for (int r = 0; r < 10; ++r) {
    for (int c = 0; c < 10; ++c) {
      EXPECT_FLOAT_EQ(dist_map(r, c), kInf)
        << "Cell (" << r << "," << c << ") should be DT_INF when path is outside ROI";
    }
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
