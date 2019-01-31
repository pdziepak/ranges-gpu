/*
 * Copyright © 2019 Paweł Dziepak
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <gtest/gtest.h>

#include "ranges-gpu/action/reduce.hpp"
#include "ranges-gpu/action/to_cpu.hpp"

#include "ranges-gpu/view/filter.hpp"
#include "ranges-gpu/view/iota.hpp"
#include "ranges-gpu/view/to_gpu.hpp"
#include "ranges-gpu/view/transform.hpp"
#include "ranges-gpu/view/zip.hpp"

void simple_to_cpu() {
  auto actual = ranges_gpu::view::iota(1, 5) |
                ranges_gpu::view::transform([] __host__ __device__(int x) { return float(x) / 10; }) |
                ranges_gpu::action::to_cpu();
  auto expected = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
  static_assert(std::is_same<decltype(actual), decltype(expected)>::value, "");
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

TEST(example, simple_to_cpu) {
  simple_to_cpu();
}

void filtered_to_cpu() {
  auto initial = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto actual = initial | ranges_gpu::view::to_gpu() |
                ranges_gpu::view::filter([] __host__ __device__(int x) { return x % 3 == 0; }) |
                ranges_gpu::view::transform([] __host__ __device__(int x) { return x / 3; }) |
                ranges_gpu::action::to_cpu();
  auto expected = std::vector<int>{1, 2, 3};
  static_assert(std::is_same<decltype(actual), decltype(expected)>::value, "");
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

TEST(example, filtered_to_cpu) {
  filtered_to_cpu();
}

void simple_reduce() {
  auto initial = ranges_gpu::view::iota(1, 10) | ranges_gpu::action::to_gpu();
  auto actual = initial | ranges_gpu::action::reduce(0, [] __host__ __device__(int a, int b) { return a + b; });
  auto expected = 45;
  static_assert(std::is_same<decltype(actual), decltype(expected)>::value, "");
  EXPECT_EQ(actual, expected);
}

TEST(example, simple_reduce) {
  simple_reduce();
}

void zip_filter_transform() {
  auto on_cpu = std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80, 90};
  auto actual = ranges_gpu::view::zip(ranges_gpu::view::iota(1, 10), on_cpu | ranges_gpu::view::to_gpu()) |
                ranges_gpu::view::transform(
                    [] __host__ __device__(ranges_gpu::pair<int, int> v) { return v.first + v.second * 2; }) |
                ranges_gpu::view::filter([] __host__ __device__(int x) { return x % 2 && x / 10 <= 8; }) |
                ranges_gpu::action::to_cpu();
  auto expected = std::vector<int>{21, 63};
  static_assert(std::is_same<decltype(actual), decltype(expected)>::value, "");
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

TEST(example, zip_filter_transform) {
  zip_filter_transform();
}

void above_average() {
  auto input = std::vector<int>{37, 44, 68, 74, 92, 4, 34, 97, 14, 27};
  auto on_gpu = input | ranges_gpu::view::to_gpu() | ranges_gpu::action::to_gpu();
  auto average = float(on_gpu | ranges_gpu::action::reduce(0, [] __host__ __device__(int a, int b) { return a + b; })) /
                 input.size();
  EXPECT_EQ(average, 49.1f);
  auto actual = on_gpu | ranges_gpu::view::filter([average] __host__ __device__(float v) { return v > average; }) |
                   ranges_gpu::action::to_cpu();
  auto expected = std::vector<int>{68, 74, 92, 97};
  static_assert(std::is_same<decltype(actual), decltype(expected)>::value, "");
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()));
}

TEST(example, above_average) {
  above_average();
}
