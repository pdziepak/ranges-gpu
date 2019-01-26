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

#include "ranges-gpu/view/zip.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/action/reduce.hpp"
#include "ranges-gpu/action/to_cpu.hpp"
#include "ranges-gpu/view/iota.hpp"
#include "ranges-gpu/view/to_gpu.hpp"
#include "ranges-gpu/view/transform.hpp"

struct sum2 {
  __device__ int operator()(ranges_gpu::pair<int, int> v) const noexcept { return v.first + v.second; }
};

TEST(zip, from_gpu) {
  auto out = ranges_gpu::view::zip(ranges_gpu::view::iota(0, 5), ranges_gpu::view::iota(9, 14)) |
             ranges_gpu::view::transform(sum2{}) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 5);
  auto expected = {9, 11, 13, 15, 17};
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

TEST(zip, from_cpu) {
  auto in1 = std::vector<int>{1, 2, 3};
  auto in2 = std::vector<int>{8, 10, 12};
  auto out = ranges_gpu::view::zip(in1 | ranges_gpu::view::to_gpu(), in2 | ranges_gpu::view::to_gpu()) |
             ranges_gpu::view::transform(sum2{}) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 3);
  auto expected = {9, 12, 15};
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

TEST(zip, from_cpu_gpu) {
  auto in1 = ranges_gpu::view::iota(1, 4);
  auto in2 = std::vector<int>{8, 10, 12};
  auto out = ranges_gpu::view::zip(in1, in2 | ranges_gpu::view::to_gpu()) | ranges_gpu::view::transform(sum2{}) |
             ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 3);
  auto expected = {9, 12, 15};
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

struct sum {
  __device__ int operator()(int a, int b) const noexcept { return a + b; }
};

struct sum3 {
  __device__ int operator()(ranges_gpu::pair<int, ranges_gpu::pair<int, int>> v) const noexcept {
    return v.first + v.second.first + v.second.second;
  }
};

TEST(zip, nested) {
  auto in1 = std::vector<int>{1, 2, 3};
  auto in2 = std::vector<int>{8, 10, 12};
  auto in3 = std::vector<int>{30, 40, 50};
  auto v =
      ranges_gpu::view::zip(in1 | ranges_gpu::view::to_gpu(),
                            ranges_gpu::view::zip(in2 | ranges_gpu::view::to_gpu(), in3 | ranges_gpu::view::to_gpu())) |
      ranges_gpu::view::transform(sum3{});
  auto val = v | ranges_gpu::action::reduce(0, sum{});
  EXPECT_EQ(val, 156);
  auto out = v | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 3);
  auto expected = {39, 52, 65};
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}
