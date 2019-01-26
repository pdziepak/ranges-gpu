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

#include "ranges-gpu/action/reduce.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/view/iota.hpp"
#include "ranges-gpu/view/to_gpu.hpp"

struct add {
  template<typename T> __device__ T operator()(T a, T b) const noexcept { return a + b; }
};

TEST(reduce, empty) {
  auto value = ranges_gpu::view::iota(8, 8) | ranges_gpu::action::reduce(4, add{});
  static_assert(std::is_same<decltype(value), int>::value, "");
  EXPECT_EQ(value, 4);
}

TEST(reduce, small_sum) {
  auto value = ranges_gpu::view::iota(2, 6) | ranges_gpu::action::reduce(1, add{});
  static_assert(std::is_same<decltype(value), int>::value, "");
  EXPECT_EQ(value, 15);
}

TEST(reduce, medium_sum) {
  auto value = ranges_gpu::view::iota(0, 4 * 1024 + 16) | ranges_gpu::action::reduce(1, add{});
  static_assert(std::is_same<decltype(value), int>::value, "");
  EXPECT_EQ(value, 8452217);
}

TEST(reduce, large_sum) {
  auto value = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull) | ranges_gpu::action::reduce(2ull, add{});
  static_assert(std::is_same<decltype(value), unsigned long long>::value, "");
  EXPECT_EQ(value, 4947800752130ull);
}

TEST(reduce, small_sum_from_cpu) {
  auto in = std::array<int, 4>{1, 2, 3, 4};
  auto value = in | ranges_gpu::view::to_gpu() | ranges_gpu::action::reduce(0, add{});
  static_assert(std::is_same<decltype(value), int>::value, "");
  EXPECT_EQ(value, 10);
}
