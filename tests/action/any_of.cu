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

#include "ranges-gpu/action/any_of.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/view/iota.hpp"

struct is_even {
  __device__ bool operator()(int a) const noexcept { return a % 2 == 0; }
};

TEST(any_of, empty) {
  auto value = ranges_gpu::view::iota(0, 0) | ranges_gpu::action::any_of(is_even{});
  static_assert(std::is_same<decltype(value), bool>::value, "");
  EXPECT_FALSE(value);
}

TEST(any_of, trivial_true) {
  auto value = ranges_gpu::view::iota(0, 16) | ranges_gpu::action::any_of(is_even{});
  static_assert(std::is_same<decltype(value), bool>::value, "");
  EXPECT_TRUE(value);
}

struct is_greater_than_16 {
  __device__ bool operator()(int a) const noexcept { return a > 16; }
};

TEST(any_of, trivial_false) {
  auto value = ranges_gpu::view::iota(0, 16) | ranges_gpu::action::any_of(is_greater_than_16{});
  static_assert(std::is_same<decltype(value), bool>::value, "");
  EXPECT_FALSE(value);
}
