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

#include "ranges-gpu/view/iota.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/action/to_cpu.hpp"
#include "ranges-gpu/view/transform.hpp"

TEST(iota, trivial) {
  auto out = ranges_gpu::view::iota(1, 5) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 4);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
  EXPECT_EQ(out[2], 3);
  EXPECT_EQ(out[3], 4);
}

struct mul3 {
  __device__ int operator()(int x) const { return x * 3; }
};

TEST(iota, transform) {
  auto out = ranges_gpu::view::iota(2, 5) | ranges_gpu::view::transform(mul3{}) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), 3);
  EXPECT_EQ(out[0], 6);
  EXPECT_EQ(out[1], 9);
  EXPECT_EQ(out[2], 12);
}
