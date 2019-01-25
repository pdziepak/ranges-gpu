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

#include "ranges-gpu/view/transform.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/action/to_cpu.hpp"
#include "ranges-gpu/array.hpp"

void simple_test() {
  auto in = std::array<int, 4>{1, 2, 3, 4};
  auto gpu = ranges_gpu::array<int>(in);

  auto out = gpu | ranges_gpu::view::transform([] __device__(int x) { return x * 2; }) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), in.size());
  auto expected = std::array<int, 4>{};
  std::transform(in.begin(), in.end(), expected.begin(), [](int x) { return x * 2; });
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

TEST(transform, simple) {
  simple_test();
}

void with_capture_test(int y) {
  auto in = std::array<int, 4>{1, 2, 3, 4};
  auto gpu = ranges_gpu::array<int>(in);

  auto out = gpu | ranges_gpu::view::transform([y] __device__(int x) { return x + y; }) | ranges_gpu::action::to_cpu();
  EXPECT_EQ(out.size(), in.size());
  auto expected = std::array<int, 4>{};
  std::transform(in.begin(), in.end(), expected.begin(), [&](int x) { return x + y; });
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

TEST(transform, with_capture) {
  with_capture_test(4);
}
