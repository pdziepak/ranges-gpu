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

#include "ranges-gpu/array.hpp"

#include <gtest/gtest.h>

TEST(array, transfers) {
  auto in = std::array<int, 4>{1, 2, 3, 4};
  auto gpu = ranges_gpu::array<int>(in);
  EXPECT_EQ(gpu.size(), 4);
  auto out = std::vector<int>(4);
  copy(gpu, out);
  EXPECT_EQ(out[0], in[0]);
  EXPECT_EQ(out[1], in[1]);
  EXPECT_EQ(out[2], in[2]);
  EXPECT_EQ(out[3], in[3]);

  auto gpu2 = gpu;
  auto in2 = std::array<int, 4>{5, 6, 7, 8};
  copy(in2, gpu);

  copy(gpu2, out);
  EXPECT_EQ(out[0], in[0]);
  EXPECT_EQ(out[1], in[1]);
  EXPECT_EQ(out[2], in[2]);
  EXPECT_EQ(out[3], in[3]);

  copy(gpu, out);
  EXPECT_EQ(out[0], in2[0]);
  EXPECT_EQ(out[1], in2[1]);
  EXPECT_EQ(out[2], in2[2]);
  EXPECT_EQ(out[3], in2[3]);
}
