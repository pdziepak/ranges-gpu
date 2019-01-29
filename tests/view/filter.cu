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

#include "ranges-gpu/view/filter.hpp"

#include <gtest/gtest.h>

#include "ranges-gpu/action/reduce.hpp"
#include "ranges-gpu/action/to_cpu.hpp"
#include "ranges-gpu/view/iota.hpp"
#include "ranges-gpu/view/transform.hpp"

template<int N> struct is_divisible_by {
  template<typename T> __device__ bool operator()(T x) const noexcept { return x % N == 0; }
};

struct add {
  template<typename T, typename U> __device__ auto operator()(T x, U y) const noexcept { return x + y; }
};

TEST(filter, reduce) {
  auto value = ranges_gpu::view::iota(0, 5) | ranges_gpu::view::filter(is_divisible_by<2>{}) |
               ranges_gpu::action::reduce(1, add{});
  EXPECT_EQ(value, 7);
}

template<int N> struct is_less_than {
  template<typename T> __device__ bool operator()(T x) const noexcept { return x < N; }
};

TEST(filter, reduce_large) {
  auto value = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<2>{}) |
               ranges_gpu::action::reduce(1ull, add{});
  EXPECT_EQ(value, 2473912172557ull);
  value = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_less_than<5>{}) |
          ranges_gpu::action::reduce(1ull, add{});
  EXPECT_EQ(value, 11);
}

TEST(filter, to_cpu) {
  auto out =
      ranges_gpu::view::iota(0, 15) | ranges_gpu::view::filter(is_divisible_by<2>{}) | ranges_gpu::action::to_cpu();
  auto expected = std::vector<int>{};
  for (auto i = 0; i < 15; i += 2) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0, 15) | ranges_gpu::view::filter(is_divisible_by<3>{}) | ranges_gpu::action::to_cpu();
  expected = std::vector<int>{};
  for (auto i = 0; i < 15; i += 3) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0, 15) | ranges_gpu::view::filter(is_divisible_by<1>{}) | ranges_gpu::action::to_cpu();
  expected = std::vector<int>{};
  for (auto i = 0; i < 15; i += 1) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

TEST(filter, to_cpu_large) {
  auto out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<2>{}) |
             ranges_gpu::action::to_cpu();
  auto expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 3 * 1024 * 1024ull + 8; i += 2) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<3>{}) |
        ranges_gpu::action::to_cpu();
  expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 3 * 1024 * 1024ull + 8; i += 3) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<1>{}) |
        ranges_gpu::action::to_cpu();
  expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 3 * 1024 * 1024ull + 8; i += 1) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}

struct add1 {
  __device__ int operator()(int x) const noexcept { return x + 1; }
};

TEST(filter, reduce_transform) {
  auto value = ranges_gpu::view::iota(0, 5) | ranges_gpu::view::filter(is_divisible_by<2>{}) |
               ranges_gpu::view::transform(add1{}) | ranges_gpu::action::reduce(0, add{});
  EXPECT_EQ(value, 9);
}

TEST(filter, nested) {
  auto out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<2>{}) |
             ranges_gpu::view::filter(is_less_than<15>{}) | ranges_gpu::action::to_cpu();
  auto expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 15ull; i += 2) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<3>{}) |
        ranges_gpu::view::filter(is_less_than<15>{}) | ranges_gpu::action::to_cpu();
  expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 15ull; i += 3) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));

  out = ranges_gpu::view::iota(0ull, 3 * 1024 * 1024ull + 8) | ranges_gpu::view::filter(is_divisible_by<1>{}) |
        ranges_gpu::view::filter(is_less_than<15>{}) | ranges_gpu::action::to_cpu();
  expected = std::vector<unsigned long long>{};
  for (auto i = 0ull; i < 15ull; i += 1) { expected.emplace_back(i); }
  EXPECT_EQ(out.size(), expected.size());
  EXPECT_TRUE(std::equal(out.begin(), out.end(), expected.begin(), expected.end()));
}
