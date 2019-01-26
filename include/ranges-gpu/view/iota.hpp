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

#pragma once

#include "core.hpp"
#include "detail.hpp"

namespace ranges_gpu {
namespace view {

template<typename W> struct iota_view : base {
  W initial_{};
  W bound_{};

public:
  using value_type = W;

  iota_view() = default;
  iota_view(W initial, W bound) noexcept : initial_(std::move(initial)), bound_(std::move(bound)) {}

  static constexpr bool needs_preparing() noexcept { return false; }

  __device__ constexpr value_type operator[](size_t idx) const noexcept {
    auto v = initial_;
    v += idx;
    assert(v < bound_);
    return v;
  }

  __host__ __device__ constexpr size_t size() const noexcept { return bound_ - initial_; }
};

template<typename W> auto iota(W initial, W bound) noexcept {
  return iota_view<W>(std::move(initial), std::move(bound));
}

} // namespace view
} // namespace ranges_gpu
