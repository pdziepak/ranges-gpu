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

namespace ranges_gpu {
namespace view {
namespace detail {

template<typename V, typename = std::enable_if_t<enable_view<V>>> auto to_view(V v) -> V {
  return v;
}

template<typename T> class array_view : public base {
  T const* begin_ = nullptr;
  T const* end_ = nullptr;

public:
  using value_type = T;

  constexpr array_view() noexcept = default;
  __host__ __device__ constexpr array_view(T* first, T* last) noexcept : begin_(first), end_(last) {}
  template<typename R> __host__ constexpr array_view(R&& r) noexcept : begin_(r.data()), end_(r.data() + r.size()) {}

  constexpr array_view(array_view&) noexcept = default;
  constexpr array_view(array_view const&) noexcept = default;
  constexpr array_view(array_view&&) noexcept = default;

  static constexpr bool needs_preparing() noexcept { return false; }

  __device__ constexpr T const& operator[](size_t idx) const noexcept {
    assert(idx < size());
    return begin_[idx];
  }

  static constexpr bool known_size() noexcept { return true; }
  __host__ __device__ constexpr size_t size_bound() const noexcept { return size(); }
  __host__ __device__ constexpr size_t size() const noexcept { return end_ - begin_; }
};

template<typename R, typename = std::enable_if_t<!enable_view<R>>> auto to_view(R const& r) {
  return array_view<typename R::value_type>(r);
}

} // namespace detail
} // namespace view
} // namespace ranges_gpu
