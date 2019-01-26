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

template<typename V> struct take_view : base {
  V in_;
  size_t max_size_;

public:
  using value_type = typename V::value_type;

  take_view(V v, size_t n) noexcept : in_(std::move(v)), max_size_(n) {}

  static constexpr bool needs_preparing() noexcept { return V::needs_preparing(); }
  template<typename U = V, typename = std::enable_if_t<U::needs_preparing()>> auto prepare() {
    auto ret = in_.prepare();
    using new_view_type = std::decay_t<decltype(std::get<1>(ret))>;
    return std::make_tuple(std::get<0>(ret), take_view<new_view_type>(std::get<1>(ret), max_size_));
  }

  __device__ constexpr value_type operator[](size_t idx) const noexcept {
    assert(idx < size());
    return in_[idx];
  }

  __host__ __device__ constexpr size_t size() const noexcept { return max_size_ < in_.size() ? max_size_ : in_.size(); }
};

namespace detail {

struct take {
  size_t n_;
};

template<typename V> auto operator|(V&& v, detail::take t) {
  using view_type = decltype(detail::to_view(v));
  return take_view<view_type>(detail::to_view(std::forward<V>(v)), std::move(t.n_));
}

}; // namespace detail

inline auto take(size_t n) noexcept {
  return detail::take{n};
}

} // namespace view
} // namespace ranges_gpu
