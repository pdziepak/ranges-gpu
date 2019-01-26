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

template<typename V, typename F> struct transform_view : base {
  V in_;
  F fn_;

public:
  using value_type = std::result_of_t<F(typename V::value_type)>;

  transform_view(V v, F fn) noexcept : in_(std::move(v)), fn_(std::move(fn)) {}

  static constexpr bool needs_preparing() noexcept { return V::needs_preparing(); }
  template<typename U = V, typename = std::enable_if_t<U::needs_preparing()>> auto prepare() && {
    auto ret = std::move(in_).prepare();
    using new_view_type = std::decay_t<decltype(std::get<1>(ret))>;
    return std::make_tuple(std::move(std::get<0>(ret)), transform_view<new_view_type, F>(std::get<1>(ret), fn_));
  }

  __device__ constexpr value_type operator[](size_t idx) const noexcept {
    assert(idx < size());
    return fn_(in_[idx]);
  }

  __host__ __device__ constexpr size_t size() const noexcept { return in_.size(); }
};

namespace detail {

template<typename F> struct transform { F fn_; };

template<typename V, typename F> auto operator|(V&& v, detail::transform<F> t) {
  using view_type = decltype(detail::to_view(v));
  return transform_view<view_type, F>(detail::to_view(std::forward<V>(v)), std::move(t.fn_));
}

}; // namespace detail

template<typename F> auto transform(F&& fn) noexcept {
  return detail::transform<std::decay_t<F>>{std::forward<F>(fn)};
}

} // namespace view
} // namespace ranges_gpu
