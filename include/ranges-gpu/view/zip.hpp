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

#include <iterator>
#include <tuple>

#include "core.hpp"
#include "detail.hpp"

#include "ranges-gpu/array.hpp"

namespace ranges_gpu {

template<typename T, typename U> struct pair {
  T first;
  U second;
};

template<typename T, typename U> __host__ __device__ auto make_pair(T t, U u) -> pair<T, U> {
  return {std::move(t), std::move(u)};
}

namespace view {

// FIXME: Make this work with any number of Vs (would be easier with C++17)
template<typename V1, typename V2> struct zip_view : base {
  V1 v1_;
  V2 v2_;

public:
  using value_type = pair<typename V1::value_type, typename V2::value_type>;

  zip_view(V1 v1, V2 v2) noexcept : v1_(std::move(v1)), v2_(std::move(v2)) {}

  static constexpr bool needs_preparing() noexcept { return V1::needs_preparing() || V2::needs_preparing(); }
  template<typename U1 = V1, typename U2 = V2,
           typename = std::enable_if_t<U1::needs_preparing() && !U2::needs_preparing()>>
  auto prepare() && {
    auto ret1 = std::move(v1_).prepare();
    using new_v1_type = std::decay_t<decltype(std::get<1>(ret1))>;
    return std::make_tuple(std::move(std::get<0>(ret1)), zip_view<new_v1_type, V2>(std::get<1>(ret1), std::move(v2_)));
  }
  template<typename U1 = V1, typename U2 = V2,
           typename = std::enable_if_t<!U1::needs_preparing() && U2::needs_preparing()>, typename = void>
  auto prepare() && {
    auto ret2 = std::move(v2_).prepare();
    using new_v2_type = std::decay_t<decltype(std::get<1>(ret2))>;
    return std::make_tuple(std::move(std::get<0>(ret2)), zip_view<V1, new_v2_type>(std::move(v1_), std::get<1>(ret2)));
  }
  template<typename U1 = V1, typename U2 = V2,
           typename = std::enable_if_t<U1::needs_preparing() && U2::needs_preparing()>, typename = void,
           typename = void>
  auto prepare() && {
    auto ret1 = std::move(v1_).prepare();
    using new_v1_type = std::decay_t<decltype(std::get<1>(ret1))>;
    auto ret2 = std::move(v2_).prepare();
    using new_v2_type = std::decay_t<decltype(std::get<1>(ret2))>;
    auto bufs1 = std::move(std::get<0>(ret1));
    auto bufs2 = std::move(std::get<0>(ret2));
    bufs1.insert(bufs1.end(), std::make_move_iterator(bufs2.begin()), std::make_move_iterator(bufs2.end()));
    return std::make_tuple(std::move(bufs1), zip_view<new_v1_type, new_v2_type>(std::move(std::get<1>(ret1)),
                                                                                std::move(std::get<1>(ret2))));
  }

  __device__ constexpr value_type const& operator[](size_t idx) const noexcept {
    assert(idx < size());
    return make_pair(v1_[idx], v2_[idx]);
  }

  __host__ __device__ constexpr size_t size() const noexcept {
    assert(v1_.size() == v2_.size());
    return v1_.size();
  }
};

template<typename V1, typename V2> auto zip(V1&& v1, V2&& v2) {
  auto in1 = detail::to_view(std::forward<V1>(v1));
  using in1_type = decltype(in1);
  auto in2 = detail::to_view(std::forward<V2>(v2));
  using in2_type = decltype(in2);
  return zip_view<in1_type, in2_type>(std::move(in1), std::move(in2));
}

} // namespace view
} // namespace ranges_gpu
