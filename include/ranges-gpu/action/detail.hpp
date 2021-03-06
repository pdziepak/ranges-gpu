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

#include "ranges-gpu/view/detail.hpp"

namespace ranges_gpu {
namespace action {
namespace detail {

template<bool KnownSize> struct known_size_tag {};

template<bool NeedsPreparing> struct needs_preparing_tag {};

template<typename V, typename Fn> decltype(auto) with_prepared(needs_preparing_tag<false>, V&& v, Fn&& fn) {
  return std::forward<Fn>(fn)(std::forward<V>(v));
}

template<typename V, typename Fn> decltype(auto) with_prepared(needs_preparing_tag<true>, V&& v, Fn&& fn) {
  auto ret = std::forward<V>(v).prepare();
  return std::forward<Fn>(fn)(std::move(std::get<1>(ret)));
}

inline constexpr size_t next_pow2(size_t x) noexcept {
  return 1 << (64 - __builtin_clzll(x - 1));
}

} // namespace detail
} // namespace action
} // namespace ranges_gpu
