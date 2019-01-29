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

#include <tuple>

#include "core.hpp"
#include "detail.hpp"

#include "ranges-gpu/array.hpp"
#include "ranges-gpu/detail.hpp"

namespace ranges_gpu {
namespace view {

namespace detail {

template<typename R> struct cpu_view : base {
  R const* in_;

public:
  using value_type = typename R::value_type;

  cpu_view(R const& r) noexcept : in_(&r) {}

  static constexpr bool needs_preparing() noexcept { return true; }
  auto prepare() && {
    auto gpu = array<value_type>(size());
    copy(*in_, gpu);
    auto v = to_view(gpu);
    auto bufs = std::vector<ranges_gpu::detail::untyped_buffer>();
    bufs.emplace_back(reinterpret_cast<char*>(std::move(gpu).release()));
    return std::make_tuple(std::move(bufs), std::move(v));
  }

  static constexpr bool known_size() noexcept { return true; }
  constexpr size_t size() const noexcept { return in_->size(); }
};

} // namespace detail

struct to_gpu {};

template<typename R> auto operator|(R const& r, to_gpu) {
  return detail::cpu_view<R>(r);
}

} // namespace view
} // namespace ranges_gpu
