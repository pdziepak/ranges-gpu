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

#include "ranges-gpu/action/reduce.hpp"
#include "ranges-gpu/view/transform.hpp"

namespace ranges_gpu {
namespace action {

template<typename V, typename F> auto any_of_fn(V&& v, F fn) -> bool {
  return std::forward<V>(v) | ranges_gpu::view::transform(std::move(fn)) |
         ranges_gpu::action::reduce(uint8_t(0), [] __device__(bool a, bool b) { return a || b; });
}

namespace detail {

template<typename F> struct any_of { F fn_; };

template<typename V, typename F> auto operator|(V&& v, detail::any_of<F> a) {
  return any_of_fn(std::forward<V>(v), std::move(a.fn_));
}

} // namespace detail

template<typename F> auto any_of(F fn) {
  return detail::any_of<F>{std::move(fn)};
}

} // namespace action
} // namespace ranges_gpu
