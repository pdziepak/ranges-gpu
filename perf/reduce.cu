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

#include "perf.hpp"

#include "ranges-gpu/action/reduce.hpp"
#include "ranges-gpu/action/to_gpu.hpp"
#include "ranges-gpu/view/iota.hpp"

void sum_i32_iota(benchmark::State& s) {
  int bound = s.range(0);
  measure_gpu_time(s, [&] {
    auto value = ranges_gpu::view::iota(0, bound) |
                 ranges_gpu::action::reduce(0, [] __device__(auto a, auto b) { return a + b; });
    benchmark::DoNotOptimize(value);
  });
}

BENCHMARK(sum_i32_iota)
    ->RangeMultiplier(100)
    ->Range(10'000, 100'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

void sum_i32_array(benchmark::State& s) {
  auto array = ranges_gpu::view::iota(0, int(s.range(0))) | ranges_gpu::action::to_gpu();
  measure_gpu_time(s, [&] {
    auto value = array | ranges_gpu::action::reduce(0, [] __device__(auto a, auto b) { return a + b; });
    benchmark::DoNotOptimize(value);
  });
}
BENCHMARK(sum_i32_array)
    ->RangeMultiplier(100)
    ->Range(10'000, 100'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
