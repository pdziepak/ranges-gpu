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

#include <iostream>
#include <mutex>

#include <benchmark/benchmark.h>

static void print_gpu_info() {
  int dev;
  [[maybe_unused]] auto ret = cudaGetDevice(&dev);
  assert(ret == cudaSuccess);
  cudaDeviceProp prop;
  ret = cudaGetDeviceProperties(&prop, dev);
  assert(ret == cudaSuccess);
  std::cout << "Run on " << prop.name << " (" << prop.clockRate / 1'000
            << " MHz GPU)\n  Global memory: " << prop.totalGlobalMem / 1'000'000
            << " MB\n  Shared memory per block: " << prop.sharedMemPerBlock / 1'000
            << " kB\n  Registers per block: " << prop.regsPerBlock
            << "\n  Maximum threads per block: " << prop.maxThreadsPerBlock << "\n  Warp size: " << prop.warpSize
            << "\n";
}

static void print_gpu_info_once() {
  static std::once_flag once;
  std::call_once(once, print_gpu_info);
}

template<typename F> void measure_gpu_time(benchmark::State& state, F&& fn) {
  print_gpu_info_once();

  cudaEvent_t start;
  [[maybe_unused]] auto ret = cudaEventCreate(&start);
  assert(ret == cudaSuccess);
  cudaEvent_t stop;
  ret = cudaEventCreate(&stop);
  assert(ret == cudaSuccess);

  for (auto _ : state) {
    cudaEventRecord(start);
    fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float dt;
    cudaEventElapsedTime(&dt, start, stop);
    state.SetIterationTime(double(dt) / 1'000);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
