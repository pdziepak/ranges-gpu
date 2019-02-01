# Ranges GPU

[![Build Status](https://travis-ci.com/pdziepak/ranges-gpu.svg?branch=master)](https://travis-ci.com/pdziepak/ranges-gpu)

Experimental ranges library for CUDA. Heavily inspired by the C++ ranges.

## About

Ranges GPU is a header-only library implementing ranges for CUDA. Like the C++ ranges it allows composing lazy views into pipelines that can an appropriate time be materialised. The construction of pipelines happens on the CPU, while the actual computation needed to materialise the result are done on the GPU. The goals are to minimise the number of host-device data transfers and kernel launches and, hopefully, end up with code that can be optimised by the compiler to something not much worse than hand-written solutions (that's definitely not the case at the moment though, see [TODO](#TODO) below).

## Building

Ranges GPU is a header-only library. The default configuration will build unit tests and microbenchmarks.

The build requirements are as follows:

* CMake 3.10
* GCC 7
* CUDA 10
* Google Test (optional)
* Google Benchmark (optional)

Older versions may work but are not tested.

`nvidia/cuda` docker container may be a good place to start.

The build instructions are quite usual for a CMake-based project:

```
cd <build-directory>
cmake -DCMAKE_BUILD_TYPE=<Debug|Release> -G Ninja <source-directory>
ninja
ninja test
```

## Design

The main two concepts in the Ranges GPU library are actions and views. The former are eager, take a GPU range or a view as an input and produce either a GPU or CPU array or a value. The latter are lazy and may either represent a completely new range or in some transformation of the input range or ranges.

Views can be composed together using `operator|` to create more complex pipelines.

Only actions cause data transfers and kernel launches. Views are just lightweight description of what needs to be done when they are materialised by an action.

### Actions

* `all_of` – the equivalent of `std::all_of`. Built on top of `action::reduce` returns true if all elements in the input range satisfy the given predicate.
* `any_of` – the equivalent of `std::any_of`. Built on top of `action::reduce` returns true if any element of the input range satisfies the given predicate.
* `none_of` – the equivalent of `std::none_of`. Built on top of `action::reduce` returns true if none of the elements in the input range satisfies the given predicate.
* `reduce` – folds the range with the given associative and commutative binary function.
* `to_cpu` – materialises the input range and transforms it to the host memory.
* `to_gpu` – materialises the input range in the GPU memory.

### Views

* `filter` – a view of the input range without the elements that do not satisfy the given predicate.
* `iota` – a view of a sequence of elements.
* `take` – a view containing only N first elements of the input range.
* `to_gpu` – a view of a range in the host memory.
* `transform` – a view of the input range with the given unary function applied to all its elements.
* `zip` – a view of pairs, with the Nth pair containing the Nth elements of the first and the second input range.

## Examples

### Generating a range on the GPU

```c++
auto result = ranges_gpu::view::iota(1, 5)
            | ranges_gpu::view::transform([] __host__ __device__(int x) {
                    return float(x) / 10;
                })
            | ranges_gpu::action::to_cpu()
            ;
// result contains: {0.1f, 0.2f, 0.3f, 0.4f}
```

This example uses `view::iota` to create (on the GPU) a range of 4 integers `{1, 2, 3, 4}`, transforms it by dividing each of them by `10` and transfers the result to the host memory.

### Filtering and transforming a range on the GPU

```c++
auto vector = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
auto result = vector
            | ranges_gpu::view::to_gpu()
            | ranges_gpu::view::filter([] __device__(int x) {
                    return x % 3 == 0;
                })
            | ranges_gpu::view::transform([] __device__(int x) {
                    return x / 3;
                })
            | ranges_gpu::action::to_cpu()
            ;
// result contains: {1, 2, 3}
```

The code above transfers a vector to the GPU memory, filters it by removing elements non divisible by `3`, divides the remaining ones by `3` and copies the result back to the CPU.

### Reducing a GPU range

```c++
auto gpu_array = ranges_gpu::view::iota(1, 10)
               | ranges_gpu::action::to_gpu()
               ;
auto result = gpu_array
            | ranges_gpu::action::reduce(0, [] __device__(int a, int b) {
                    return a + b;
                })
            ;
// result is equal 45
```

In this example, an GPU array of 9 integers (`[1, 9]`) is created first. Then, all of them are added to each other using `action::reduce` and the result is returned on the host.

### Zipping ranges with filter and transform

```c++
auto on_cpu = std::vector<int>{10, 20, 30, 40, 50, 60, 70, 80, 90};
auto result = ranges_gpu::view::zip(
                    ranges_gpu::view::iota(1, 10),
                    on_cpu | ranges_gpu::view::to_gpu()
                )
            | ranges_gpu::view::transform([] __device__(ranges_gpu::pair<int, int> v) {
                    return v.first + v.second * 2;
                })
            | ranges_gpu::view::filter([] __device__(int x) {
                    return x % 2 && x / 10 <= 8;
                })
            | ranges_gpu::action::to_cpu()
            ;
// result contains: {21, 63}
```

The first step here is to create two ranges on the GPU:

* a range of 9 integers `[1, 9]` using `view::iota`
* a view of a CPU range `on_cpu`

Those two ranges are then zipped together using `view::zip`. The result is a range of pairs. Those pairs are then transformed and filtered by `view::transform` and `view::filter`, respectively. Finally, the resulting range is copied back to the host memory.

### Picking elements greater than the average

```c++
auto input = std::vector<int>{37, 44, 68, 74, 92, 4, 34, 97, 14, 27};
auto on_gpu = input
            | ranges_gpu::view::to_gpu()
            | ranges_gpu::action::to_gpu()
            ;
float average = on_gpu
              | ranges_gpu::action::reduce(0, [] __device__(int a, int b) {
                    return a + b;
                  })
              ;
average /= input.size();
// average is 49.1f

auto result = on_gpu
            | ranges_gpu::view::filter([average] __device__(float v) {
                    return v > average;
                })
            | ranges_gpu::action::to_cpu()
            ;
// result contains: {68, 74, 92, 97}
```

This example consists of three steps:

1. An input CPU range is copied to the GPU. This is done separately and not as a part of any other pipeline because we are going to need it more than once.
2. The elements of the GPU range are added together in order to get the average value.
3. The GPU range is filtered by removing elements that are not greater than the average. The result is copied back to the CPU.

## TODO

* Performance improvements
  * Most algorithms use very naive implementations. There is a lot to improve.
  * Consider how allowing the user to do more tuning user will affect the design.
  * There may be opportunities for more efficient template specialisations.
* Process data in batches – at the moment all elements of a range are processed at once. Doing so in sufficiently large batches would allow ranges that do not fit completely in the GPU memory (including infinite ranges).
* Add more actions and views:
  * `concat`
  * `sort`
  * `unique`
  * support for ranges of ranges would enable `split` and `join`
* More formal design and documentation. In particular, clarify which classes and their members are public interface and which are implementation details.
* Not all views can be composed with views of unknown size (those that use `view::filter`) – fix that.
* Ensure good integration with the standard C++ ranges.
* Add support for CUDA streams.
* Add more items to the TODO.
