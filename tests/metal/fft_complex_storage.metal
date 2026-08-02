// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/fft/readwrite.h"

template <typename T>
kernel void fft_complex_storage(
    const device complex_t<T>* in [[buffer(0)]],
    device complex_t<T>* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  auto value = FFTValueTraits<complex_t<T>>::load(in[index]);
  out[index] = FFTValueTraits<complex_t<T>>::store(value);
}

template [[host_name("fft_complex_storage_half")]] [[kernel]]
decltype(fft_complex_storage<half>) fft_complex_storage<half>;

template [[host_name("fft_complex_storage_bfloat16")]] [[kernel]]
decltype(fft_complex_storage<bfloat16_t>) fft_complex_storage<bfloat16_t>;
