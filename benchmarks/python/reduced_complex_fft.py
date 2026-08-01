# Copyright © 2026 John Elliott

"""Exercise reduced-precision complex FFT lanes without adding a public dtype.

The benchmark uses interleaved ``[..., 2]`` scalar arrays as raw complex
storage and MLX custom Metal kernels as the dispatch vehicle.  The kernels call
the same templated FFT implementation used by MLX's built-in FFT path.

Examples:

    python benchmarks/python/reduced_complex_fft.py
    python benchmarks/python/reduced_complex_fft.py --static-audit
    python benchmarks/python/reduced_complex_fft.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

RADICES = (13, 11, 8, 7, 6, 5, 4, 3, 2)
MAX_STOCKHAM_FFT_SIZE = 4096
MAX_RADER_FFT_SIZE = 2048
MAX_BLUESTEIN_FFT_SIZE = 2048
MIN_THREADGROUP_MEM_SIZE = 256
MIN_COALESCE_WIDTH = 4


@dataclass(frozen=True)
class FFTPlan:
    kind: str
    n: int
    fft_n: int
    stockham: tuple[int, ...]
    rader: tuple[int, ...]
    rader_n: int = 1
    n1: int = 0
    n2: int = 0


@dataclass(frozen=True)
class Lane:
    name: str
    storage_type: str
    scalar_type: str
    bytes_per_complex: int

    @property
    def dtype(self):
        return {
            "float": mx.float32,
            "half": mx.float16,
            "bfloat": mx.bfloat16,
        }[self.name]


LANES = {
    "float": Lane("float", "complex_t<float>", "float", 8),
    "half": Lane("half", "complex_t<half>", "half", 4),
    "bfloat": Lane("bfloat", "complex_t<bfloat16_t>", "bfloat16_t", 4),
}

CASES = {
    "stockham": 256,
    "rader": 17,
    "bluestein": 47,
    "four-step": 8192,
}


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def prime_factors(n: int) -> list[int]:
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        else:
            divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def plan_stockham(n: int) -> tuple[int, ...]:
    steps = [0] * len(RADICES)
    original_n = n
    if n == 1:
        return tuple(steps)
    for index, radix in enumerate(RADICES):
        if is_power_of_two(original_n) and original_n < 512 and radix > 4:
            continue
        while n % radix == 0:
            steps[index] += 1
            n //= radix
            if n == 1:
                return tuple(steps)
    raise ValueError(f"{original_n} is not Stockham-decomposable")


def plan_fft(n: int) -> FFTPlan:
    zeros = (0,) * len(RADICES)
    if n > MAX_STOCKHAM_FFT_SIZE and is_power_of_two(n):
        n2 = 1024 if n > 65536 else 64
        return FFTPlan("four-step", n, n, zeros, zeros, n1=n // n2, n2=n2)
    if n > MAX_STOCKHAM_FFT_SIZE:
        raise ValueError("multi-upload Bluestein is outside this focused proof")

    rader = zeros
    rader_n = 1
    remaining_n = n
    radix_set = set(RADICES)
    for factor in prime_factors(n):
        if factor in radix_set:
            continue
        if rader_n > 1 or n > MAX_RADER_FFT_SIZE:
            fft_n = next_power_of_two(2 * n - 1)
            return FFTPlan("bluestein", n, fft_n, plan_stockham(fft_n), zeros)
        if any(value not in radix_set for value in prime_factors(factor - 1)):
            fft_n = next_power_of_two(2 * n - 1)
            return FFTPlan("bluestein", n, fft_n, plan_stockham(fft_n), zeros)
        rader = plan_stockham(factor - 1)
        rader_n = factor
        remaining_n //= factor

    return FFTPlan(
        "rader" if rader_n > 1 else "stockham",
        n,
        n,
        plan_stockham(remaining_n),
        rader,
        rader_n=rader_n,
    )


def compute_elems_per_thread(plan: FFTPlan) -> int:
    used = {
        radix
        for steps_by_radix in (plan.stockham, plan.rader)
        for radix, steps in zip(RADICES, steps_by_radix)
        if steps
    }
    if 7 in used and (11 in used or 13 in used):
        return 7
    if 11 in used and 13 in used:
        return 11
    tuned = {3159: 13, 3645: 5, 3969: 7, 1982: 5}
    if plan.n in tuned:
        return tuned[plan.n]
    ordered = sorted(used)
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        if 11 in used or 13 in used:
            return sum(ordered) // 2
        return ordered[1]
    if len(ordered) >= 3:
        return ordered[1]
    raise ValueError(f"empty FFT plan for n={plan.n}")


def step_plan(plan: FFTPlan, step: int) -> FFTPlan:
    n = plan.n1 if step == 0 else plan.n2
    zeros = (0,) * len(RADICES)
    return FFTPlan("stockham", n, n, plan_stockham(n), zeros)


def source_without_local_includes(path: Path) -> str:
    return "\n".join(
        line
        for line in path.read_text().splitlines()
        if not line.startswith('#include "mlx/') and line != "#pragma once"
    )


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one source fragment: {old!r}")
    return source.replace(old, new, 1)


def demote_kernel_functions(source: str) -> str:
    source, kernel_count = re.subn(r"\[\[kernel\]\]\s*", "", source)
    if kernel_count != 4:
        raise RuntimeError(f"expected four FFT kernel templates, found {kernel_count}")
    source = re.sub(
        r"\s*\[\[(?:buffer\(\d+\)|thread_position_in_grid|threads_per_grid)\]\]",
        "",
        source,
    )
    source = source.replace("constant const int&", "const int")
    shared_pointer = (
        "threadgroup fft_complex_t<typename FFTIOTypeTraits<in_T, out_T>::scalar_T>* "
        "shared_in"
    )
    source = replace_once(
        source,
        "device out_T* out,\n    const int n,",
        f"device out_T* out,\n    {shared_pointer},\n    const int n,",
    )
    source = replace_once(
        source,
        "const device short* raders_g_minus_q,\n    const int n,",
        f"const device short* raders_g_minus_q,\n    {shared_pointer},\n    const int n,",
    )
    source = replace_once(
        source,
        "        w_k,\n    const int length,",
        f"        w_k,\n    {shared_pointer},\n    const int length,",
    )
    source = replace_once(
        source,
        "device out_T* out,\n    const int n1,",
        f"device out_T* out,\n    {shared_pointer},\n    const int n1,",
    )
    shared_declaration = (
        "  threadgroup fft_complex_t<scalar_T> shared_in[tg_mem_size];\n"
    )
    if source.count(shared_declaration) != 4:
        raise RuntimeError("expected one threadgroup allocation per FFT kernel")
    source = source.replace(shared_declaration, "")
    return source


def specialized_fft_header(root: Path, plan: FFTPlan, inverse: bool) -> str:
    radix = source_without_local_includes(
        root / "mlx/backend/metal/kernels/fft/radix.h"
    )
    readwrite = source_without_local_includes(
        root / "mlx/backend/metal/kernels/fft/readwrite.h"
    )
    fft = source_without_local_includes(root / "mlx/backend/metal/kernels/fft.h")

    values: dict[str, bool | int] = {
        "inv_": inverse,
        "is_power_of_2_": is_power_of_two(plan.fft_n),
        "elems_per_thread_": compute_elems_per_thread(plan),
        "rader_m_": plan.n // plan.rader_n,
    }
    values.update(
        {f"radix_{radix}_steps_": steps for radix, steps in zip(RADICES, plan.stockham)}
    )
    values.update(
        {f"rader_{radix}_steps_": steps for radix, steps in zip(RADICES, plan.rader)}
    )

    pattern = re.compile(
        r"STEEL_CONST\s+(bool|int)\s+(\w+)\s+" r"\[\[function_constant\(\d+\)\]\];"
    )

    def replace_constant(match: re.Match[str]) -> str:
        metal_type, name = match.groups()
        value = values[name]
        literal = str(value).lower() if isinstance(value, bool) else str(value)
        return f"static constant constexpr const {metal_type} {name} = {literal};"

    fft, constant_count = pattern.subn(replace_constant, fft)
    if constant_count != 22:
        raise RuntimeError(
            f"expected 22 FFT function constants, found {constant_count}"
        )
    if "function_constant" in fft:
        raise RuntimeError("not all FFT function constants were specialized")

    return "\n".join(
        [
            "#include <metal_common>",
            "#include <metal_math>",
            "#include <metal_stdlib>",
            '#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")',
            '#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")',
            radix,
            readwrite,
            demote_kernel_functions(fft),
        ]
    )


def mod_exp(x: int, y: int, modulus: int) -> int:
    result = 1
    while y:
        if y & 1:
            result = result * x % modulus
        y >>= 1
        x = x * x % modulus
    return result


def primitive_root(n: int) -> int:
    factors = prime_factors(n - 1)
    for candidate in range(2, n - 1):
        if all(mod_exp(candidate, (n - 1) // factor, n) != 1 for factor in factors):
            return candidate
    raise ValueError(f"no primitive root for {n}")


def complex_to_lanes(values: np.ndarray, lane: Lane) -> Any:
    packed = np.stack((values.real, values.imag), axis=-1).astype(np.float32)
    return mx.array(packed, dtype=lane.dtype)


def lanes_to_complex(values: Any) -> np.ndarray:
    packed = np.array(values.astype(mx.float32))
    return packed[..., 0] + 1j * packed[..., 1]


def rader_constants(plan: FFTPlan, lane: Lane) -> tuple[Any, Any, Any]:
    root = primitive_root(plan.rader_n)
    inverse = mod_exp(root, plan.rader_n - 2, plan.rader_n)
    g_q = np.array(
        [mod_exp(root, index, plan.rader_n) for index in range(plan.rader_n - 1)],
        dtype=np.int16,
    )
    g_minus_q = np.array(
        [mod_exp(inverse, index, plan.rader_n) for index in range(plan.rader_n - 1)],
        dtype=np.int16,
    )
    angles = g_minus_q.astype(np.float64) * (-2.0 * np.pi / plan.rader_n)
    b_q = np.exp(1j * angles)
    b_q = np.fft.fft(b_q).astype(np.complex64)
    return (
        complex_to_lanes(b_q, lane),
        mx.array(g_q, dtype=mx.int16),
        mx.array(g_minus_q, dtype=mx.int16),
    )


def bluestein_constants(plan: FFTPlan, lane: Lane) -> tuple[Any, Any]:
    w_k = np.empty(plan.n, dtype=np.complex128)
    w_q = np.zeros(plan.fft_n, dtype=np.complex128)
    for index in range(-plan.n + 1, plan.n):
        theta = index**2 * np.pi / plan.n
        w_q[index + plan.n - 1] = np.exp(1j * theta)
        if index >= 0:
            w_k[index] = np.exp(-1j * theta)
    w_q = np.fft.fft(w_q)
    return complex_to_lanes(w_q, lane), complex_to_lanes(w_k, lane)


def dispatch_geometry(
    fft_n: int, elems_per_thread: int, total_batch: int, four_step: bool
) -> tuple[int, tuple[int, int, int], tuple[int, int, int]]:
    threads_per_fft = math.ceil(fft_n / elems_per_thread)
    threadgroup_batch = max(MIN_THREADGROUP_MEM_SIZE // fft_n, 1)
    if four_step:
        threadgroup_batch = max(threadgroup_batch, MIN_COALESCE_WIDTH)
    threadgroup_mem = next_power_of_two(threadgroup_batch * fft_n)
    batch_groups = math.ceil(total_batch / threadgroup_batch)
    return (
        threadgroup_mem,
        (batch_groups, threadgroup_batch, threads_per_fft),
        (1, threadgroup_batch, threads_per_fft),
    )


class ReducedComplexFFTRunner:
    def __init__(
        self,
        root: Path,
        plan: FFTPlan,
        lane: Lane,
        batch: int,
        inverse: bool,
    ):
        self.root = root
        self.plan = plan
        self.lane = lane
        self.batch = batch
        self.inverse = inverse
        self.kernels = []
        self._build()

    def _kernel(self, suffix: str, header: str, source: str, input_names: list[str]):
        kind = self.plan.kind.replace("-", "_")
        return mx.fast.metal_kernel(
            name=(
                f"reduced_complex_fft_{kind}_{self.plan.n}_"
                f"{self.lane.name}_{'inv' if self.inverse else 'fwd'}_{suffix}"
            ),
            input_names=input_names,
            output_names=["out"],
            header=header,
            source=source,
            ensure_row_contiguous=False,
        )

    def _common_source(self, threadgroup_mem: int) -> str:
        return f"""
            using storage_T = {self.lane.storage_type};
            using scalar_T = {self.lane.scalar_type};
            using complex_T = fft_complex_t<scalar_T>;
            const device storage_T* storage_in =
                reinterpret_cast<const device storage_T*>(inp);
            device storage_T* storage_out =
                reinterpret_cast<device storage_T*>(out);
            threadgroup complex_T shared_in[{threadgroup_mem}];
        """

    def _build(self) -> None:
        if self.plan.kind == "four-step":
            for step in (0, 1):
                current = step_plan(self.plan, step)
                total_batch = self.batch * (self.plan.n2 if step == 0 else self.plan.n1)
                threadgroup_mem, grid, threadgroup = dispatch_geometry(
                    current.n,
                    compute_elems_per_thread(current),
                    total_batch,
                    four_step=True,
                )
                header = specialized_fft_header(self.root, current, self.inverse)
                source = self._common_source(threadgroup_mem) + f"""
                    four_step_fft<
                        {threadgroup_mem}, storage_T, storage_T, {step}, false>(
                            storage_in,
                            storage_out,
                            &shared_in[0],
                            {self.plan.n1},
                            {self.plan.n2},
                            {total_batch},
                            thread_position_in_grid,
                            threads_per_grid);
                """
                kernel = self._kernel(str(step), header, source, ["inp"])
                self.kernels.append((kernel, grid, threadgroup, []))
            return

        threadgroup_mem, grid, threadgroup = dispatch_geometry(
            self.plan.fft_n,
            compute_elems_per_thread(self.plan),
            self.batch,
            four_step=False,
        )
        header = specialized_fft_header(self.root, self.plan, self.inverse)
        source = self._common_source(threadgroup_mem)
        input_names = ["inp"]
        extras: list[Any] = []
        if self.plan.kind == "stockham":
            source += f"""
                fft<{threadgroup_mem}, storage_T, storage_T>(
                    storage_in,
                    storage_out,
                    &shared_in[0],
                    {self.plan.n},
                    {self.batch},
                    thread_position_in_grid,
                    threads_per_grid);
            """
        elif self.plan.kind == "rader":
            b_q, g_q, g_minus_q = rader_constants(self.plan, self.lane)
            extras = [b_q, g_q, g_minus_q]
            input_names += ["b_q", "g_q", "g_minus_q"]
            source += f"""
                rader_fft<{threadgroup_mem}, storage_T, storage_T>(
                    storage_in,
                    storage_out,
                    reinterpret_cast<const device complex_T*>(b_q),
                    reinterpret_cast<const device short*>(g_q),
                    reinterpret_cast<const device short*>(g_minus_q),
                    &shared_in[0],
                    {self.plan.n},
                    {self.batch},
                    {self.plan.rader_n},
                    thread_position_in_grid,
                    threads_per_grid);
            """
        elif self.plan.kind == "bluestein":
            w_q, w_k = bluestein_constants(self.plan, self.lane)
            extras = [w_q, w_k]
            input_names += ["w_q", "w_k"]
            source += f"""
                bluestein_fft<{threadgroup_mem}, storage_T, storage_T>(
                    storage_in,
                    storage_out,
                    reinterpret_cast<const device complex_T*>(w_q),
                    reinterpret_cast<const device complex_T*>(w_k),
                    &shared_in[0],
                    {self.plan.n},
                    {self.plan.fft_n},
                    {self.batch},
                    thread_position_in_grid,
                    threads_per_grid);
            """
        else:
            raise ValueError(f"unsupported plan {self.plan.kind}")
        kernel = self._kernel("0", header, source, input_names)
        self.kernels.append((kernel, grid, threadgroup, extras))

    def run(self, values: Any) -> Any:
        result = values
        for kernel, grid, threadgroup, extras in self.kernels:
            result = kernel(
                inputs=[result, *extras],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=[result.shape],
                output_dtypes=[self.lane.dtype],
                stream=mx.gpu,
            )[0]
        return result


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    error = actual - expected
    reference_rms = np.sqrt(np.mean(np.abs(expected) ** 2))
    rmse = np.sqrt(np.mean(np.abs(error) ** 2))
    normalized_rmse = float(rmse / reference_rms) if reference_rms else float(rmse)
    snr_db = -20.0 * math.log10(max(normalized_rmse, np.finfo(float).tiny))
    return {
        "max_abs": float(np.max(np.abs(error))),
        "normalized_rmse": normalized_rmse,
        "snr_db": snr_db,
    }


def correctness_case(
    root: Path, plan: FFTPlan, lane: Lane, batch: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + plan.n)
    host_values = rng.normal(scale=0.1, size=(batch, plan.n, 2)).astype(np.float32)
    values = mx.array(host_values, dtype=lane.dtype)
    quantized_input = lanes_to_complex(values)

    forward = ReducedComplexFFTRunner(root, plan, lane, batch, inverse=False)
    output = forward.run(values)
    mx.eval(output)
    expected = np.fft.fft(quantized_input, axis=1)

    inverse = ReducedComplexFFTRunner(root, plan, lane, batch, inverse=True)
    roundtrip = inverse.run(output)
    mx.eval(roundtrip)
    result = {
        "case": plan.kind,
        "n": plan.n,
        "lane": lane.name,
        "bytes_per_complex": lane.bytes_per_complex,
        "forward": error_metrics(lanes_to_complex(output), expected),
        "roundtrip": error_metrics(lanes_to_complex(roundtrip), quantized_input),
    }
    return result


def benchmark_runner(
    runner: ReducedComplexFFTRunner,
    values: Any,
    warmup: int,
    samples: int,
    dispatches_per_sample: int,
) -> dict[str, float]:
    for _ in range(warmup):
        outputs = [runner.run(values) for _ in range(dispatches_per_sample)]
        mx.eval(*outputs)
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        outputs = [runner.run(values) for _ in range(dispatches_per_sample)]
        mx.eval(*outputs)
        timings.append((time.perf_counter() - start) * 1e3 / dispatches_per_sample)
    median_ms = statistics.median(timings)
    complex_values = runner.batch * runner.plan.n
    raw_io_bytes = complex_values * runner.lane.bytes_per_complex * 2
    return {
        "median_ms": median_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "raw_io_gbps": raw_io_bytes / (median_ms * 1e6),
        "million_complex_per_second": complex_values / (median_ms * 1e3),
    }


def static_instantiations(lanes: list[Lane]) -> str:
    lines = [
        "",
        "using proof_complex32_t = complex_t<half>;",
        "using proof_bcomplex32_t = complex_t<bfloat16_t>;",
        "#define instantiate_reduced_c2c(tg_mem_size, type) \\",
        "  instantiate_fft(tg_mem_size, type, type) \\",
        "  instantiate_rader(tg_mem_size, type, type) \\",
        "  instantiate_bluestein(tg_mem_size, type, type) \\",
        "  instantiate_four_step(tg_mem_size, type, type, 0, false) \\",
        "  instantiate_four_step(tg_mem_size, type, type, 1, false)",
    ]
    for lane in lanes:
        storage_type = {
            "half": "proof_complex32_t",
            "bfloat": "proof_bcomplex32_t",
        }[lane.name]
        for memory in (256, 512, 1024, 2048, 4096):
            lines.append(f"instantiate_reduced_c2c({memory}, {storage_type})")
    return "\n".join(lines) + "\n"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or "warning: ignoring file" in result.stderr:
        raise RuntimeError(
            f"command failed ({' '.join(command)}):\n{result.stdout}{result.stderr}"
        )
    return result


def find_air_directory(build_directory: Path) -> Path:
    direct = build_directory / "fft.air"
    if direct.exists():
        return build_directory
    matches = list(build_directory.rglob("fft.air"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one fft.air below {build_directory}, found {len(matches)}"
        )
    return matches[0].parent


def cmake_deployment_target(build_directory: Path) -> str:
    matches = list(build_directory.rglob("CMakeCache.txt"))
    for cache in matches:
        match = re.search(
            r"^CMAKE_OSX_DEPLOYMENT_TARGET(?::\w+)?=(.+)$",
            cache.read_text(),
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1).strip()
    raise ValueError(f"deployment target not found below {build_directory}")


def compile_static_audit(
    root: Path,
    deployment_target: str,
    selected_lanes: list[Lane],
    full_metallib_build: Path | None,
) -> dict[str, Any]:
    source = (root / "mlx/backend/metal/kernels/fft.metal").read_text()
    configurations: list[tuple[str, list[Lane]]] = [("baseline", [])]
    configurations.extend((lane.name, [lane]) for lane in selected_lanes)
    if len(selected_lanes) > 1:
        configurations.append(("combined", selected_lanes))

    full_air_files: list[Path] | None = None
    if full_metallib_build:
        air_directory = find_air_directory(full_metallib_build)
        full_air_files = sorted(
            path for path in air_directory.glob("*.air") if path.name != "fft.air"
        )
        if not full_air_files:
            raise ValueError(f"no companion AIR modules found in {air_directory}")

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="mlx-reduced-fft-audit-") as temp:
        temp_path = Path(temp)
        for name, lanes in configurations:
            metal_path = temp_path / f"{name}.metal"
            air_path = temp_path / f"{name}.air"
            library_path = temp_path / f"{name}.metallib"
            metal_path.write_text(source + static_instantiations(lanes))
            start = time.perf_counter()
            run_command(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-x",
                    "metal",
                    "-Wall",
                    "-Wextra",
                    "-fno-fast-math",
                    "-Wno-c++17-extensions",
                    "-Wno-c++20-extensions",
                    f"-mmacosx-version-min={deployment_target}",
                    "-c",
                    str(metal_path),
                    f"-I{root}",
                    "-o",
                    str(air_path),
                ]
            )
            run_command(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    f"-mmacosx-version-min={deployment_target}",
                    str(air_path),
                    "-o",
                    str(library_path),
                ]
            )
            symbols = run_command(
                ["xcrun", "metal-nm", "-g", str(library_path)]
            ).stdout.splitlines()
            results[name] = {
                "metallib_bytes": library_path.stat().st_size,
                "exported_symbols": len(symbols),
                "compile_seconds": time.perf_counter() - start,
            }
            if full_air_files:
                full_library_path = temp_path / f"{name}-full.metallib"
                run_command(
                    [
                        "xcrun",
                        "-sdk",
                        "macosx",
                        "metal",
                        f"-mmacosx-version-min={deployment_target}",
                        *(str(path) for path in full_air_files),
                        str(air_path),
                        "-o",
                        str(full_library_path),
                    ]
                )
                results[name]["full_metallib_bytes"] = full_library_path.stat().st_size

    baseline = results["baseline"]["metallib_bytes"]
    for value in results.values():
        value["delta_bytes"] = value["metallib_bytes"] - baseline
        value["delta_percent"] = (
            100.0 * value["delta_bytes"] / baseline if baseline else 0.0
        )
        if "full_metallib_bytes" in value:
            full_baseline = results["baseline"]["full_metallib_bytes"]
            value["full_delta_bytes"] = value["full_metallib_bytes"] - full_baseline
            value["full_delta_percent"] = (
                100.0 * value["full_delta_bytes"] / full_baseline
            )
    return results


def git_commit(root: Path) -> str:
    return run_command(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip()


def parse_names(value: str, choices: dict[str, Any]) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = sorted(set(names) - set(choices))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown values: {', '.join(unknown)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=",".join(CASES),
        help=f"comma-separated subset of: {', '.join(CASES)}",
    )
    parser.add_argument(
        "--lanes",
        default=",".join(LANES),
        help=f"comma-separated subset of: {', '.join(LANES)}",
    )
    parser.add_argument("--correctness-batch", type=int, default=3)
    parser.add_argument("--benchmark-elements", type=int, default=2**22)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--dispatches-per-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--static-audit", action="store_true")
    parser.add_argument(
        "--full-metallib-build",
        type=Path,
        help="CMake build tree whose AIR modules should be relinked for full-size deltas",
    )
    parser.add_argument(
        "--deployment-target",
        help="Metal deployment target; defaults to 14.0 or the supplied CMake build",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    try:
        case_names = parse_names(args.cases, CASES)
        lane_names = parse_names(args.lanes, LANES)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    for name in (
        "correctness_batch",
        "benchmark_elements",
        "warmup",
        "samples",
        "dispatches_per_sample",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    plans = [plan_fft(CASES[name]) for name in case_names]
    lanes = [LANES[name] for name in lane_names]
    deployment_target = args.deployment_target or "14.0"
    if args.full_metallib_build:
        build_target = cmake_deployment_target(args.full_metallib_build)
        if args.deployment_target and args.deployment_target != build_target:
            parser.error(
                "--deployment-target must match the full metallib build "
                f"({build_target})"
            )
        deployment_target = build_target

    results: dict[str, Any] = {
        "configuration": {
            "commit": git_commit(root),
            "device": mx.device_info(),
            "platform": platform.platform(),
            "cases": case_names,
            "lanes": lane_names,
            "correctness_batch": args.correctness_batch,
            "deployment_target": deployment_target,
            "dispatches_per_sample": args.dispatches_per_sample,
            "benchmark_elements": args.benchmark_elements,
            "warmup": args.warmup,
            "samples": args.samples,
            "seed": args.seed,
        },
        "correctness": [],
        "performance": [],
    }
    print("Correctness")
    for plan in plans:
        for lane in lanes:
            result = correctness_case(
                root, plan, lane, args.correctness_batch, args.seed
            )
            results["correctness"].append(result)
            forward = result["forward"]
            roundtrip = result["roundtrip"]
            print(
                f"  {plan.kind:10s} n={plan.n:5d} {lane.name:6s} "
                f"forward_nrmse={forward['normalized_rmse']:.6g} "
                f"forward_snr={forward['snr_db']:.2f}dB "
                f"roundtrip_nrmse={roundtrip['normalized_rmse']:.6g}"
            )

    if not args.no_benchmark:
        print("Performance")
        for plan in plans:
            case_results = []
            for lane in lanes:
                batch = max(1, math.ceil(args.benchmark_elements / plan.n))
                rng = np.random.default_rng(args.seed + plan.n + lane.bytes_per_complex)
                host_values = rng.normal(scale=0.1, size=(batch, plan.n, 2)).astype(
                    np.float32
                )
                values = mx.array(host_values, dtype=lane.dtype)
                runner = ReducedComplexFFTRunner(root, plan, lane, batch, inverse=False)
                metrics = benchmark_runner(
                    runner,
                    values,
                    args.warmup,
                    args.samples,
                    args.dispatches_per_sample,
                )
                entry = {
                    "case": plan.kind,
                    "n": plan.n,
                    "batch": batch,
                    "lane": lane.name,
                    "bytes_per_complex": lane.bytes_per_complex,
                    **metrics,
                }
                case_results.append(entry)
                results["performance"].append(entry)
            float_result = next(
                (entry for entry in case_results if entry["lane"] == "float"), None
            )
            for entry in case_results:
                if float_result:
                    entry["speedup_vs_float"] = (
                        float_result["median_ms"] / entry["median_ms"]
                    )
                print(
                    f"  {plan.kind:10s} n={plan.n:5d} {entry['lane']:6s} "
                    f"median={entry['median_ms']:.4f}ms "
                    f"raw_io={entry['raw_io_gbps']:.2f}GB/s "
                    f"vs_float={entry.get('speedup_vs_float', 1.0):.3f}x"
                )

    if args.static_audit:
        print("Static FFT metallib audit")
        reduced_lanes = [lane for lane in lanes if lane.name != "float"]
        if not reduced_lanes:
            parser.error("--static-audit requires the half or bfloat lane")
        audit = compile_static_audit(
            root,
            deployment_target,
            reduced_lanes,
            args.full_metallib_build,
        )
        results["static_audit"] = audit
        for name, value in audit.items():
            print(
                f"  {name:8s} {value['metallib_bytes']:,}B "
                f"delta={value['delta_bytes']:+,}B "
                f"({value['delta_percent']:+.4f}%) "
                f"symbols={value['exported_symbols']}"
            )
            if "full_metallib_bytes" in value:
                print(
                    f"           full={value['full_metallib_bytes']:,}B "
                    f"delta={value['full_delta_bytes']:+,}B "
                    f"({value['full_delta_percent']:+.4f}%)"
                )

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
