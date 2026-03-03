import numpy as np
import numexpr as ne
import jax
from jax import numpy as jnp
from jax import jit
import time
from arrays import rust_array, rust_with_pow, rust_with_div_pow, rust_array_par, rust_array_par_pow, rust_array_par_pow_chunk # ty: ignore[unresolved-import]
import matplotlib.pyplot as plt

def numpy_array(a, b, c, d):
    return ((a*b)/c)**d

def numpy_no_alloc_explicit(a, b, c, d):
    # we need to create an empty array of the same type and shape
    out = np.empty_like(a)
    np.multiply(a, b, out=out)
    np.divide(out, c, out=out)
    np.power(out, d, out=out)
    return out

def ne_evaluate(a, b, c, d):
    return ne.evaluate("((a*b)/c)**d")

def list_comprehension(a_list, b_list, c_list, d_list):
    e_list = [a*b for a, b, in zip(a_list, b_list)]
    f_list = [e/c for e, c, in zip(e_list, c_list)]
    g_list = [f**d for f, d, in zip(f_list, d_list)]
    return g_list

def dumb_loop(a_array, b_array, c_array, d_array):
    n = len(a_array)
    out_array = np.zeros(n)

    for i, (a_scalar, b_scalar, c_scalar, d_scalar), in enumerate(zip(a_array, b_array, c_array, d_array)):
        out_array[i] = ((a_scalar*b_scalar)/c_scalar)**d_scalar

    return out_array

def bench(n):
    np_times = []
    np_ip_times = []
    ne_times = []
    jax_times = []
    # list_times = []
    # loop_times = []
    rust_times = []
    rust_par_times = []
    rust_pow_times = []
    rust_par_pow_times = []
    rust_par_pow_chunk_times = []
    rust_div_pow_times = []

    for k in np.arange(1000):
        a = np.random.rand(n) * 100
        b = np.random.rand(n) * 100
        c = np.random.rand(n) * 100
        d = np.random.rand(n)

        a_jarray = jnp.array(a)
        b_jarray = jnp.array(b)
        c_jarray = jnp.array(c)
        d_jarray = jnp.array(d)

        # start = time.perf_counter_ns()
        # list_res = list_comprehension(a, b, c, d)
        # t = time.perf_counter_ns() - start
        # list_times.append(t)
        #
        # start = time.perf_counter_ns()
        # loop_res = dumb_loop(a, b, c, d)
        # t = time.perf_counter_ns() - start
        # loop_times.append(t)

        start = time.perf_counter_ns()
        rust_res = rust_array(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_times.append(t)

        start = time.perf_counter_ns()
        rust_pow_res = rust_with_pow(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_pow_times.append(t)

        start = time.perf_counter_ns()
        rust_par_res = rust_array_par(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_par_times.append(t)

        start = time.perf_counter_ns()
        rust_par_pow_res = rust_array_par_pow(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_par_pow_times.append(t)

        start = time.perf_counter_ns()
        rust_par_pow_chunk_res = rust_array_par_pow_chunk(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_par_pow_chunk_times.append(t)

        start = time.perf_counter_ns()
        np_res = numpy_array(a, b, c, d)
        t = time.perf_counter_ns() - start
        np_times.append(t)

        start = time.perf_counter_ns()
        np_ip_res = numpy_no_alloc_explicit(a, b, c, d)
        t = time.perf_counter_ns() - start
        np_ip_times.append(t)

        start = time.perf_counter_ns()
        ne_res = ne_evaluate(a, b, c, d)
        t = time.perf_counter_ns() - start
        ne_times.append(t)

        start = time.perf_counter_ns()
        jax_evaluate = jit(numpy_array)
        jax_res = jax_evaluate(a_jarray, b_jarray, c_jarray, d_jarray)
        t = time.perf_counter_ns() - start
        jax_times.append(t)

        start = time.perf_counter_ns()
        divpow_res = rust_with_div_pow(a, b, c, d)
        t = time.perf_counter_ns() - start
        rust_div_pow_times.append(t)

        assert(np.allclose(rust_res, rust_pow_res, rtol=1e-9))
        assert(np.allclose(rust_res, np_res, rtol=1e-9))
        assert(np.allclose(rust_res, np_ip_res, rtol=1e-9))
        assert(np.allclose(rust_res, ne_res, rtol=1e-9))
        assert(np.allclose(rust_res, divpow_res, rtol=1e-9))
        # assert(np.allclose(rust_res, list_res, rtol=1e-9))
        # assert(np.allclose(rust_res, loop_res, rtol=1e-9))
        assert(np.allclose(rust_res, rust_par_res, rtol=1e-9))
        assert(np.allclose(rust_res, rust_par_pow_res, rtol=1e-9))
        assert(np.allclose(rust_res, rust_par_pow_chunk_res, rtol=1e-9))
        assert(np.allclose(rust_res, jax_res, rtol=1e-6))

    mean_times = {
        "np_time": np.array(np_times).mean(),
        "np_in_place_time": np.array(np_ip_times).mean(),
        "ne_time": np.array(ne_times).mean(),
        "jax_time": np.array(jax_times).mean(),
        # "list_time": np.array(list_times).mean(),
        # "loop_time": np.array(loop_times).mean(),
        "rust_time": np.array(rust_times).mean(),
        "rust_par_time": np.array(rust_par_times).mean(),
        "rust_pow_time": np.array(rust_pow_times).mean(),
        "rust_par_pow_time": np.array(rust_par_pow_times).mean(),
        "rust_par_pow_chunk_time": np.array(rust_par_pow_chunk_times).mean(),
        "rust_div_pow_time": np.array(rust_div_pow_times).mean(),
    }

    return mean_times

def plot_results(all_mean_times):
    counts = [x[0] for x in all_mean_times]
    
    labels = {
        "np_time": "NumPy",
        "np_in_place_time": "NumPy (in-place)",
        "ne_time": "NumExpr",
        "jax_time": "JAX",
        # "list_time": "List Comprehension",
        # "loop_time": "Dumb Loop",
        "rust_time": "Rust",
        "rust_pow_time": "Rust+pow",
        "rust_div_pow_time": "Rust+div+pow",
        "rust_par_time": "Rust+par",
        "rust_par_pow_time": "Rust+par+pow",
        "rust_par_pow_chunk_time": "Rust+par+pow_chunk",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for key, label in labels.items():
        ratios = [x[1]["np_time"] / x[1][key] for x in all_mean_times]
        ax.plot(counts, ratios, marker='o', markersize=4, label=label)

    ax.set_xscale('log')
    ax.set_xlabel('Count')
    ax.set_ylabel('Speedup vs NumPy')
    ax.set_title('Benchmark: Speedup Relative to NumPy')
    ax.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

all_mean_times = []
for i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    all_mean_times.append((i, bench(i)))

plot_results(all_mean_times)

