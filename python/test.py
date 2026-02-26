import sys
import numpy as np
import numexpr as ne
import jax
from jax import numpy as jnp
from jax import jit
import time
from arrays import rust_array, rust_with_pow, rust_with_div_pow, rust_array_par, rust_array_par_pow, rust_array_par_pow_chunk

try:
    n = int(sys.argv[1])
except IndexError:
    n = 10**3

def numpy_array(a, b, c, d):
    return ((a*b)/c)**d

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

np_times = []
ne_times = []
jax_times = []
list_times = []
loop_times = []
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

    start = time.perf_counter_ns()
    list_res = list_comprehension(a, b, c, d)
    t = time.perf_counter_ns() - start
    list_times.append(t)

    start = time.perf_counter_ns()
    loop_res = dumb_loop(a, b, c, d)
    t = time.perf_counter_ns() - start
    loop_times.append(t)

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
    assert(np.allclose(rust_res, ne_res, rtol=1e-9))
    assert(np.allclose(rust_res, divpow_res, rtol=1e-9))
    assert(np.allclose(rust_res, list_res, rtol=1e-9))
    assert(np.allclose(rust_res, loop_res, rtol=1e-9))
    assert(np.allclose(rust_res, rust_par_res, rtol=1e-9))
    assert(np.allclose(rust_res, rust_par_pow_res, rtol=1e-9))
    assert(np.allclose(rust_res, rust_par_pow_chunk_res, rtol=1e-9))
    assert(np.allclose(rust_res, jax_res, rtol=1e-6))

numpy_mean = np.array(np_times).mean()
print("Mean times") 
print("NumPy:              ", " 1.0x")
print("NumExpr:            ", f" {numpy_mean/np.array(ne_times).mean():.3f}x")
print("Jax:                ", f" {numpy_mean/np.array(jax_times).mean():.3f}x")
print("List Comprehension: ", f" {numpy_mean/np.array(list_times).mean():.3f}x")
print("Dumb Loop:          ", f" {numpy_mean/np.array(loop_times).mean():.3f}x")
print("Rust:               ", f" {numpy_mean / np.array(rust_times).mean():.3f}x")
print("Rust+pow:           ", f" {numpy_mean / np.array(rust_pow_times).mean():.3f}x")
print("Rust+div+pow:       ", f" {numpy_mean / np.array(rust_div_pow_times).mean():.3f}x")
print("Rust+par:           ", f" {numpy_mean / np.array(rust_par_times).mean():.3f}x")
print("Rust+par+pow:       ", f" {numpy_mean / np.array(rust_par_pow_times).mean():.3f}x")
print("Rust+par+pow_chunk: ", f" {numpy_mean / np.array(rust_par_pow_chunk_times).mean():.3f}x")

print(f"{n}, 1.0, {numpy_mean/np.array(ne_times).mean():.3f}, {numpy_mean/np.array(jax_times).mean():.3f}, {numpy_mean/np.array(list_times).mean():.3f}, {numpy_mean/np.array(loop_times).mean():.3f}, {numpy_mean / np.array(rust_times).mean():.3f}, {numpy_mean / np.array(rust_pow_times).mean():.3f}, {numpy_mean / np.array(rust_div_pow_times).mean():.3f}, {numpy_mean / np.array(rust_par_times).mean():.3f}, {numpy_mean / np.array(rust_par_pow_times).mean():.3f}, {numpy_mean / np.array(rust_par_pow_chunk_times).mean():.3f}")

# print("NumExpr:            ", np.array(ne_times).mean(), f" {numpy_mean/np.array(ne_times).mean():.3f}x")
# print("Jax:                ", np.array(jax_times).mean(), f" {numpy_mean/np.array(jax_times).mean():.3f}x")
# print("List Comprehension: ", np.array(list_times).mean(), f" {numpy_mean/np.array(list_times).mean():.3f}x")
# print("Dumb Loop:          ", np.array(loop_times).mean(), f" {numpy_mean/np.array(loop_times).mean():.3f}x")
# print("Rust:               ", np.array(rust_times).mean(), f" {numpy_mean / np.array(rust_times).mean():.3f}x")
# print("Rust+pow:           ", np.array(rust_pow_times).mean(), f" {numpy_mean / np.array(rust_pow_times).mean():.3f}x")
# print("Rust+div+pow:       ", np.array(rust_div_pow_times).mean(), f" {numpy_mean / np.array(rust_div_pow_times).mean():.3f}x")
# print("Rust+par:           ", np.array(rust_par_times).mean(), f" {numpy_mean / np.array(rust_par_times).mean():.3f}x")
# print("Rust+par+pow:       ", np.array(rust_par_pow_times).mean(), f" {numpy_mean / np.array(rust_par_pow_times).mean():.3f}x")
# print("Rust+par+pow_chunk: ", np.array(rust_par_pow_chunk_times).mean(), f" {numpy_mean / np.array(rust_par_pow_chunk_times).mean():.3f}x")



