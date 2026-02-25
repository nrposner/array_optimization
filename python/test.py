import numpy as np
import time
from arrays import rust_array, rust_with_pow, rust_with_div_pow

def numpy_array(a, b, c, d):
    return ((a*b)/c)**d

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
list_times = []
loop_times = []
rust_times = []
rust_pow_times = []
rust_div_pow_times = []

for k in np.arange(1000):
    a = np.random.rand(1000) * 100
    b = np.random.rand(1000) * 100
    c = np.random.rand(1000) * 100
    d = np.random.rand(1000)

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
    np_res = numpy_array(a, b, c, d)
    t = time.perf_counter_ns() - start
    np_times.append(t)

    start = time.perf_counter_ns()
    divpow_res = rust_with_div_pow(a, b, c, d)
    t = time.perf_counter_ns() - start
    rust_div_pow_times.append(t)

    assert(np.allclose(rust_res, rust_pow_res, rtol=1e-9))
    assert(np.allclose(rust_res, np_res, rtol=1e-9))
    assert(np.allclose(rust_res, divpow_res, rtol=1e-9))
    assert(np.allclose(rust_res, list_res, rtol=1e-9))
    assert(np.allclose(rust_res, loop_res, rtol=1e-9))

numpy_mean = np.array(np_times).mean()
print("Mean times") 
print("NumPy:              ", numpy_mean, " 1.0x")
print("List Comprehension: ", np.array(list_times).mean(), f" {numpy_mean/np.array(list_times).mean():.3f}x")
print("Dumb Loop:          ", np.array(loop_times).mean(), f" {numpy_mean/np.array(loop_times).mean():.3f}x")
print("Rust:               ", np.array(rust_times).mean(), f" {numpy_mean / np.array(rust_times).mean():.3f}x")
print("Rust+pow:           ", np.array(rust_pow_times).mean(), f" {numpy_mean / np.array(rust_pow_times).mean():.3f}x")
print("Rust+div+pow:       ", np.array(rust_div_pow_times).mean(), f" {numpy_mean / np.array(rust_div_pow_times).mean():.3f}x")



