import numpy as np
import time
from arrays import rust_array, rust_with_pow, rust_with_div_pow

def numpy_array(a, b, c, d):
    return ((a*b)/c)**d

rust_times = []
rust_pow_times = []
np_times = []
rust_div_pow_times = []

for k in np.arange(1000):
    a = np.random.rand(1000) * 100
    b = np.random.rand(1000) * 100
    c = np.random.rand(1000) * 100
    d = np.random.rand(1000)

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
    # print("Rust: ", rust_res[0:10])
    # print("Divpow: ", divpow_res[0:10])

print("Mean times") 
print("NumPy: ", np.array(np_times).mean())
print("Rust: ", np.array(rust_times).mean())
print("Rust+pow: ", np.array(rust_pow_times).mean())
print("Rust+div+pow: ", np.array(rust_div_pow_times).mean())



