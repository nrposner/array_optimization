import numpy as np
import numexpr as ne
from arrays import rust_array, rust_with_pow, rust_with_div_pow
from timeit import timeit
 
n = 10**5
# a = np.random.rand(10**6)
# b = np.random.rand(10**6)
a = np.random.rand(n) * 100
b = np.random.rand(n) * 100
c = np.random.rand(n) * 100
d = np.random.rand(n)
 
def numpy_test():
    return ((a*b)/c)**d
    # return 2*a + 3*b
 
def numexpr_test():
    # return ne.evaluate("2*a + 3*b")
    return ne.evaluate("((a*b)/c)**d")

def rust_test():
    return rust_array(a, b, c, d)

def rust_pow_test():
    return rust_with_pow(a, b, c, d)
 
print(f"NumPy duration   : {timeit(numpy_test, number=5000):.2f}s")
print(f"NumExpr duration : {timeit(numexpr_test, number=5000):.2f}s")
print(f"Rust duration    : {timeit(rust_test, number=5000):.2f}s")
print(f"Rust_pow duration: {timeit(rust_pow_test, number=5000):.2f}s")
