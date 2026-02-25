use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
use pyo3::prelude::*;
use vforce::arithmetic::{pow_array_in_place, div_array_in_place};

#[pyfunction]
pub fn rust_array<'py>(
    py: Python<'py>,
    // get array objects via numpy bindings
    a_arr: PyReadonlyArray1<f64>,
    b_arr: PyReadonlyArray1<f64>,
    c_arr: PyReadonlyArray1<f64>,
    d_arr: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {

    // take arrays as read-only slices
    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();
    let c_slice = c_arr.as_slice().unwrap();
    let d_slice = d_arr.as_slice().unwrap();

    // create a new array of the right length and take it as a mutable slice
    let out_arr = unsafe { PyArray1::new(py, a_slice.len(), false)};
    let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

    // our dumb loop goes through element by element, fusing the 
    // math operations into a single step without intermediate allocations
    for (i, (((a, b), c), d)) in a_slice.iter()
        .zip(b_slice)
        .zip(c_slice)
        .zip(d_slice)
        .enumerate() {

        // do the math and load the result directly into the relevant index in the out slice
        out_slice[i] = ((a*b)/c).powf(*d);
    }
    // return the out array to Python
    out_arr
}

#[pyfunction]
pub fn rust_with_pow<'py>(
    py: Python<'py>,
    // get array objects via numpy bindings
    a_arr: PyReadonlyArray1<f64>,
    b_arr: PyReadonlyArray1<f64>,
    c_arr: PyReadonlyArray1<f64>,
    d_arr: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {

    // take arrays as read-only slices
    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();
    let c_slice = c_arr.as_slice().unwrap();
    let d_slice = d_arr.as_slice().unwrap();

    // create a new array of the right length and take it as a mutable slice
    let out_arr = unsafe { PyArray1::new(py, a_slice.len(), false)};
    let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

    pow_helper(a_slice, b_slice, c_slice, d_slice, out_slice);

    // return the out array to Python
    out_arr
}

fn pow_helper(a: &[f64], b: &[f64], c: &[f64], d: &[f64], out: &mut [f64]) {
    #[cfg(target_os = "macos")]
    {
        for (i, ((a, b), c)) in a.iter().zip(b).zip(c).enumerate() {
            out[i] = (a * b) / c;
        }

        let _ = pow_array_in_place(out, d);
    }

    #[cfg(not(target_os = "macos"))]
    {
        for (i, (((a, b), c), d)) in a.iter().zip(b).zip(c).zip(d).enumerate() {
            out[i] = ((a * b) / c).powf(*d);
        }
    }
}


#[pyfunction]
#[cfg(target_os = "macos")]
pub fn rust_with_div_pow<'py>(
    py: Python<'py>,
    // get array objects via numpy bindings
    a_arr: PyReadonlyArray1<f64>,
    b_arr: PyReadonlyArray1<f64>,
    c_arr: PyReadonlyArray1<f64>,
    d_arr: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {

    // take arrays as read-only slices
    let a_slice = a_arr.as_slice().unwrap();
    let b_slice = b_arr.as_slice().unwrap();
    let c_slice = c_arr.as_slice().unwrap();
    let d_slice = d_arr.as_slice().unwrap();

    // create a new array of the right length and take it as a mutable slice
    let out_arr = unsafe { PyArray1::new(py, a_slice.len(), false)};
    let out_slice = unsafe {out_arr.as_slice_mut().unwrap()};

    div_pow_helper(a_slice, b_slice, c_slice, d_slice, out_slice);

    out_arr
}

fn div_pow_helper(a: &[f64], b: &[f64], c: &[f64], d: &[f64], out: &mut [f64]) {
    #[cfg(target_os = "macos")]
    {
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            out[i] = a * b;
        }

        let _ = div_array_in_place(out, c);

        let _ = pow_array_in_place(out, d);
    }

    #[cfg(not(target_os = "macos"))]
    {
        for (i, (((a, b), c), d)) in a.iter().zip(b).zip(c).zip(d).enumerate() {
            out[i] = ((a * b) / c).powf(*d);
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn arrays(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_array, m)?)?;
    m.add_function(wrap_pyfunction!(rust_with_pow, m)?)?;
    m.add_function(wrap_pyfunction!(rust_with_div_pow, m)?)?;
    Ok(())
}
