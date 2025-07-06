use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::fmt;
use tch::{Kind, Tensor};

/// Tensor to Numpy conversion utilities
pub struct TensorConverter;

impl TensorConverter {
    fn create_empty_array_2d<T: numpy::Element + Clone>(
        py: Python,
        rows: usize,
        cols: usize,
    ) -> PyResult<&PyArray2<T, D>> {
        let array = unsafe { PyArray2::<T>::new(py, [rows, cols], false) };
        Ok(array)
    }

    fn create_empty_array_1d<T: numpy::Element + Clone>(
        py: Python,
        length: usize,
    ) -> PyResult<&PyArray1<T, D>> {
        let array = unsafe { PyArray1::<T>::new(py, [length], false) };
        Ok(array)
    }

    /// Generic tch::Tensor to numpy array
    fn tensor_to_numpy_generic<T>(py: Python, tensor: &Tensor, kind: Kind) -> PyResult<PyObject>
    where
        T: numpy::Element + Clone,
        Vec<T>: TryFrom<Tensor>,
        <Vec<T> as TryFrom<Tensor>>::Error: fmt::Display,
    {
        let shape = tensor.size();
        let data: Vec<T> = tensor
            .to_kind(kind)
            .contiguous()
            .view([-1])
            .try_into()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract tensor data: {e}")))?;

        match shape.len() {
            1 => {
                let length = shape[0] as usize;

                if length == 0 {
                    let array = Self::create_empty_array_1d(py, length);
                    Ok(array.into_pyobject(py)?.unbind().into());
                }
                let array = PyArray1::from_vec(py, data);
                Ok(array.into_pyobject(py)?.unbind().into())
            }
            2 => {
                let rows = shape[0] as usize;
                let cols = shape[1] as usize;
                let vec2d: Vec<Vec<T>> = data.chunks(cols).map(|chunk| chunk.to_vec()).collect();

                // Handle zero-dimension early
                if rows == 0 || cols == 0 {
                    // Return empty PyArray2 with shape [rows, cols]
                    let array = Self::create_empty_array_2d(py, rows, cols);
                    return Ok(array.into_pyobject(py)?.unbind().into());
                }

                if vec2d.len() != rows {
                    return Err(PyRuntimeError::new_err(format!(
                        "Tensor shape mismatch during reshaping: expected {} rows, got {}",
                        rows,
                        vec2d.len()
                    )));
                }

                let array = PyArray2::from_vec2(py, &vec2d)?;
                Ok(array.into_pyobject(py)?.unbind().into())
            }
            _ => Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor dimensions {:?}D for conversion to numpy",
                tensor.kind()
            ))),
        }
    }

    /// Convert tch::Tensor to numpy array (integer tensors)
    pub fn tensor_to_numpy_i64(py: Python, tensor: &Tensor) -> PyResult<PyObject> {
        Self::tensor_to_numpy_generic::<i64>(py, tensor, Kind::Int64)
    }

    /// Convert tch::Tensor to numpy array (float32 Tensor)
    pub fn tensor_to_numpy_f32(py: Python, tensor: &Tensor) -> PyResult<PyObject> {
        Self::tensor_to_numpy_generic::<f32>(py, tensor, Kind::Float)
    }

    /// Generic tensor to numpy conversion with type detection
    pub fn tensor_to_numpy(py: Python, tensor: &Tensor) -> PyResult<PyObject> {
        match tensor.kind() {
            Kind::Int64 | Kind::Int16 | Kind::Int8 | Kind::Int => {
                Self::tensor_to_numpy_i64(py, tensor)
            }
            Kind::Float | Kind::Double => Self::tensor_to_numpy_f32(py, tensor),
            _ => Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor type: {:?}",
                tensor.kind()
            ))),
        }
    }
}
