use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use std::fmt;
use tch::{Kind, Tensor};

/// Tensor to Numpy conversion utilities
pub struct TensorConverter;

impl TensorConverter {
    /// Generic tch::Tensor to numpy array
    fn tensor_to_numpy(py: Python, tensor: &Tensor, kind: Kind) -> PyResult<PyObject>
    where
        T: Element + Clone,
        Vec<T>: TryFrom<Tensor>,
        <Vec<T> as TryFrom<Tensor>>::Error: fmt::Display,
    {
        let shape = tensor.size();
        let data: Vec<T> = tensor
            .to_kind(kind)
            .try_into()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract tensor data: {e}")))?;

        match shape.len() {
            1 => {
                let array = PyArray1::from_vec(py, data);
                Ok(array.into_pyobject(py)?.unbind())
            }
            2 => {
                let rows = shape[0] as usize;
                let cols = shape[1] as usize;
                let vec2d: Vec<Vec<T>> = data.chunks(cols).map(|chunk| chunk.to_vec()).collect();

                if vec2d.len() != rows {
                    return Err(PyRuntimeError::new_err(format!(
                        "Tensor shape mismatch during reshaping: expected {} rows, got {}",
                        rows,
                        vec2d.len()
                    )));
                }

                let array = PyArray2::from_vec2(py, &vec2d)?;
                Ok(array.into_pyobject(py)?.unbind())
            }
            _ => Err(PyRuntimeError::new_err(format!(
                "Unsupported tensor dimensions {:?}D for conversion to numpy",
                tensor.kind()
            ))),
        }
    }

    /// Convert tch::Tensor to numpy array (integer tensors)
    pub fn tensor_to_numpy_i64(py: Python, tensor: &Tensor) -> PyResult<PyObject> {
        tensor_to_numpy::<i64>(py, tensor, Kind::Int64)
    }

    /// Convert tch::Tensor to numpy array (float32 Tensor)
    pub fn tensor_to_numpy_f32(py: Python, tensor: &Tensor) -> PyResult<PyObject> {
        tensor_to_numpy::<f32>(py, tensor, Kind::Float)
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
