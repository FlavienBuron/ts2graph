use std::f32;
use std::f64::consts::LN_2;
use std::fmt;

#[derive(Debug, Clone)]
pub struct GraphError(pub String);

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph error: {}", self.0)
    }
}

impl std::error::Error for GraphError {}

impl From<GraphError> for PyErr {
    fn from(value: GraphError) -> PyErr {
        PyValueError::new_err(value.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum DecayFunction {
    Exponential,
    Inverse,
    InverseSquare,
    Logarithmic,
    Linear,
}

impl DecayFunction {
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str {
            n if n.contains("exp") => Some(Self::Exponential),
            n if n.contains("inv") && !n.contains("squ") => Some(Self::Inverse),
            n if n.contains("squ") => Some(Self::InverseSquare),
            n if n.contains("log") => Some(Self::Logarithmic),
            n if n.contains("linear") => Some(Self::Linear),
            _ => None,
        }
    }

    pub fn decay_value(&self, hop: i64, max_hop: i64) -> f32 {
        match self {
            DecayFunction::Exponential => 0.9f32.powi(hop as i32),
            DecayFunction::Inverse => {
                if hop == 0 {
                    1.0
                } else {
                    1.0 / (hop as f32)
                }
            }
            DecayFunction::InverseSquare => {
                if hop == 0 {
                    1.0
                } else {
                    1.0 / (hop * hop) as f32
                }
            }
            DecayFunction::Logarithmic => {
                if hop > 0 {
                    1.0 / (1.0 + (hop as f64).ln()) as f32
                } else {
                    1.0
                }
            }
            DecayFunction::Linear => f32::max(0.0, 1.0 - ((hop - 1) as f32 / max_hop as f32)),
        }
    }
}
