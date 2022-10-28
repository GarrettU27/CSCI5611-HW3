use crate::matrix::{Matrix, InvalidArgumentError};

pub struct NeuralNetwork {
    use_relu: Vec<bool>,
    weights: Vec<Matrix<isize>>,
    biases: Vec<Matrix<isize>>
}

impl NeuralNetwork {
    pub fn new(use_relu: Vec<bool>, weights: Vec<Matrix<isize>>, biases: Vec<Matrix<isize>>) -> Result<Self, InvalidArgumentError> {
        if use_relu.len() != weights.len() || weights.len() != biases.len() {
            return Err(InvalidArgumentError)
        }

        Ok(NeuralNetwork {
            use_relu,
            weights,
            biases
        })
    }

    pub fn compute_output(&self, input: Matrix<isize>) {

    }
}

fn relu(x: isize) -> isize {
    if x > 0 {
        x
    } else {
        0
    }
}