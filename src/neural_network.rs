use crate::matrix::{Matrix, MatrixError};

#[derive(Debug)]
pub enum NeuralNetworkError {
    InvalidArgument,
    BadInternalState,
    MatrixError(MatrixError)
}

impl From<MatrixError> for NeuralNetworkError {
    fn from(err: MatrixError) -> Self {
        NeuralNetworkError::MatrixError(err)
    }
}

pub struct NeuralNetwork {
    use_relu: Vec<bool>,
    weights: Vec<Matrix<f64>>,
    biases: Vec<Matrix<f64>>
}

impl NeuralNetwork {
    pub fn new(use_relu: Vec<bool>, weights: Vec<Matrix<f64>>, biases: Vec<Matrix<f64>>) -> Result<Self, NeuralNetworkError> {
        if use_relu.len() != weights.len() || weights.len() != biases.len() {
            return Err(NeuralNetworkError::InvalidArgument)
        }

        Ok(NeuralNetwork {
            use_relu,
            weights,
            biases
        })
    }

    pub fn compute_output(&self, input: Matrix<f64>) -> Result<Matrix<f64>, NeuralNetworkError> {
        let mut input_val = input;

        for i in 0..self.weights.len() {
            let weight = self.weights.get(i).ok_or(NeuralNetworkError::BadInternalState)?;
            let bias = self.biases.get(i).ok_or(NeuralNetworkError::BadInternalState)?;
            let use_relu = self.use_relu.get(i).ok_or(NeuralNetworkError::BadInternalState)?;

            input_val = weight.mul(&input_val)?.add(bias)?;

            if *use_relu {
                input_val = relu_matrix(input_val);
            }
        }

        Ok(input_val)
    }
}

fn relu_matrix(x: Matrix<f64>) -> Matrix<f64> {
    let mut rows = Vec::new();

    for row in x.rows {
        let mut new_row = Vec::new();
        for val in row {
            new_row.push(relu(val));
        }
        rows.push(new_row);
    }

    Matrix::new(rows).expect("Somehow, reluing the matrix changed its size")
}

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{neural_network::relu, matrix::Matrix};
    use super::{relu_matrix, NeuralNetwork};

    fn matrix_a() -> Matrix<f64> {
        Matrix::new( vec![
            vec![1.0, -2.0, 3.0, -4.0],
            vec![4.0, -2.0, -2.0, 1.0],
            vec![1.0, -1.0, 1.0, -1.0]
        ]).unwrap()
    }

    #[test]
    fn proper_relu_returned() {
        assert_eq!(relu(4.0), 4.0);
        assert_eq!(relu(-12.0), 0.0);
    }

    #[test]
    fn proper_relu_matrix_returned() {
        let a = relu_matrix(matrix_a());
        let b = Matrix::new( vec![
            vec![1.0, 0.0, 3.0, 0.0],
            vec![4.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0]
        ]).unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn proper_output_computed() {
        let weights = vec![
            Matrix::new( vec![
                vec![-4.0, -5.0, 1.0, -2.0],
                vec![-5.0, 3.0, 0.0, -9.0],
                vec![-4.0, 9.0, -4.0, 5.0],
                vec![3.0, 5.0, 9.0, -2.0],
                vec![-8.0, -6.0, 1.0, 6.0]
            ]).unwrap(),
            Matrix::new( vec![
                vec![9.0, 0.0, 3.0, 6.0, 7.0],
                vec![4.0, -9.0, 8.0, 3.0, -3.0],
                vec![-6.0, -4.0, -3.0, -8.0, 0.0],
                vec![4.0, 9.0, -4.0, 4.0, -8.0],
                vec![-4.0, 0.0, 0.0, -3.0, 6.0]
            ]).unwrap(),
            Matrix::new( vec![
                vec![-4.0, -2.0, 6.0, 0.0, 4.0]
            ]).unwrap()
        ];

        let biases = vec![
            Matrix::new( vec![
                vec![-3.0], 
                vec![-7.0], 
                vec![6.0], 
                vec![-9.0], 
                vec![9.0],
            ]).unwrap(),
            Matrix::new( vec![
                vec![-2.0], 
                vec![7.0], 
                vec![0.0], 
                vec![-5.0], 
                vec![6.0],
            ]).unwrap(),
            Matrix::new( vec![
                vec![0.0], 
            ]).unwrap(),
        ];

        let use_relu = vec![true, true, false];

        let input = Matrix::new( vec![
            vec![1.0], 
            vec![1.0], 
            vec![1.0], 
            vec![1.0], 
        ]).unwrap();

        let neural_network = NeuralNetwork::new(use_relu, weights, biases).unwrap();
        let a = neural_network.compute_output(input).unwrap();
        let b = Matrix::new( vec![
            vec![-566.0], 
        ]).unwrap();

        assert_eq!(a, b);
    }
}