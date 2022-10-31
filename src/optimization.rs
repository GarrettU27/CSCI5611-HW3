use rand::Rng;
use crate::{matrix::Matrix, neural_network::NeuralNetwork};

struct OptimizedInput {
    guess: Matrix<f64>,
    neural_network: NeuralNetwork
}

impl OptimizedInput {
    fn new(guess: Matrix<f64>, neural_network: NeuralNetwork) -> Self {
        Self {
            guess,
            neural_network
        }
    }

    fn loss(output: &Matrix<f64>) -> f64 {
        let mut loss = 0.0;

        for row in output.rows {
            for val in row {
                loss += val.abs();
            }
        }

        loss
    }

    fn random_search(&mut self) {
        let mut rng = rand::thread_rng();
        let mut output = self.neural_network.compute_output(self.guess).unwrap();

        while Self::loss(&output) > 0.1 {

        }
    }
}