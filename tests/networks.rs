use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use CSCI5611_HW3::{matrix::Matrix, neural_network::NeuralNetwork};

#[test]
fn proper_outputs_for_networks() {
    let filename = "./networks.txt";
    let file = File::open(filename).expect("Unable to open file");
    let lines = BufReader::new(file).lines();

    let mut weights: Vec<Matrix<f64>> = Vec::new();
    let mut biases: Vec<Matrix<f64>> = Vec::new();
    let mut use_relu: Vec<bool> = Vec::new();

    let mut input: Matrix<f64> = Matrix { rows: Vec::new() };
    let mut output: Matrix<f64> = Matrix { rows: Vec::new() };

    for line_result in lines {
        if let Ok(line) = line_result {
            // println!("{}", line);

            if line == "" {
                let neural_network =
                    NeuralNetwork::new(use_relu.clone(), weights.clone(), biases.clone())
                        .expect("Unable to create proper neural network!");
                let calculated_output = neural_network
                    .compute_output(input)
                    .expect("Unable to compute output!");

                assert_eq!(output, calculated_output);

                weights = Vec::new();
                biases = Vec::new();
                use_relu = Vec::new();
                input = Matrix { rows: Vec::new() };
                output = Matrix { rows: Vec::new() };

                continue;
            }

            let key_value: Vec<&str> = line.split(":").collect();

            let key = *key_value.first().expect("No key on this line!");
            let key_pieces: Vec<&str> = key.split(" ").collect();
            let key_name = *key_pieces.first().expect("This key has no name!");

            let value = *key_value.get(1).expect("No value on this line!");

            match key_name {
                "Weights" => weights.push(create_matrix_from_string(value)),

                "Biases" => biases.push(create_matrix_from_string(value)),

                "Relu" => match value.trim() {
                    "true" => use_relu.push(true),
                    "false" => use_relu.push(false),
                    _ => panic!("Relu has strange value"),
                },

                "Example_Input" => input = create_matrix_from_string(value),

                "Example_Output" => output = create_matrix_from_string(value),

                _ => {}
            }
        }
    }

    let neural_network = NeuralNetwork::new(use_relu.clone(), weights.clone(), biases.clone())
        .expect("Unable to create proper neural network!");
    let calculated_output = neural_network
        .compute_output(input)
        .expect("Unable to compute output!");

    assert_eq!(output, calculated_output);
}

fn create_matrix_from_string(string: &str) -> Matrix<f64> {
    let no_opening_brackets = string.replace("[", "");
    let no_closing_brackets = no_opening_brackets.replace("]]", "");
    let row_strings = no_closing_brackets.split("],");
    let rows: Vec<Vec<f64>> = row_strings
        .map(|row_string| {
            row_string
                .split(",")
                .map(|s| s.trim().parse().expect("Given matrix has non-float value!"))
                .collect()
        })
        .collect();

    Matrix::new(rows).expect("Matrix was not created properly")
}
