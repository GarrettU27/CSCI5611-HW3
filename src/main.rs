use CSCI5611_HW3::matrix::Matrix;


fn main() {
    let a = Matrix::new( vec![
            vec![1, 2, 3, 4],
            vec![4, 2, 2, 1],
            vec![1, 1, 1, 1]
        ]
    ).unwrap();

    let b = Matrix::new(vec![
            vec![1, 4, 1],
            vec![2, 3, 1],
            vec![3, 2, 1],
            vec![4, 1, 1]
        ]
    ).unwrap();

    let c = a.mul(&b).unwrap();

    print!("{:?}", c);

    for val in &c {
        println!("c value: {}", val);
    }

    println!("Hello, world!");
}
