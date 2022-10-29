use std::ops;

#[derive(Debug)]
pub enum MatrixError {
    InvalidArgument,
    BadInternalState
}

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: Vec<Vec<T>>
}

impl<T> Matrix<T> {
    pub fn new(rows: Vec<Vec<T>>) -> Result<Self, MatrixError> {
        let row_length = rows.first().ok_or(MatrixError::InvalidArgument)?.len();

        for row in &rows {
            if row.len() != row_length {
                return Err(MatrixError::InvalidArgument);
            }
        }

        Ok(Matrix { rows })
    }

    pub fn row_num(&self) -> usize {
        self.rows.len()
    }

    pub fn col_num(&self) -> usize {
        self.rows.first().unwrap().len()
    }
}

impl<T> Matrix<T> 
where T: Clone + ToOwned<Owned = T> {
    pub fn row(&self, i: usize) -> Option<Vec<T>> {
        match self.rows.get(i) {
            Some(row_slice) => Some(row_slice.to_owned()),
            None => None
        }
    }

    pub fn col(&self, j: usize) -> Option<Vec<T>> {
        let mut col: Vec<T> = Vec::new();

        for row in &self.rows {
            if let Some(val) = row.get(j) {
                col.push(val.to_owned());
            } else {
                return None
            }
        }

        Some(col)
    }

    pub fn val(&self, i: usize, j: usize) -> Option<T> {
        if let Some(row) = self.rows.get(i) {
            if let Some(col) = row.get(j) {
                return Some(col.to_owned())
            }
        }

        None
    }
}

impl<T> Matrix<T>
where T: ops::Add<Output = T> + ops::Mul<Output = T> + ToOwned<Owned = T> + Clone + Default {
    pub fn mul(&self, rhs: &Self) -> Result<Self, MatrixError> {
        if self.col_num() != rhs.row_num() {
            return Err(MatrixError::InvalidArgument);
        }

        let mut rows: Vec<Vec<T>> = Vec::new();

        for i in 0..self.row_num() {
            let mut new_row: Vec<T> = Vec::new();
            let row = self.row(i).expect("Matrix is missing a required row");

            for j in 0..rhs.col_num() {
                let col = rhs.col(j).expect("Matrix is missing a required column");
                let val = dot(&row, &col)?;
                new_row.push(val);
            }
            rows.push(new_row);
        }

        Ok(Matrix{ rows })
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, MatrixError> {
        if self.col_num() != rhs.col_num() || self.row_num() != rhs.row_num() {
            return Err(MatrixError::InvalidArgument)
        }

        let mut rows: Vec<Vec<T>> = Vec::new();

        for i in 0..self.row_num() {
            let mut row: Vec<T> = Vec::new();

            for j in 0..self.col_num() {
                let value_a = self.val(i, j).expect("Required value missing");
                let value_b = rhs.val(i, j).expect("Required value missing");

                row.push(value_a + value_b);
            }

            rows.push(row);
        }

        Ok(Matrix{ rows })
    }
}

impl<T> PartialEq for Matrix<T> where T: PartialEq{
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
    }
}

impl<T> Eq for Matrix<T> where T: Eq {}

pub fn dot<T>(lhs: &[T], rhs: &[T]) -> Result<T, MatrixError>
where T: ops::Add<Output = T> + ops::Mul<Output = T> + Default + ToOwned<Owned = T>, {
    let mut sum = T::default();

    for i in 0..lhs.len() {
        let a = lhs.get(i).ok_or(MatrixError::InvalidArgument)?;
        let b = rhs.get(i).ok_or(MatrixError::InvalidArgument)?;

        sum = sum + a.to_owned()*b.to_owned();
    }

    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::{Matrix, dot};

    fn matrix_a() -> Matrix<isize> {
        Matrix::new( vec![
            vec![1, 2, 3, 4],
            vec![4, 2, 2, 1],
            vec![1, 1, 1, 1]
        ]).unwrap()
    }

    fn matrix_b() -> Matrix<isize> {
        Matrix::new(vec![
            vec![1, 4, 1],
            vec![2, 3, 1],
            vec![3, 2, 1],
            vec![4, 1, 1]
        ]).unwrap()
    }

    fn matrix_c() -> Matrix<isize> {
        Matrix::new( vec![
            vec![1, 1, 3, 4],
            vec![2, 2, 4, 1],
            vec![1, 2, 1, 2]
        ]).unwrap()
    }

    fn matrix_product() -> Matrix<isize> {
        Matrix::new(vec![
            vec![30, 20, 10],
            vec![18, 27, 9],
            vec![10, 10, 4]
        ]).unwrap()
    }

    fn matrix_sum() -> Matrix<isize> {
        Matrix::new( vec![
            vec![2, 3, 6, 8],
            vec![6, 4, 6, 2],
            vec![2, 3, 2, 3]
        ]).unwrap()
    }

    #[test]
    fn proper_row_returned() {
        assert_eq!(matrix_a().row(0).unwrap(), vec![1, 2, 3, 4])
    }

    #[test]
    fn proper_row_num_returned() {
        assert_eq!(matrix_a().row_num(), 3)
    }

    #[test]
    fn proper_col_returned() {
        assert_eq!(matrix_a().col(0).unwrap(), vec![1, 4, 1])
    }

    #[test]
    fn proper_col_num_returned() {
        assert_eq!(matrix_a().col_num(), 4)
    }

    #[test]
    fn proper_val_returned() {
        assert_eq!(matrix_a().val(1, 2).unwrap(), 2)
    }

    #[test]
    fn proper_multiplication() {
        let a = matrix_a().mul(&matrix_b()).unwrap();
        let b = matrix_product();

        assert_eq!(a, b)
    }

    #[test] 
    fn proper_addition() {
        let a = matrix_a().add(&matrix_c()).unwrap();
        let b = matrix_sum();

        assert_eq!(a, b)
    }

    #[test]
    fn proper_dot() {
        let a = vec![1, 2, 3, 4];
        let b = vec![4, 3, 2, 1];
        let c = dot(&a, &b).unwrap();

        assert_eq!(c, 20)
    }
}