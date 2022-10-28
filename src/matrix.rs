use std::ops;

#[derive(Debug)]
pub struct InvalidArgumentError;

#[derive(Debug)]
pub struct Matrix<T> {
    rows: Vec<Vec<T>>
}

impl<T> Matrix<T> {
    pub fn new(rows: Vec<Vec<T>>) -> Result<Self, InvalidArgumentError> {
        let row_length = rows.first().ok_or(InvalidArgumentError)?.len();

        for row in &rows {
            if row.len() != row_length {
                return Err(InvalidArgumentError);
            }
        }

        Ok(Matrix { rows })
    }

    pub fn row_num(&self) -> usize {
        self.rows.len()
    }

    pub fn val(&self, i: usize, j: usize) -> Option<&T> {
        if let Some(row) = self.rows.get(i) {
            return row.get(j)
        }

        None
    }

    pub fn col_num(&self) -> usize {
        self.rows.first().unwrap().len()
    }
}

impl<T> Matrix<T> 
where T: Clone + ToOwned<Owned = T> {
    pub fn row(&self, i: usize) -> Vec<T> {
        let row_slice = self.rows.get(i).unwrap();
        row_slice.to_owned()
    }

    pub fn col(&self, j: usize) -> Vec<T> {
        let mut col: Vec<T> = Vec::new();

        for row in &self.rows {
            let val = row.get(j).unwrap();
            col.push(val.to_owned());
        }

        col
    }
}

impl<T> Matrix<T>
where T: ops::Add<Output = T> + ops::Mul<Output = T> + ToOwned<Owned = T> + Clone + Default {
    pub fn mul(&self, rhs: &Self) -> Result<Self, InvalidArgumentError> {
        if self.col_num() != rhs.row_num() {
            return Err(InvalidArgumentError);
        }

        let mut rows: Vec<Vec<T>> = Vec::new();

        for i in 0..self.row_num() {
            let mut new_row: Vec<T> = Vec::new();
            let row = self.row(i);

            for j in 0..rhs.col_num() {
                let col = rhs.col(j);
                let val = dot(&row, &col)?;
                new_row.push(val);
            }
            rows.push(new_row);
        }

        Ok(Matrix{ rows })
    }
}

impl <'a, T> Matrix<T> {
    pub fn iter(&'a self) -> MatrixIntoIterator<'a, T> {
        MatrixIntoIterator {
            matrix: self,
            index: 0
        }
    }
}

impl<'a, T> IntoIterator for &'a Matrix<T> {
    type Item = &'a T;
    type IntoIter = MatrixIntoIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct MatrixIntoIterator<'a, T> {
    matrix: &'a Matrix<T>,
    index: usize
}

impl<'a, T> Iterator for MatrixIntoIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let col_num = self.matrix.col_num();
        let val = self.matrix.val(self.index / col_num, self.index % col_num);
        self.index += 1;

        val
    }
}

pub fn dot<T>(lhs: &[T], rhs: &[T]) -> Result<T, InvalidArgumentError>
where T: ops::Add<Output = T> + ops::Mul<Output = T> + Default + ToOwned<Owned = T>, {
    let mut sum = T::default();

    for i in 0..lhs.len() {
        let a = lhs.get(i).ok_or(InvalidArgumentError)?;
        let b = rhs.get(i).ok_or(InvalidArgumentError)?;

        sum = sum + a.to_owned()*b.to_owned();
    }

    Ok(sum)
}