use std::fmt;

/// A contiguous, row-major matrix.
///
/// Dimensions are kept even though the benchmark currently operates on square
/// matrices, so dimension checks remain explicit at kernel boundaries.
#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Box<[T]>,
}

impl<T> Matrix<T> {
    /// Constructs a matrix from row-major storage.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        assert_eq!(
            data.len(),
            len,
            "matrix data length does not match its dimensions"
        );

        Self {
            rows,
            cols,
            data: data.into_boxed_slice(),
        }
    }

    /// Creates a matrix by evaluating `f` once for every element.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        let mut data = Vec::with_capacity(len);

        for row in 0..rows {
            for col in 0..cols {
                data.push(f(row, col));
            }
        }

        Self::from_vec(rows, cols, data)
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[must_use]
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "row index is out of range");
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows, "row index is out of range");
        let start = row * self.cols;
        &mut self.data[start..start + self.cols]
    }

    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> &T {
        assert!(
            row < self.rows && col < self.cols,
            "matrix index is out of range"
        );
        &self.data[row * self.cols + col]
    }
}

impl<T: Default + Clone> Matrix<T> {
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("matrix dimensions overflow usize");
        Self::from_vec(rows, cols, vec![T::default(); len])
    }
}

impl<T: fmt::Debug> fmt::Debug for Matrix<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Matrix")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("data", &self.data)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn storage_is_row_major() {
        let matrix = Matrix::from_fn(2, 3, |row, col| row * 10 + col);
        assert_eq!(matrix.as_slice(), &[0, 1, 2, 10, 11, 12]);
        assert_eq!(matrix.row(1), &[10, 11, 12]);
    }
}
