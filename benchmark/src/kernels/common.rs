use super::Element;
use crate::Matrix;

pub(crate) fn assert_gemm_dimensions<T>(lhs: &Matrix<T>, rhs: &Matrix<T>, output: &Matrix<T>) {
    assert!(
        lhs.is_square() && rhs.is_square() && output.is_square(),
        "this benchmark supports only square matrices"
    );
    assert!(lhs.rows() > 0, "matrix dimension must be greater than zero");
    assert_eq!(lhs.cols(), rhs.rows(), "incompatible GEMM input dimensions");
    assert_eq!(output.rows(), lhs.rows(), "output row count is incorrect");
    assert_eq!(
        output.cols(),
        rhs.cols(),
        "output column count is incorrect"
    );
}

/// Multiplies contiguous rows using the `i-k-j` order.
///
/// `output_rows` holds whole rows beginning at `first_row`; this form is
/// shared by the Rayon and static-schedule kernels, whose row slices are known
/// to be disjoint by construction.
pub(crate) fn ikj_rows<T: Element>(
    lhs: &[T],
    rhs: &[T],
    output_rows: &mut [T],
    first_row: usize,
    n: usize,
) {
    debug_assert_eq!(lhs.len(), n * n);
    debug_assert_eq!(rhs.len(), n * n);
    debug_assert_eq!(output_rows.len() % n, 0);

    for (local_row, output_row) in output_rows.chunks_exact_mut(n).enumerate() {
        let lhs_row = &lhs[(first_row + local_row) * n..(first_row + local_row + 1) * n];

        for (&a_ik, rhs_row) in lhs_row.iter().zip(rhs.chunks_exact(n)) {
            for (out, &b_kj) in output_row.iter_mut().zip(rhs_row) {
                *out += a_ik * b_kj;
            }
        }
    }
}
