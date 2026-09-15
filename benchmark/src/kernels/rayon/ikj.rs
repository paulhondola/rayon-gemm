use rayon::prelude::*;

use crate::Matrix;
use crate::kernels::{Element, GemmKernel, assert_gemm_dimensions, ikj_rows};

/// Rayon work-stealing implementation of the contiguous `i-k-j` kernel.
///
/// It runs in the currently installed Rayon pool. The benchmark runner
/// installs a per-thread-count pool, avoiding global-pool configuration and
/// ensuring that pool construction is outside the timed region.
pub struct RayonIkjGemm;

impl<T: Element> GemmKernel<T> for RayonIkjGemm {
    fn name(&self) -> &'static str {
        "rayon-ikj"
    }

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>) {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.cols();
        output.as_mut_slice().fill(T::default());

        output
            .as_mut_slice()
            .par_chunks_exact_mut(n)
            .enumerate()
            .for_each(|(row, output_row)| {
                ikj_rows(lhs.as_slice(), rhs.as_slice(), output_row, row, n)
            });
    }
}
