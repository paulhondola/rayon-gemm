use crate::Matrix;
use crate::kernels::{Element, GemmKernel, assert_gemm_dimensions};

/// Sequential cache-blocked GEMM. A block size of 64 is a practical default,
/// while the CLI permits empirical tuning per target CPU and precision.
pub struct TiledGemm {
    block_size: usize,
}

impl TiledGemm {
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block size must be greater than zero");
        Self { block_size }
    }
}

impl<T: Element> GemmKernel<T> for TiledGemm {
    fn name(&self) -> &'static str {
        "tiled"
    }

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>) {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.cols();
        output.as_mut_slice().fill(T::default());

        for ii in (0..n).step_by(self.block_size) {
            let i_end = (ii + self.block_size).min(n);
            for kk in (0..n).step_by(self.block_size) {
                let k_end = (kk + self.block_size).min(n);
                for jj in (0..n).step_by(self.block_size) {
                    let j_end = (jj + self.block_size).min(n);
                    for row in ii..i_end {
                        let lhs_row = lhs.row(row);
                        let output_tile = &mut output.row_mut(row)[jj..j_end];

                        for (&a_ik, rhs_row) in lhs_row[kk..k_end]
                            .iter()
                            .zip(rhs.as_slice()[kk * n..k_end * n].chunks_exact(n))
                        {
                            for (out, &b_kj) in output_tile.iter_mut().zip(&rhs_row[jj..j_end]) {
                                *out += a_ik * b_kj;
                            }
                        }
                    }
                }
            }
        }
    }
}
