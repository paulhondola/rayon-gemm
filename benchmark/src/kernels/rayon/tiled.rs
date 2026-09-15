use rayon::prelude::*;

use crate::Matrix;
use crate::kernels::{Element, GemmKernel, assert_gemm_dimensions};

/// Rayon work-stealing implementation, partitioned into whole blocks of rows.
pub struct RayonTiledGemm {
    block_size: usize,
}

impl RayonTiledGemm {
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block size must be greater than zero");
        Self { block_size }
    }
}

impl<T: Element> GemmKernel<T> for RayonTiledGemm {
    fn name(&self) -> &'static str {
        "rayon-tiled"
    }

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>) {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.cols();
        let block_size = self.block_size;
        output.as_mut_slice().fill(T::default());

        // Each task receives an integral group of output rows. `par_chunks`
        // proves those mutable groups are disjoint without pointer arithmetic.
        output
            .as_mut_slice()
            .par_chunks_mut(block_size * n)
            .enumerate()
            .for_each(|(tile_index, output_rows)| {
                let ii = tile_index * block_size;

                for kk in (0..n).step_by(block_size) {
                    let k_end = (kk + block_size).min(n);
                    for jj in (0..n).step_by(block_size) {
                        let j_end = (jj + block_size).min(n);
                        for (local_row, output_row) in output_rows.chunks_exact_mut(n).enumerate() {
                            let lhs_row =
                                &lhs.as_slice()[(ii + local_row) * n..(ii + local_row + 1) * n];
                            let output_tile = &mut output_row[jj..j_end];

                            for (&a_ik, rhs_row) in lhs_row[kk..k_end]
                                .iter()
                                .zip(rhs.as_slice()[kk * n..k_end * n].chunks_exact(n))
                            {
                                for (out, &b_kj) in output_tile.iter_mut().zip(&rhs_row[jj..j_end])
                                {
                                    *out += a_ik * b_kj;
                                }
                            }
                        }
                    }
                }
            });
    }
}
