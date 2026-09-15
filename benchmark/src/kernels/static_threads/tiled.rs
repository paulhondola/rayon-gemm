use std::sync::Mutex;

use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

use crate::Matrix;
use crate::kernels::{Element, GemmKernel, assert_gemm_dimensions};

use super::static_row_counts;

/// Fixed row chunks on a persistent thread pool executing 2D cache-blocked GEMM.
pub struct StaticTiledGemm {
    pool: ThreadPool,
    block_size: usize,
}

impl StaticTiledGemm {
    pub fn new(threads: usize, block_size: usize) -> Result<Self, ThreadPoolBuildError> {
        assert!(threads > 0, "thread count must be greater than zero");
        assert!(block_size > 0, "block size must be greater than zero");
        let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
        Ok(Self { pool, block_size })
    }
}

impl<T: Element> GemmKernel<T> for StaticTiledGemm {
    fn name(&self) -> &'static str {
        "static-tiled"
    }

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>) {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.cols();
        let threads = self.pool.current_num_threads();
        assert!(
            threads <= n,
            "static-tiled needs at least one row per thread: {threads} threads for {n} rows"
        );
        output.as_mut_slice().fill(T::default());

        let mut remaining = output.as_mut_slice();
        let mut first_row = 0;
        let chunks: Vec<Mutex<(usize, &mut [T])>> = static_row_counts(n, threads)
            .map(|rows| {
                let (chunk, rest) = std::mem::take(&mut remaining).split_at_mut(rows * n);
                remaining = rest;
                let chunk_first_row = first_row;
                first_row += rows;
                Mutex::new((chunk_first_row, chunk))
            })
            .collect();

        let block_size = self.block_size;
        let (lhs_data, rhs_data) = (lhs.as_slice(), rhs.as_slice());
        self.pool.broadcast(|context| {
            let mut chunk = chunks[context.index()]
                .lock()
                .expect("each chunk is locked by exactly one worker");
            let (first_row, output_rows) = &mut *chunk;
            let local_row_count = output_rows.len() / n;

            for ii in (0..local_row_count).step_by(block_size) {
                let i_end = (ii + block_size).min(local_row_count);
                for kk in (0..n).step_by(block_size) {
                    let k_end = (kk + block_size).min(n);
                    for jj in (0..n).step_by(block_size) {
                        let j_end = (jj + block_size).min(n);
                        for local_row in ii..i_end {
                            let global_row = *first_row + local_row;
                            let lhs_row = &lhs_data[global_row * n..(global_row + 1) * n];
                            let output_tile =
                                &mut output_rows[local_row * n + jj..local_row * n + j_end];

                            for (&a_ik, rhs_row) in lhs_row[kk..k_end]
                                .iter()
                                .zip(rhs_data[kk * n..k_end * n].chunks_exact(n))
                            {
                                for (out, &b_kj) in output_tile.iter_mut().zip(&rhs_row[jj..j_end])
                                {
                                    *out += a_ik * b_kj;
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}
