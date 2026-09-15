//! Dense floating-point GEMM kernels sharing one overwrite-style interface.

pub(crate) mod common;
#[cfg(target_os = "macos")]
pub mod mps;
pub mod rayon;
pub mod serial;
pub mod static_threads;

use std::ops::{Add, AddAssign, Mul};

pub(crate) use common::{assert_gemm_dimensions, ikj_rows};
#[cfg(target_os = "macos")]
pub use mps::{MpsElement, MpsGemm};
pub use rayon::{RayonIkjGemm, RayonTiledGemm};
pub use serial::{IkjGemm, NaiveGemm, TiledGemm};
pub use static_threads::{StaticIkjGemm, StaticTiledGemm};

use crate::Matrix;

/// Helper trait dispatching MPS benchmarks for supported element types.
pub trait MpsBench: Sized {
    #[cfg(target_os = "macos")]
    fn run_mps(
        lhs: &Matrix<Self>,
        rhs: &Matrix<Self>,
        output: &mut Matrix<Self>,
        repetitions: usize,
    ) -> std::time::Duration;
}

#[cfg(target_os = "macos")]
impl MpsBench for f16 {
    fn run_mps(
        lhs: &Matrix<Self>,
        rhs: &Matrix<Self>,
        output: &mut Matrix<Self>,
        repetitions: usize,
    ) -> std::time::Duration {
        let kernel = MpsGemm::<f16>::new().expect("Failed to initialize Metal Performance Shaders");
        kernel.benchmark(lhs, rhs, output, repetitions)
    }
}

#[cfg(target_os = "macos")]
impl MpsBench for f32 {
    fn run_mps(
        lhs: &Matrix<Self>,
        rhs: &Matrix<Self>,
        output: &mut Matrix<Self>,
        repetitions: usize,
    ) -> std::time::Duration {
        let kernel = MpsGemm::<f32>::new().expect("Failed to initialize Metal Performance Shaders");
        kernel.benchmark(lhs, rhs, output, repetitions)
    }
}

#[cfg(target_os = "macos")]
impl MpsBench for f64 {
    fn run_mps(
        _lhs: &Matrix<Self>,
        _rhs: &Matrix<Self>,
        _output: &mut Matrix<Self>,
        _repetitions: usize,
    ) -> std::time::Duration {
        panic!("MPS GEMM does not support f64 precision; validation should have rejected this")
    }
}

#[cfg(not(target_os = "macos"))]
impl<T: Element> MpsBench for T {}

/// A floating-point element type the kernels can multiply.
///
/// `Default` supplies zero for clearing outputs and starting sums. `EPSILON`
/// and the `f64` conversions let callers build inputs and compare results
/// independently of precision.
pub trait Element:
    Copy
    + Default
    + Send
    + Sync
    + Add<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + MpsBench
    + 'static
{
    /// Machine epsilon of the element type, widened to `f64`.
    const EPSILON: f64;

    fn from_f64(value: f64) -> Self;

    fn to_f64(self) -> f64;
}

macro_rules! impl_element {
    ($($float:ty),*) => {
        $(
            impl Element for $float {
                const EPSILON: f64 = <$float>::EPSILON as f64;

                fn from_f64(value: f64) -> Self {
                    value as $float
                }

                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_element!(f16, f32, f64);

/// A dense matrix product kernel that computes `output = lhs * rhs`.
///
/// All inputs must have compatible dimensions. Implementations validate that
/// condition at their entry point, then use row slices in the compute loops so
/// LLVM can eliminate repeated index checks and autovectorize contiguous work.
pub trait GemmKernel<T: Element>: Send + Sync {
    fn name(&self) -> &'static str;

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>);
}

#[cfg(test)]
mod tests {
    use super::{
        Element, GemmKernel, IkjGemm, NaiveGemm, RayonIkjGemm, RayonTiledGemm, StaticIkjGemm,
        StaticTiledGemm, TiledGemm,
    };
    use crate::Matrix;

    fn inputs<T: Element>(n: usize) -> (Matrix<T>, Matrix<T>) {
        let lhs = Matrix::from_fn(n, n, |row, col| {
            T::from_f64(((row * 17 + col * 13) % 23) as f64 / 23.0)
        });
        let rhs = Matrix::from_fn(n, n, |row, col| {
            T::from_f64(((row * 7 + col * 19) % 29) as f64 / 29.0)
        });
        (lhs, rhs)
    }

    fn assert_close<T: Element>(actual: &Matrix<T>, expected: &Matrix<T>) {
        for (index, (&actual, &expected)) in actual
            .as_slice()
            .iter()
            .zip(expected.as_slice())
            .enumerate()
        {
            let (actual, expected) = (actual.to_f64(), expected.to_f64());
            // Kernels may add the same terms in a different order (SIMD lanes,
            // GPU fast-math). Measured drift is ~1.3 ε at n = 7 and ~5 ε at
            // n = 256 for every precision, so 8 ε relative leaves headroom.
            let tolerance = 8.0 * T::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: expected {expected}, got {actual} (tolerance {tolerance:e})"
            );
        }
    }

    fn every_kernel_matches_naive<T: Element>() {
        let n = 7;
        let (lhs, rhs) = inputs::<T>(n);
        let mut expected = Matrix::zeros(n, n);
        NaiveGemm.compute(&lhs, &rhs, &mut expected);

        let mut kernels: Vec<Box<dyn GemmKernel<T>>> = vec![
            Box::new(IkjGemm),
            Box::new(TiledGemm::new(3)),
            Box::new(RayonIkjGemm),
            Box::new(RayonTiledGemm::new(3)),
        ];
        // Every static thread count up to `n`, including uneven row splits.
        for threads in 1..=n {
            kernels.push(Box::new(
                StaticIkjGemm::new(threads).expect("static thread pool should build"),
            ));
            kernels.push(Box::new(
                StaticTiledGemm::new(threads, 3).expect("static tiled thread pool should build"),
            ));
        }

        for kernel in kernels {
            let mut actual = Matrix::zeros(n, n);
            kernel.compute(&lhs, &rhs, &mut actual);
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn every_kernel_matches_naive_on_a_non_tile_aligned_matrix() {
        every_kernel_matches_naive::<f16>();
        every_kernel_matches_naive::<f32>();
        every_kernel_matches_naive::<f64>();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mps_matches_naive_on_f16_and_f32() {
        let n = 7;
        {
            let (lhs, rhs) = inputs::<f16>(n);
            let mut expected = Matrix::zeros(n, n);
            NaiveGemm.compute(&lhs, &rhs, &mut expected);
            let mut actual = Matrix::zeros(n, n);
            let mps = super::MpsGemm::<f16>::new().expect("MPS should initialize");
            mps.compute(&lhs, &rhs, &mut actual);
            assert_close(&actual, &expected);
        }
        {
            let (lhs, rhs) = inputs::<f32>(n);
            let mut expected = Matrix::zeros(n, n);
            NaiveGemm.compute(&lhs, &rhs, &mut expected);
            let mut actual = Matrix::zeros(n, n);
            let mps = super::MpsGemm::<f32>::new().expect("MPS should initialize");
            mps.compute(&lhs, &rhs, &mut actual);
            assert_close(&actual, &expected);
        }
    }
}
