//! Building blocks for benchmarking dense, row-major floating-point matrix
//! products at `f16`, `f32`, or `f64` precision.
//!
//! Kernels deliberately overwrite their output matrix: after
//! `kernel.compute(&a, &b, &mut c)`, `c == a * b`.

#![feature(f16)]

pub mod kernels;
pub mod matrix;

pub use kernels::{Element, GemmKernel};
pub use matrix::Matrix;
