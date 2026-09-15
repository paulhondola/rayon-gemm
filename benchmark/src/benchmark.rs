use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use rayon::ThreadPoolBuilder;
use rayon_gemm::{
    Element, GemmKernel, Matrix,
    kernels::{
        IkjGemm, NaiveGemm, RayonIkjGemm, RayonTiledGemm, StaticIkjGemm, StaticTiledGemm, TiledGemm,
    },
};
use serde::Serialize;

use crate::{
    cli::{BenchmarkPlan, KernelChoice, Precision},
    report::BenchmarkProgress,
};

/// One measured benchmark configuration, shared by terminal and file reporters.
#[derive(Debug, Serialize)]
pub(crate) struct BenchmarkRecord {
    pub(crate) kernel: String,
    pub(crate) n: usize,
    pub(crate) threads: usize,
    pub(crate) precision: &'static str,
    pub(crate) elapsed_ms: f64,
    pub(crate) gflops: f64,
}

pub(crate) fn run(
    plan: &BenchmarkPlan,
) -> Result<Vec<BenchmarkRecord>, rayon::ThreadPoolBuildError> {
    let mut records = Vec::new();
    let progress = BenchmarkProgress::new(plan.total_configurations(), plan.no_progress);

    // Each arm monomorphizes the whole sweep, so kernels compile to native
    // arithmetic for that precision with no per-element dispatch.
    for &precision in &plan.precisions {
        match precision {
            Precision::F16 => run_precision::<f16>(plan, precision, &progress, &mut records)?,
            Precision::F32 => run_precision::<f32>(plan, precision, &progress, &mut records)?,
            Precision::F64 => run_precision::<f64>(plan, precision, &progress, &mut records)?,
        }
    }

    progress.finish();
    Ok(records)
}

fn run_precision<T: Element>(
    plan: &BenchmarkPlan,
    precision: Precision,
    progress: &BenchmarkProgress,
    records: &mut Vec<BenchmarkRecord>,
) -> Result<(), rayon::ThreadPoolBuildError> {
    for &n in &plan.sizes {
        let (lhs, rhs) = benchmark_inputs::<T>(n);
        let mut output = Matrix::zeros(n, n);

        for kernel in plan.kernels.iter().copied() {
            let thread_counts: &[usize] = if kernel.uses_workers() {
                &plan.threads
            } else {
                &[1]
            };
            for &thread_count in thread_counts {
                progress.set_target(kernel.label(), n, precision.label(), thread_count);
                let elapsed = measure(
                    kernel,
                    thread_count,
                    plan.block_size,
                    plan.repetitions,
                    &lhs,
                    &rhs,
                    &mut output,
                )?;
                progress.step();
                let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
                let gflops = 2.0 * (n as f64).powi(3) / elapsed.as_secs_f64() / 1e9;
                records.push(BenchmarkRecord {
                    kernel: kernel.label().to_owned(),
                    n,
                    threads: thread_count,
                    precision: precision.label(),
                    elapsed_ms,
                    gflops,
                });
            }
        }
    }

    Ok(())
}

fn benchmark_inputs<T: Element>(n: usize) -> (Matrix<T>, Matrix<T>) {
    let lhs = Matrix::from_fn(n, n, |row, col| {
        T::from_f64(((row * 17 + col * 13) % 23) as f64 / 23.0)
    });
    let rhs = Matrix::from_fn(n, n, |row, col| {
        T::from_f64(((row * 7 + col * 19) % 29) as f64 / 29.0)
    });
    (lhs, rhs)
}

/// Returns the mean of `repetitions` timed runs.
///
/// Every arm first runs the kernel once untimed, after any pool is built, so
/// one-time costs (the process's first Rayon call, a fresh pool's idle
/// workers) stay out of the measured mean.
fn measure<T: Element>(
    choice: KernelChoice,
    threads: usize,
    block_size: usize,
    repetitions: usize,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    output: &mut Matrix<T>,
) -> Result<Duration, rayon::ThreadPoolBuildError> {
    let mut total = Duration::ZERO;
    match choice {
        KernelChoice::Naive => {
            let kernel = NaiveGemm;
            kernel.compute(lhs, rhs, output);
            for _ in 0..repetitions {
                total += time_kernel(&kernel, lhs, rhs, output);
            }
        }
        KernelChoice::Ikj => {
            let kernel = IkjGemm;
            kernel.compute(lhs, rhs, output);
            for _ in 0..repetitions {
                total += time_kernel(&kernel, lhs, rhs, output);
            }
        }
        KernelChoice::Tiled => {
            let kernel = TiledGemm::new(block_size);
            kernel.compute(lhs, rhs, output);
            for _ in 0..repetitions {
                total += time_kernel(&kernel, lhs, rhs, output);
            }
        }
        KernelChoice::RayonIkj => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let kernel = RayonIkjGemm;
            pool.install(|| kernel.compute(lhs, rhs, output));
            for _ in 0..repetitions {
                let start = Instant::now();
                pool.install(|| kernel.compute(black_box(lhs), black_box(rhs), black_box(output)));
                black_box(output.as_slice());
                total += start.elapsed();
            }
        }
        KernelChoice::RayonTiled => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            let kernel = RayonTiledGemm::new(block_size);
            pool.install(|| kernel.compute(lhs, rhs, output));
            for _ in 0..repetitions {
                let start = Instant::now();
                pool.install(|| kernel.compute(black_box(lhs), black_box(rhs), black_box(output)));
                black_box(output.as_slice());
                total += start.elapsed();
            }
        }
        KernelChoice::StaticIkj => {
            let kernel = StaticIkjGemm::new(threads)?;
            kernel.compute(lhs, rhs, output);
            for _ in 0..repetitions {
                total += time_kernel(&kernel, lhs, rhs, output);
            }
        }
        KernelChoice::StaticTiled => {
            let kernel = StaticTiledGemm::new(threads, block_size)?;
            kernel.compute(lhs, rhs, output);
            for _ in 0..repetitions {
                total += time_kernel(&kernel, lhs, rhs, output);
            }
        }
        #[cfg(target_os = "macos")]
        KernelChoice::Mps => {
            return Ok(T::run_mps(lhs, rhs, output, repetitions));
        }
    }

    Ok(total.div_f64(repetitions as f64))
}

fn time_kernel<T: Element>(
    kernel: &impl GemmKernel<T>,
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
    output: &mut Matrix<T>,
) -> Duration {
    let start = Instant::now();
    kernel.compute(black_box(lhs), black_box(rhs), black_box(output));
    black_box(output.as_slice());
    start.elapsed()
}
