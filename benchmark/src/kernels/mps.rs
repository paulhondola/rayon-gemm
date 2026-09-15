//! Apple Silicon GPU GEMM kernel using `MetalPerformanceShaders` (`MPSMatrixMultiplication`).

use std::marker::PhantomData;
use std::time::{Duration, Instant};

use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResourceOptions,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use crate::Matrix;
use crate::kernels::{Element, GemmKernel, assert_gemm_dimensions};

/// Types natively supported by Apple's Metal Performance Shaders matrix multiplication.
///
/// Metal Performance Shaders supports half (`f16`) and single (`f32`) precision.
/// Double precision (`f64`) is not supported by Apple Silicon Metal GPUs.
pub trait MpsElement: Element {
    fn mps_data_type() -> MPSDataType;
}

impl MpsElement for f16 {
    fn mps_data_type() -> MPSDataType {
        MPSDataType::Float16
    }
}

impl MpsElement for f32 {
    fn mps_data_type() -> MPSDataType {
        MPSDataType::Float32
    }
}

/// A dense matrix multiplication kernel leveraging Apple's `MPSMatrixMultiplication`.
pub struct MpsGemm<T: MpsElement> {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    _marker: PhantomData<T>,
}

impl<T: MpsElement> MpsGemm<T> {
    /// Creates a new `MpsGemm` instance by acquiring the system default Metal device
    /// and a dedicated command queue.
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let command_queue = device.newCommandQueue()?;
        Some(Self {
            device,
            command_queue,
            _marker: PhantomData,
        })
    }

    /// Benchmarks matrix multiplication by pre-allocating shared buffers once,
    /// running an untimed warm-up iteration to bring the GPU clock up, and measuring
    /// `repetitions` timed dispatches.
    pub fn benchmark(
        &self,
        lhs: &Matrix<T>,
        rhs: &Matrix<T>,
        output: &mut Matrix<T>,
        repetitions: usize,
    ) -> Duration {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.rows();
        let count = n * n;
        let elem_size = std::mem::size_of::<T>();
        let byte_len = count * elem_size;
        let row_bytes = n * elem_size;

        autoreleasepool(|_| {
            let buf_a = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for LHS");
            let buf_b = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for RHS");
            let buf_c = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for Output");

            // Copy input matrices into unified shared memory
            unsafe {
                std::ptr::copy_nonoverlapping(
                    lhs.as_slice().as_ptr(),
                    buf_a.contents().as_ptr().cast(),
                    count,
                );
                std::ptr::copy_nonoverlapping(
                    rhs.as_slice().as_ptr(),
                    buf_b.contents().as_ptr().cast(),
                    count,
                );
            }

            let data_type = T::mps_data_type();
            let desc_a = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };
            let desc_b = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };
            let desc_c = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };

            let mat_a = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_a, &desc_a)
            };
            let mat_b = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_b, &desc_b)
            };
            let mat_c = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_c, &desc_c)
            };

            let mps_mul = unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.device,
                    false,
                    false,
                    n,
                    n,
                    n,
                    1.0,
                    0.0,
                )
            };

            // Warm-up dispatch
            {
                let cmd_buf = self
                    .command_queue
                    .commandBuffer()
                    .expect("failed to create Metal command buffer");
                unsafe {
                    mps_mul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                        &cmd_buf, &mat_a, &mat_b, &mat_c,
                    );
                }
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
            }

            // Timed repetitions
            let mut total = Duration::ZERO;
            for _ in 0..repetitions {
                let cmd_buf = self
                    .command_queue
                    .commandBuffer()
                    .expect("failed to create Metal command buffer");
                unsafe {
                    mps_mul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                        &cmd_buf, &mat_a, &mat_b, &mat_c,
                    );
                }
                let start = Instant::now();
                cmd_buf.commit();
                cmd_buf.waitUntilCompleted();
                total += start.elapsed();
            }

            // Copy result back to CPU output matrix
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf_c.contents().as_ptr().cast(),
                    output.as_mut_slice().as_mut_ptr(),
                    count,
                );
            }

            total.div_f64(repetitions as f64)
        })
    }
}

impl<T: MpsElement> GemmKernel<T> for MpsGemm<T> {
    fn name(&self) -> &'static str {
        "mps"
    }

    fn compute(&self, lhs: &Matrix<T>, rhs: &Matrix<T>, output: &mut Matrix<T>) {
        assert_gemm_dimensions(lhs, rhs, output);
        let n = lhs.rows();
        let count = n * n;
        let elem_size = std::mem::size_of::<T>();
        let byte_len = count * elem_size;
        let row_bytes = n * elem_size;

        autoreleasepool(|_| {
            let buf_a = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for LHS");
            let buf_b = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for RHS");
            let buf_c = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate Metal buffer for Output");

            unsafe {
                std::ptr::copy_nonoverlapping(
                    lhs.as_slice().as_ptr(),
                    buf_a.contents().as_ptr().cast(),
                    count,
                );
                std::ptr::copy_nonoverlapping(
                    rhs.as_slice().as_ptr(),
                    buf_b.contents().as_ptr().cast(),
                    count,
                );
            }

            let data_type = T::mps_data_type();
            let desc_a = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };
            let desc_b = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };
            let desc_c = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    n, n, row_bytes, data_type,
                )
            };

            let mat_a = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_a, &desc_a)
            };
            let mat_b = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_b, &desc_b)
            };
            let mat_c = unsafe {
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buf_c, &desc_c)
            };

            let mps_mul = unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.device,
                    false,
                    false,
                    n,
                    n,
                    n,
                    1.0,
                    0.0,
                )
            };

            let cmd_buf = self
                .command_queue
                .commandBuffer()
                .expect("failed to create Metal command buffer");
            unsafe {
                mps_mul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &cmd_buf, &mat_a, &mat_b, &mat_c,
                );
            }
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf_c.contents().as_ptr().cast(),
                    output.as_mut_slice().as_mut_ptr(),
                    count,
                );
            }
        });
    }
}
