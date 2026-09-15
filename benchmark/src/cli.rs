use std::{
    ffi::OsStr,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};

const DEFAULT_SIZES: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

#[derive(Debug, Parser)]
#[command(about = "Benchmark safe, row-major floating-point GEMM kernels")]
pub(crate) struct Cli {
    /// Matrix dimensions, as a comma-delimited list.
    #[arg(long, value_delimiter = ',')]
    sizes: Vec<usize>,

    /// Worker counts, as a comma-delimited list. Defaults to powers of two up to available CPUs.
    #[arg(long, value_delimiter = ',')]
    threads: Vec<usize>,

    /// Kernel(s) to run. Omit to run every kernel.
    #[arg(long, value_delimiter = ',', value_enum)]
    kernel: Vec<KernelChoice>,

    /// Element precision(s), as a comma-delimited list. Defaults to f32.
    #[arg(long, value_delimiter = ',', value_enum)]
    precision: Vec<Precision>,

    /// Number of measured runs per configuration, after one untimed warm-up
    /// run; records contain their mean.
    #[arg(long, default_value_t = 1)]
    repetitions: usize,

    /// Tile edge length for the blocked kernels.
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// Destination path prefix without extension (e.g. 'data/f16'). Both .csv
    /// and .json files will be created; missing parent directories are created.
    #[arg(long)]
    output: PathBuf,

    /// Disable the interactive progress bar.
    #[arg(long)]
    no_progress: bool,
}

/// Fully resolved configuration used by the benchmark runner.
#[derive(Debug)]
pub(crate) struct BenchmarkPlan {
    pub(crate) sizes: Vec<usize>,
    pub(crate) threads: Vec<usize>,
    pub(crate) kernels: Vec<KernelChoice>,
    pub(crate) precisions: Vec<Precision>,
    pub(crate) repetitions: usize,
    pub(crate) block_size: usize,
    pub(crate) csv_output: File,
    pub(crate) json_output: File,
    pub(crate) no_progress: bool,
}

impl BenchmarkPlan {
    /// Returns the exact number of configurations that will be measured.
    pub(crate) fn total_configurations(&self) -> usize {
        let configs_per_matrix: usize = self
            .kernels
            .iter()
            .map(|kernel| {
                if kernel.uses_workers() {
                    self.threads.len()
                } else {
                    1
                }
            })
            .sum();

        self.precisions.len() * self.sizes.len() * configs_per_matrix
    }
}

impl Cli {
    pub(crate) fn into_plan(self) -> Result<BenchmarkPlan, String> {
        validate_cli(&self)?;

        let sizes = if self.sizes.is_empty() {
            DEFAULT_SIZES.to_vec()
        } else {
            self.sizes
        };
        let threads = if self.threads.is_empty() {
            default_thread_counts()
        } else {
            self.threads
        };
        let precisions = if self.precision.is_empty() {
            vec![Precision::F32]
        } else {
            self.precision
        };
        let kernels = if self.kernel.is_empty() {
            #[allow(unused_mut)]
            let mut list = vec![
                KernelChoice::Naive,
                KernelChoice::Ikj,
                KernelChoice::Tiled,
                KernelChoice::RayonIkj,
                KernelChoice::RayonTiled,
                KernelChoice::StaticIkj,
                KernelChoice::StaticTiled,
            ];
            #[cfg(target_os = "macos")]
            if !precisions.contains(&Precision::F64) {
                list.push(KernelChoice::Mps);
            }
            list
        } else {
            self.kernel
        };

        // Validate the resolved sweep before touching the filesystem, so a
        // rejected plan never creates directories or an output file.
        validate_static_threads(&kernels, &threads, &sizes)?;
        validate_mps_precision(&kernels, &precisions)?;
        let (csv_output, json_output) = open_outputs(&self.output)?;

        Ok(BenchmarkPlan {
            sizes,
            threads,
            kernels,
            precisions,
            repetitions: self.repetitions,
            block_size: self.block_size,
            csv_output,
            json_output,
            no_progress: self.no_progress,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum KernelChoice {
    Naive,
    Ikj,
    Tiled,
    RayonIkj,
    RayonTiled,
    StaticIkj,
    StaticTiled,
    #[cfg(target_os = "macos")]
    #[value(name = "mps")]
    Mps,
}

impl KernelChoice {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Naive => "naive-ijk",
            Self::Ikj => "ikj",
            Self::Tiled => "tiled",
            Self::RayonIkj => "rayon-ikj",
            Self::RayonTiled => "rayon-tiled",
            Self::StaticIkj => "static-ikj",
            Self::StaticTiled => "static-tiled",
            #[cfg(target_os = "macos")]
            Self::Mps => "mps",
        }
    }

    pub(crate) fn uses_workers(self) -> bool {
        matches!(
            self,
            Self::RayonIkj | Self::RayonTiled | Self::StaticIkj | Self::StaticTiled
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Precision {
    F16,
    F32,
    F64,
}

impl Precision {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

fn validate_cli(cli: &Cli) -> Result<(), String> {
    if cli.repetitions == 0 {
        return Err("--repetitions must be greater than zero".into());
    }
    if cli.block_size == 0 {
        return Err("--block-size must be greater than zero".into());
    }
    if cli.sizes.contains(&0) {
        return Err("all --sizes values must be greater than zero".into());
    }
    if cli.threads.contains(&0) {
        return Err("all --threads values must be greater than zero".into());
    }
    Ok(())
}

/// `static-ikj` and `static-tiled` give every worker at least one row, so a
/// worker count above the smallest matrix dimension cannot be honored and is
/// rejected up front.
fn validate_static_threads(
    kernels: &[KernelChoice],
    threads: &[usize],
    sizes: &[usize],
) -> Result<(), String> {
    if !kernels.contains(&KernelChoice::StaticIkj) && !kernels.contains(&KernelChoice::StaticTiled)
    {
        return Ok(());
    }
    let max_threads = threads.iter().copied().max().unwrap_or(1);
    let min_size = sizes.iter().copied().min().unwrap_or(usize::MAX);
    if max_threads > min_size {
        return Err(format!(
            "static kernels need at least one row per thread; --threads {max_threads} exceeds --sizes {min_size}"
        ));
    }
    Ok(())
}

fn validate_mps_precision(
    kernels: &[KernelChoice],
    precisions: &[Precision],
) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    if kernels.contains(&KernelChoice::Mps) && precisions.contains(&Precision::F64) {
        return Err(
            "MPS GEMM only supports f16 and f32 precisions; f64 is not supported by Metal Performance Shaders".into(),
        );
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (kernels, precisions);
    Ok(())
}

fn validate_and_resolve_output_paths(path: &Path) -> Result<(PathBuf, PathBuf), String> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err("--output must include a destination prefix (e.g. 'data/f16')".into());
    }
    if path.is_dir() {
        return Err(format!(
            "output path '{}' is an existing directory; --output must specify a destination prefix without extension (e.g. '{}/results')",
            path.display(),
            path.display()
        ));
    }
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        return Err(format!(
            "--output must not include a file extension (got '.{ext}'); specify the path prefix (e.g. '{}') to generate both .csv and .json files",
            path.with_extension("").display()
        ));
    }

    let csv_path = path.with_extension("csv");
    let json_path = path.with_extension("json");
    Ok((csv_path, json_path))
}

/// Creates missing parent directories and opens both output files (.csv and .json)
/// before any benchmark runs, so an unwritable path fails immediately instead of
/// after the sweep. Neither file is truncated here: existing result files keep
/// their contents until new records are written.
fn open_outputs(path: &Path) -> Result<(File, File), String> {
    let (csv_path, json_path) = validate_and_resolve_output_paths(path)?;

    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create output directory '{}': {error}",
                parent.display()
            )
        })?;
    }

    let csv_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&csv_path)
        .map_err(|error| format!("cannot open output file '{}': {error}", csv_path.display()))?;

    let json_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&json_path)
        .map_err(|error| format!("cannot open output file '{}': {error}", json_path.display()))?;

    Ok((csv_file, json_file))
}

fn default_thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let mut counts = Vec::new();
    let mut current = 1;
    while current < max {
        counts.push(current);
        current *= 2;
    }
    counts.push(max);
    counts
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsStr, fs, path::PathBuf};

    use clap::Parser;

    use super::{Cli, KernelChoice, Precision, open_outputs, validate_and_resolve_output_paths};

    /// A per-process path under the system temp directory, so tests never
    /// write into the repository and parallel test runs do not collide.
    fn temp_output(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("rayon-gemm-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn empty_sweeps_expand_to_defaults() {
        let output = temp_output("defaults");
        let plan = Cli {
            sizes: Vec::new(),
            threads: Vec::new(),
            kernel: Vec::new(),
            precision: Vec::new(),
            repetitions: 1,
            block_size: 64,
            output: output.clone(),
            no_progress: false,
        }
        .into_plan()
        .expect("default plan should be valid");

        assert_eq!(plan.sizes, [64, 128, 256, 512, 1024, 2048, 4096]);
        assert!(plan.threads.contains(&1));
        assert_eq!(plan.kernels.first(), Some(&KernelChoice::Naive));
        #[cfg(target_os = "macos")]
        assert_eq!(plan.kernels.last(), Some(&KernelChoice::Mps));
        #[cfg(not(target_os = "macos"))]
        assert_eq!(plan.kernels.last(), Some(&KernelChoice::StaticTiled));
        assert_eq!(plan.precisions, [Precision::F32]);
        assert!(!plan.no_progress);
        assert!(plan.total_configurations() > 0);
        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[test]
    fn precision_flag_accepts_a_comma_delimited_sweep() {
        let output = temp_output("precision");
        let plan = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--precision"),
            OsStr::new("f16,f64"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("precision list should parse")
        .into_plan()
        .expect("plan should be valid");

        assert_eq!(plan.precisions, [Precision::F16, Precision::F64]);
        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[test]
    fn static_threads_above_the_matrix_dimension_are_rejected_before_running() {
        let output = temp_output("static-threads");
        let error = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--sizes"),
            OsStr::new("8,64"),
            OsStr::new("--threads"),
            OsStr::new("4,16"),
            OsStr::new("--kernel"),
            OsStr::new("static-ikj"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("arguments should parse")
        .into_plan()
        .expect_err("more static threads than rows must be rejected");

        assert!(error.contains("--threads 16 exceeds --sizes 8"));
        assert!(
            !output.with_extension("csv").exists(),
            "a rejected plan must not create the output file"
        );
        assert!(
            !output.with_extension("json").exists(),
            "a rejected plan must not create the output file"
        );
    }

    #[test]
    fn output_with_extension_is_rejected() {
        let error = validate_and_resolve_output_paths(PathBuf::from("results.csv").as_path())
            .expect_err("extension should be rejected");
        assert!(error.contains("must not include a file extension"));

        let error = validate_and_resolve_output_paths(PathBuf::from("data/f16.json").as_path())
            .expect_err("extension should be rejected");
        assert!(error.contains("must not include a file extension"));
    }

    #[test]
    fn output_as_existing_directory_is_rejected() {
        let dir = temp_output("existing_dir");
        fs::create_dir_all(&dir).expect("create test dir");

        let error = validate_and_resolve_output_paths(&dir)
            .expect_err("existing directory must be rejected");
        assert!(error.contains("is an existing directory"));

        fs::remove_dir_all(dir).expect("remove test dir");
    }

    #[test]
    fn output_creates_both_csv_and_json_files() {
        let output = temp_output("both_files");
        let (csv_file, json_file) = open_outputs(&output).expect("both files should open");
        drop(csv_file);
        drop(json_file);

        assert!(output.with_extension("csv").is_file());
        assert!(output.with_extension("json").is_file());

        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[test]
    fn missing_output_directories_are_created_before_running() {
        let root = temp_output("nested");
        let output = root.join("a/b/results");

        let (csv_file, json_file) =
            open_outputs(&output).expect("missing parent directories should be created");
        drop(csv_file);
        drop(json_file);

        assert!(output.with_extension("csv").is_file());
        assert!(output.with_extension("json").is_file());
        fs::remove_dir_all(root).expect("remove test directories");
    }

    #[test]
    fn unusable_output_paths_are_rejected_before_running() {
        let blocker = temp_output("blocker");
        fs::write(&blocker, b"").expect("create a regular file");

        let error = open_outputs(&blocker.join("results"))
            .expect_err("a regular file cannot be a parent directory");

        assert!(error.contains("output directory"));
        fs::remove_file(blocker).expect("remove test file");
    }

    #[test]
    fn existing_output_is_not_truncated_until_records_are_written() {
        let output = temp_output("existing");
        let csv_path = output.with_extension("csv");
        let json_path = output.with_extension("json");
        fs::write(&csv_path, b"previous csv").expect("seed existing csv");
        fs::write(&json_path, b"previous json").expect("seed existing json");

        let (csv_file, json_file) = open_outputs(&output).expect("existing output should open");
        drop(csv_file);
        drop(json_file);

        assert_eq!(
            fs::read(&csv_path).expect("read existing output"),
            b"previous csv"
        );
        assert_eq!(
            fs::read(&json_path).expect("read existing output"),
            b"previous json"
        );
        let _ = fs::remove_file(csv_path);
        let _ = fs::remove_file(json_path);
    }

    #[test]
    fn no_progress_flag_is_parsed() {
        let output = temp_output("no_progress");
        let plan = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--no-progress"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("arguments should parse")
        .into_plan()
        .expect("plan should be valid");

        assert!(plan.no_progress);
        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[test]
    fn total_configurations_counts_worker_and_single_thread_kernels_correctly() {
        let output = temp_output("count");
        let plan = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--sizes"),
            OsStr::new("64,128"),
            OsStr::new("--precision"),
            OsStr::new("f32,f64"),
            OsStr::new("--kernel"),
            OsStr::new("naive,rayon-ikj"),
            OsStr::new("--threads"),
            OsStr::new("1,2,4"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("arguments should parse")
        .into_plan()
        .expect("plan should be valid");

        // 2 precisions * 2 sizes * (1 for naive + 3 for rayon-ikj) = 2 * 2 * 4 = 16
        assert_eq!(plan.total_configurations(), 16);
        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mps_parses_as_a_kernel_choice() {
        let output = temp_output("mps");
        let plan = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--kernel"),
            OsStr::new("mps"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("mps kernel should parse")
        .into_plan()
        .expect("mps plan should be valid");

        assert_eq!(plan.kernels, [KernelChoice::Mps]);
        assert_eq!(plan.total_configurations(), 7); // 7 default sizes * 1 precision * 1 config
        let _ = fs::remove_file(output.with_extension("csv"));
        let _ = fs::remove_file(output.with_extension("json"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn mps_with_f64_precision_is_rejected_before_running() {
        let output = temp_output("mps_f64");
        let error = Cli::try_parse_from([
            OsStr::new("rayon-gemm"),
            OsStr::new("--kernel"),
            OsStr::new("mps"),
            OsStr::new("--precision"),
            OsStr::new("f64"),
            OsStr::new("--output"),
            output.as_os_str(),
        ])
        .expect("arguments should parse")
        .into_plan()
        .expect_err("mps with f64 must be rejected");

        assert!(error.contains("f64 is not supported by Metal Performance Shaders"));
        assert!(!output.with_extension("csv").exists());
        assert!(!output.with_extension("json").exists());
    }
}
