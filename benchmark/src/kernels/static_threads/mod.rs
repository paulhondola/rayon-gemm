mod ikj;
mod tiled;

pub use ikj::StaticIkjGemm;
pub use tiled::StaticTiledGemm;

/// Row counts per worker, as OpenMP's static schedule assigns them: every
/// worker gets `n / threads` rows and the first `n % threads` get one more.
pub(crate) fn static_row_counts(n: usize, threads: usize) -> impl Iterator<Item = usize> {
    (0..threads).map(move |worker| n / threads + usize::from(worker < n % threads))
}

#[cfg(test)]
mod tests {
    use super::static_row_counts;

    #[test]
    fn static_schedule_uses_every_thread_with_balanced_rows() {
        assert_eq!(
            static_row_counts(64, 10).collect::<Vec<_>>(),
            [7, 7, 7, 7, 6, 6, 6, 6, 6, 6]
        );
        assert_eq!(static_row_counts(64, 12).count(), 12);
        assert_eq!(static_row_counts(7, 7).collect::<Vec<_>>(), [1; 7]);
    }
}
