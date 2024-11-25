use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::TwoAdicField;
use p3_goldilocks::Goldilocks;
use tynm::type_name;

use plonky2::field::polynomial::PolynomialCoeffs;
use plonky2_field::types::Sample;

mod allocator;

pub(crate) fn bench_ffts<F: TwoAdicField + Sample>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("fft<{}>", type_name::<F>()));

    for size_log in [14, 16, 18] {
        let size = 1 << size_log;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let coeffs = PolynomialCoeffs::new(F::rand_vec(size));
            b.iter(|| coeffs.clone().fft_with_options(None, None));
        });
    }
}

pub(crate) fn bench_ldes<F: TwoAdicField + Sample>(c: &mut Criterion) {
    const RATE_BITS: usize = 3;

    let mut group = c.benchmark_group(format!("lde<{}>", type_name::<F>()));

    for size_log in [13, 14, 15, 16] {
        let orig_size = 1 << (size_log - RATE_BITS);
        let lde_size = 1 << size_log;

        group.bench_with_input(BenchmarkId::from_parameter(lde_size), &lde_size, |b, _| {
            let coeffs = PolynomialCoeffs::new(F::rand_vec(orig_size));
            b.iter(|| {
                let padded_coeffs = coeffs.lde(RATE_BITS);
                padded_coeffs.fft_with_options(Some(RATE_BITS), None)
            });
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_ffts::<Goldilocks>(c);
    bench_ffts::<BabyBear>(c);
    bench_ldes::<Goldilocks>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
