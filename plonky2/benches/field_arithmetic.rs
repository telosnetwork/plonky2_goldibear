use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{batch_multiplicative_inverse, TwoAdicField};
use p3_goldilocks::Goldilocks;
use plonky2_field::types::Sample;
use tynm::type_name;

mod allocator;

pub(crate) fn bench_field<F: TwoAdicField + Sample>(c: &mut Criterion) {
    c.bench_function(&format!("mul-throughput<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || (F::rand(), F::rand(), F::rand(), F::rand()),
            |(mut x, mut y, mut z, mut w)| {
                for _ in 0..25 {
                    (x, y, z, w) = (x * y, y * z, z * w, w * x);
                }
                (x, y, z, w)
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("mul-latency<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || F::rand(),
            |mut x| {
                for _ in 0..100 {
                    x = x * x;
                }
                x
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("sqr-throughput<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || (F::rand(), F::rand(), F::rand(), F::rand()),
            |(mut x, mut y, mut z, mut w)| {
                for _ in 0..25 {
                    (x, y, z, w) = (x.square(), y.square(), z.square(), w.square());
                }
                (x, y, z, w)
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("sqr-latency<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || F::rand(),
            |mut x| {
                for _ in 0..100 {
                    x = x.square();
                }
                x
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("add-throughput<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || {
                (
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                    F::rand(),
                )
            },
            |(mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j)| {
                for _ in 0..10 {
                    (a, b, c, d, e, f, g, h, i, j) = (
                        a + b,
                        b + c,
                        c + d,
                        d + e,
                        e + f,
                        f + g,
                        g + h,
                        h + i,
                        i + j,
                        j + a,
                    );
                }
                (a, b, c, d, e, f, g, h, i, j)
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("add-latency<{}>", type_name::<F>()), |b| {
        b.iter_batched(
            || F::rand(),
            |mut x| {
                for _ in 0..100 {
                    x = x + x;
                }
                x
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function(&format!("try_inverse<{}>", type_name::<F>()), |b| {
        b.iter_batched(|| F::rand(), |x| x.try_inverse(), BatchSize::SmallInput)
    });

    c.bench_function(
        &format!(
            "batch_multiplicative_inverse::<F>-tiny<{}>",
            type_name::<F>()
        ),
        |b| {
            b.iter_batched(
                || (0..2).map(|_| F::rand()).collect::<Vec<_>>(),
                |x| batch_multiplicative_inverse::<F>(&x),
                BatchSize::SmallInput,
            )
        },
    );

    c.bench_function(
        &format!(
            "batch_multiplicative_inverse::<F>-small<{}>",
            type_name::<F>()
        ),
        |b| {
            b.iter_batched(
                || (0..4).map(|_| F::rand()).collect::<Vec<_>>(),
                |x| batch_multiplicative_inverse::<F>(&x),
                BatchSize::SmallInput,
            )
        },
    );

    c.bench_function(
        &format!(
            "batch_multiplicative_inverse::<F>-medium<{}>",
            type_name::<F>()
        ),
        |b| {
            b.iter_batched(
                || (0..16).map(|_| F::rand()).collect::<Vec<_>>(),
                |x| batch_multiplicative_inverse::<F>(&x),
                BatchSize::SmallInput,
            )
        },
    );

    c.bench_function(
        &format!(
            "batch_multiplicative_inverse::<F>-large<{}>",
            type_name::<F>()
        ),
        |b| {
            b.iter_batched(
                || (0..256).map(|_| F::rand()).collect::<Vec<_>>(),
                |x| batch_multiplicative_inverse::<F>(&x),
                BatchSize::LargeInput,
            )
        },
    );

    c.bench_function(
        &format!(
            "batch_multiplicative_inverse::<F>-huge<{}>",
            type_name::<F>()
        ),
        |b| {
            b.iter_batched(
                || (0..65536).map(|_| F::rand()).collect::<Vec<_>>(),
                |x| batch_multiplicative_inverse::<F>(&x),
                BatchSize::LargeInput,
            )
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_field::<Goldilocks>(c);
    bench_field::<BinomialExtensionField<Goldilocks, 2>>(c);
    bench_field::<BabyBear>(c);
    bench_field::<BinomialExtensionField<BabyBear, 4>>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
