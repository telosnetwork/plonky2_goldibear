use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_goldilocks::Goldilocks;
use plonky2::hash::hash_types::RichField;
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon_goldilocks::Poseidon64Hash;
use plonky2::plonk::config::Hasher;
use tynm::type_name;

mod allocator;

const ELEMS_PER_LEAF: usize = 135;

pub(crate) fn bench_merkle_tree<F: RichField, H: Hasher<F>>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!(
        "merkle-tree<{}, {}>",
        type_name::<F>(),
        type_name::<H>()
    ));
    group.sample_size(10);

    for size_log in [13, 14, 15] {
        let size = 1 << size_log;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            let leaves = vec![F::rand_vec(ELEMS_PER_LEAF); size];
            b.iter(|| MerkleTree::<F, H>::new(leaves.clone(), 0));
        });
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_merkle_tree::<Goldilocks, Poseidon64Hash>(c);
    bench_merkle_tree::<Goldilocks, KeccakHash<25>>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
