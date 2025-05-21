use anyhow::anyhow;
use criterion::{criterion_group, criterion_main, Criterion};
use log::{info, Level};
use p3_baby_bear::BabyBear;
use p3_goldilocks::Goldilocks;
use plonky2::gates::noop::NoopGate;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{
    AlgebraicHasher, GenericConfig, Poseidon2BabyBearConfig, PoseidonGoldilocksConfig,
};
use plonky2::plonk::proof::{ProofWithPublicInputs, ProofWithPublicInputsTarget};
use plonky2::plonk::prover::prove;
use plonky2::util::proving_process_info::ProvingProcessInfo;
use plonky2_field::types::HasExtension;
use tynm::type_name;

mod allocator;

type ProofTuple<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize> = (
    ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
    CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
);

/// Creates a dummy proof which should have `2 ** log2_size` rows.
fn dummy_proof<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    config: &CircuitConfig,
    log2_size: usize,
) -> anyhow::Result<ProofTuple<F, C, D, NUM_HASH_OUT_ELTS>>
where
{
    // 'size' is in degree, but we want number of noop gates. A non-zero amount of padding will be added and size will be rounded to the next power of two. To hit our target size, we go just under the previous power of two and hope padding is less than half the proof.
    let num_dummy_gates = match log2_size {
        0 => return Err(anyhow!("size must be at least 1")),
        1 => 0,
        2 => 1,
        n => (1 << (n - 1)) + 1,
    };
    info!("Constructing inner proof with {num_dummy_gates} gates");
    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
    for _ in 0..num_dummy_gates {
        builder.add_gate(NoopGate, vec![]);
    }
    builder.print_gate_counts(0);

    let data = builder.build::<C>();
    let inputs = PartialWitness::new();

    let mut timing = ProvingProcessInfo::new("prove", Level::Debug);
    let proof =
        prove::<F, C, D, NUM_HASH_OUT_ELTS>(&data.prover_only, &data.common, inputs, &mut timing)?;
    timing.print();
    data.verify(proof.clone())?;

    Ok((proof, data.verifier_only, data.common))
}

fn get_recursive_circuit_data<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    InnerC: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    input_proof_common_circuit_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    config: &CircuitConfig,
) -> (
    ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    CircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
)
where
    InnerC::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
    let input_proof_target = builder.add_virtual_proof_with_pis(input_proof_common_circuit_data);

    let input_proof_verifier_data_target = builder
        .add_virtual_verifier_data(input_proof_common_circuit_data.config.fri_config.cap_height);

    builder.verify_proof::<InnerC>(
        &input_proof_target,
        &input_proof_verifier_data_target,
        input_proof_common_circuit_data,
    );
    let zero = builder.zero();
    builder.register_public_input(zero);
    builder.print_gate_counts(0);

    let circuit_data = builder.build::<C>();
    (
        input_proof_target,
        input_proof_verifier_data_target,
        circuit_data,
    )
}
fn recursive_proof<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    input_proof: &ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    input_verifier_data: &VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,

    input_proof_target: &ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    input_proof_verifier_data_target: &VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    circuit_data: &CircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
) -> anyhow::Result<ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>>
where
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let mut pw = PartialWitness::new();
    pw.set_proof_with_pis_target(input_proof_target, input_proof);
    pw.set_verifier_data_target(input_proof_verifier_data_target, input_verifier_data);

    let mut timing = ProvingProcessInfo::new("prove", Level::Info);
    let proof = prove::<F, C, D, NUM_HASH_OUT_ELTS>(
        &circuit_data.prover_only,
        &circuit_data.common,
        pw,
        &mut timing,
    )?;
    //timing.print();

    Ok(proof)
}

pub(crate) fn bench_recursion<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    c: &mut Criterion,
    config: &CircuitConfig,
) where
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let inner = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(config, 12).unwrap();
    let (_, _, common_data) = &inner;
    info!(
        "Initial proof degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );

    // Recursively verify the proof
    let (inner_proof_target, inner_proof_verifier_data_target, middle_circuit_data) =
        get_recursive_circuit_data::<F, C, C, D, NUM_HASH_OUT_ELTS>(&inner.2, config);
    let middle_proof = recursive_proof::<F, C, D, NUM_HASH_OUT_ELTS>(
        &inner.0,
        &inner.1,
        &inner_proof_target,
        &inner_proof_verifier_data_target,
        &middle_circuit_data,
    )
    .unwrap();
    info!(
        "Single recursion proof degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );
    let (middle_proof_target, middle_proof_verifier_data_target, outer_circuit_data) =
        get_recursive_circuit_data::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            &middle_circuit_data.common,
            config,
        );

    let mut group = c.benchmark_group(format!("recursion<{}>", type_name::<F>()));
    group.sample_size(20);
    group.bench_function(format!("recursive_proof<{}>", type_name::<F>()), |b| {
        b.iter(|| {
            let _outer_proof = recursive_proof::<F, C, D, NUM_HASH_OUT_ELTS>(
                &middle_proof,
                &middle_circuit_data.verifier_only,
                &middle_proof_target,
                &middle_proof_verifier_data_target,
                &outer_circuit_data,
            )
            .unwrap();
            info!(
                "Double recursion proof degree {} = 2^{}",
                common_data.degree(),
                common_data.degree_bits()
            );
        });
    });
}

pub(crate) fn bench_merge<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    c: &mut Criterion,
    config: &CircuitConfig,
) where
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let inner = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(config, 12).unwrap();
    let (_, _, common_data) = &inner;
    info!(
        "Initial proof degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );

    // Recursively verify the proof
    let (inner_proof_target, inner_proof_verifier_data_target, middle_circuit_data) =
        get_recursive_circuit_data::<F, C, C, D, NUM_HASH_OUT_ELTS>(&inner.2, config);
    let middle_proof = recursive_proof::<F, C, D, NUM_HASH_OUT_ELTS>(
        &inner.0,
        &inner.1,
        &inner_proof_target,
        &inner_proof_verifier_data_target,
        &middle_circuit_data,
    )
    .unwrap();
    info!(
        "Single recursion proof degree {} = 2^{}",
        common_data.degree(),
        common_data.degree_bits()
    );
    let (
        middle_proof_target_one,
        middle_proof_target_two,
        middle_proof_verifier_data_target,
        outer_circuit_data,
    ) = get_merge_circuit_data::<F, C, C, D, NUM_HASH_OUT_ELTS>(
        &middle_circuit_data.common,
        config,
    );

    let mut group = c.benchmark_group(format!("recursion<{}>", type_name::<F>()));
    group.sample_size(20);
    group.bench_function(format!("merge_proofs<{}>", type_name::<F>()), |b| {
        b.iter(|| {
            let _outer_proof = merge_proof::<F, C, D, NUM_HASH_OUT_ELTS>(
                &middle_proof,
                &middle_circuit_data.verifier_only,
                &middle_proof_target_one,
                &middle_proof_target_two,
                &middle_proof_verifier_data_target,
                &outer_circuit_data,
            )
            .unwrap();
            info!(
                "Double recursion proof degree {} = 2^{}",
                common_data.degree(),
                common_data.degree_bits()
            );
        });
    });
}
fn get_merge_circuit_data<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    InnerC: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    input_proof_common_circuit_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    config: &CircuitConfig,
) -> (
    ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    CircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
)
where
    InnerC::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
    let input_proof_target_one =
        builder.add_virtual_proof_with_pis(input_proof_common_circuit_data);
    let input_proof_target_two =
        builder.add_virtual_proof_with_pis(input_proof_common_circuit_data);

    let input_proof_verifier_data_target = builder
        .add_virtual_verifier_data(input_proof_common_circuit_data.config.fri_config.cap_height);

    builder.verify_proof::<InnerC>(
        &input_proof_target_one,
        &input_proof_verifier_data_target,
        input_proof_common_circuit_data,
    );
    builder.verify_proof::<InnerC>(
        &input_proof_target_two,
        &input_proof_verifier_data_target,
        input_proof_common_circuit_data,
    );
    builder.print_gate_counts(0);

    let circuit_data = builder.build::<C>();
    (
        input_proof_target_one,
        input_proof_target_two,
        input_proof_verifier_data_target,
        circuit_data,
    )
}
fn merge_proof<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    input_proof: &ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    input_verifier_data: &VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,

    input_proof_target_one: &ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    input_proof_target_two: &ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    input_proof_verifier_data_target: &VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    circuit_data: &CircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
) -> anyhow::Result<ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>>
where
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    let mut pw = PartialWitness::new();
    pw.set_proof_with_pis_target(input_proof_target_one, input_proof);
    pw.set_proof_with_pis_target(input_proof_target_two, input_proof);
    pw.set_verifier_data_target(input_proof_verifier_data_target, input_verifier_data);

    let mut timing = ProvingProcessInfo::new("prove", Level::Info);
    let proof = prove::<F, C, D, NUM_HASH_OUT_ELTS>(
        &circuit_data.prover_only,
        &circuit_data.common,
        pw,
        &mut timing,
    )?;
    //timing.print();

    Ok(proof)
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_recursion::<Goldilocks, PoseidonGoldilocksConfig, 2, 4>(
        c,
        &CircuitConfig::standard_recursion_config_gl(),
    );
    bench_recursion::<BabyBear, Poseidon2BabyBearConfig, 4, 8>(
        c,
        &CircuitConfig::standard_recursion_config_bb_wide(),
    );

    bench_merge::<Goldilocks, PoseidonGoldilocksConfig, 2, 4>(
        c,
        &CircuitConfig::standard_recursion_config_gl(),
    );
    bench_merge::<BabyBear, Poseidon2BabyBearConfig, 4, 8>(
        c,
        &CircuitConfig::standard_recursion_config_bb_wide(),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
