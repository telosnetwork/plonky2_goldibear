use anyhow::Result;
use p3_field::AbstractField;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2_field::{GOLDILOCKS_EXTENSION_FIELD_DEGREE, GOLDILOCKS_NUM_HASH_OUT_ELTS};

/// An example of using Plonky2 to prove that a given value lies in a given range.
fn main() -> Result<()> {
    const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
    const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

    let config = CircuitConfig::standard_recursion_config_gl();
    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);

    // The secret value.
    let value = builder.add_virtual_target();

    // Registered as a public input (even though it's secret) so we can print out the value later.
    builder.register_public_input(value);

    let log_max = 6;
    builder.range_check(value, log_max);

    let mut pw = PartialWitness::new();
    pw.set_target(value, F::from_canonical_usize(42));

    let data = builder.build::<C>();
    let proof = data.prove(pw)?;

    println!(
        "Value {} is less than 2^{}",
        proof.public_inputs[0], log_max,
    );

    data.verify(proof)
}
