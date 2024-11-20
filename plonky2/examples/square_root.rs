use core::marker::PhantomData;

use anyhow::Result;
use p3_field::{PrimeField64, TwoAdicField};
use plonky2::gates::arithmetic_base::ArithmeticBaseGenerator;
use plonky2::gates::poseidon_goldilocks::PoseidonGenerator;
use plonky2::gates::poseidon_goldilocks_mds::PoseidonMdsGenerator;
use plonky2::hash::hash_types::{GOLDILOCKS_NUM_HASH_OUT_ELTS, RichField};
use plonky2::iop::generator::{
    ConstantGenerator, GeneratedValues, RandomValueGenerator, SimpleGenerator,
};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData, CommonCircuitData};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::recursion::dummy_circuit::DummyProofGenerator;
use plonky2::util::serialization::{
    Buffer, DefaultGateSerializer, IoResult, Read, WitnessGeneratorSerializer, Write,
};
use plonky2::{get_generator_tag_impl, impl_generator_serializer, read_generator_impl};
use plonky2_field::types::{HasExtension, Sample};

/// A generator used by the prover to calculate the square root (`x`) of a given value
/// (`x_squared`), outside of the circuit, in order to supply it as an additional public input.
#[derive(Debug, Default)]
struct SquareRootGenerator<F: RichField + HasExtension<D>, const D: usize>
where
    
{
    x: Target,
    x_squared: Target,
    _phantom: PhantomData<F>,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for SquareRootGenerator<F, D>
where
    
{
    fn id(&self) -> String {
        "SquareRootGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.x_squared]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let x_squared = witness.get_target(self.x_squared);
        let x = sqrt(x_squared).unwrap();

        println!("Square root: {x}");

        out_buffer.set_target(self.x, x);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target(self.x)?;
        dst.write_target(self.x_squared)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let x = src.read_target()?;
        let x_squared = src.read_target()?;
        Ok(Self {
            x,
            x_squared,
            _phantom: PhantomData,
        })
    }
}

#[derive(Default)]
pub struct CustomGeneratorSerializer<
    C: GenericConfig<D, NUM_HASH_OUT_ELTS>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
> {
    pub _phantom: PhantomData<C>,
}

impl<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    WitnessGeneratorSerializer<F, D, NUM_HASH_OUT_ELTS>
    for CustomGeneratorSerializer<C, D, NUM_HASH_OUT_ELTS>
where
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension> + 'static,
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    
{
    impl_generator_serializer! {
        CustomGeneratorSerializer,
        DummyProofGenerator<F, C, D, NUM_HASH_OUT_ELTS>,
        ArithmeticBaseGenerator<F, D>,
        ConstantGenerator<F>,
        PoseidonGenerator<F, D>,
        PoseidonMdsGenerator<D>,
        RandomValueGenerator,
        SquareRootGenerator<F, D>
    }
}

/// An example of using Plonky2 to prove a statement of the form
/// "I know the square root of this field element."
fn main() -> Result<()> {
    const D: usize = 2;
    const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

    let config = CircuitConfig::standard_recursion_config_gl();

    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);

    let x = builder.add_virtual_target();
    let x_squared = builder.square(x);

    builder.register_public_input(x_squared);

    builder.add_simple_generator(SquareRootGenerator::<F, D> {
        x,
        x_squared,
        _phantom: PhantomData,
    });

    // Randomly generate the value of x^2: any quadratic residue in the field works.
    let x_squared_value = {
        let mut val = F::rand();
        while !is_quadratic_residue(val) {
            val = F::rand();
        }
        val
    };

    let mut pw = PartialWitness::new();
    pw.set_target(x_squared, x_squared_value);

    let data = builder.build::<C>();
    let proof = data.prove(pw.clone())?;

    let x_squared_actual = proof.public_inputs[0];
    println!("Field element (square): {x_squared_actual}");

    // Test serialization
    {
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = CustomGeneratorSerializer::<C, D, NUM_HASH_OUT_ELTS>::default();

        let data_bytes = data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|_| anyhow::Error::msg("CircuitData serialization failed."))?;

        let data_from_bytes = CircuitData::<F, C, D, NUM_HASH_OUT_ELTS>::from_bytes(
            &data_bytes,
            &gate_serializer,
            &generator_serializer,
        )
        .map_err(|_| anyhow::Error::msg("CircuitData deserialization failed."))?;

        assert_eq!(data, data_from_bytes);
    }

    data.verify(proof)
}

fn is_quadratic_residue<F: PrimeField64>(x: F) -> bool {
    if x.is_zero() {
        return true;
    }
    // This is based on Euler's criterion.
    let power = F::neg_one().as_canonical_u64() / 2;
    let exp = x.exp_u64(power);
    if exp == F::one() {
        return true;
    }
    if exp == F::neg_one() {
        return false;
    }
    panic!("Unreachable")
}

fn sqrt<F: PrimeField64 + TwoAdicField>(x: F) -> Option<F> {
    if x.is_zero() {
        Some(x)
    } else if is_quadratic_residue(x) {
        let t = (F::ORDER_U64 - 1) / ((2u64).pow(F::TWO_ADICITY.try_into().unwrap()));
        let mut z = F::two_adic_generator(F::bits());
        let mut w = x.exp_u64((t - 1) / 2);
        let mut x = w * x;
        let mut b = x * w;

        let mut v = F::TWO_ADICITY;

        while !b.is_one() {
            let mut k = 0usize;
            let mut b2k = b;
            while !b2k.is_one() {
                b2k = b2k * b2k;
                k += 1;
            }
            let j = v - k - 1;
            w = z;
            for _ in 0..j {
                w = w * w;
            }

            z = w * w;
            b *= z;
            x *= w;
            v = k;
        }
        Some(x)
    } else {
        None
    }
}
