#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::fmt::Debug;
use core::marker::PhantomData;

use p3_field::Field;
use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartialWitness, PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use crate::plonk::config::GenericConfig;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Given a `PartitionWitness` that has only inputs set, populates the rest of the witness using the
/// given set of generators.
pub fn generate_partial_witness<
    'a,
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    inputs: PartialWitness<F>,
    prover_data: &'a ProverOnlyCircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
    common_data: &'a CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
) -> PartitionWitness<'a, F>
where
{
    let config = &common_data.config;
    let generators = &prover_data.generators;
    let generator_indices_by_watches = &prover_data.generator_indices_by_watches;

    let mut witness = PartitionWitness::new(
        config.num_wires,
        common_data.degree(),
        &prover_data.representative_map,
    );

    for (t, v) in inputs.target_values.into_iter() {
        witness.set_target(t, v);
    }

    // Build a list of "pending" generators which are queued to be run. Initially, all generators
    // are queued.
    let mut pending_generator_indices: Vec<_> = (0..generators.len()).collect();

    // We also track a list of "expired" generators which have already returned false.
    let mut generator_is_expired = vec![false; generators.len()];
    let mut remaining_generators = generators.len();

    let mut buffer = GeneratedValues::empty();

    // Keep running generators until we fail to make progress.
    while !pending_generator_indices.is_empty() {
        let mut next_pending_generator_indices = Vec::new();

        for &generator_idx in &pending_generator_indices {
            if generator_is_expired[generator_idx] {
                continue;
            }

            let finished = generators[generator_idx].0.run(&witness, &mut buffer);
            if finished {
                generator_is_expired[generator_idx] = true;
                remaining_generators -= 1;
            }

            // Merge any generated values into our witness, and get a list of newly-populated
            // targets' representatives.
            let new_target_reps = buffer
                .target_values
                .drain(..)
                .flat_map(|(t, v)| witness.set_target_returning_rep(t, v));

            // Enqueue unfinished generators that were watching one of the newly populated targets.
            for watch in new_target_reps {
                let opt_watchers = generator_indices_by_watches.get(&watch);
                if let Some(watchers) = opt_watchers {
                    for &watching_generator_idx in watchers {
                        if !generator_is_expired[watching_generator_idx] {
                            next_pending_generator_indices.push(watching_generator_idx);
                        }
                    }
                }
            }
        }

        pending_generator_indices = next_pending_generator_indices;
    }

    assert_eq!(
        remaining_generators, 0,
        "{remaining_generators} generators weren't run"
    );

    witness
}

/// A generator participates in the generation of the witness.
pub trait WitnessGenerator<
    F: RichField + HasExtension<D>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>: 'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    /// Targets to be "watched" by this generator. Whenever a target in the watch list is populated,
    /// the generator will be queued to run.
    fn watch_list(&self) -> Vec<Target>;

    /// Run this generator, returning a flag indicating whether the generator is finished. If the
    /// flag is true, the generator will never be run again, otherwise it will be queued for another
    /// run next time a target in its watch list is populated.
    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool;

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()>;

    fn deserialize(
        src: &mut Buffer,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self>
    where
        Self: Sized;
}

/// A wrapper around an `Box<WitnessGenerator>` which implements `PartialEq`
/// and `Eq` based on generator IDs.
pub struct WitnessGeneratorRef<
    F: RichField + HasExtension<D>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(pub Box<dyn WitnessGenerator<F, D, NUM_HASH_OUT_ELTS>>);

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>
{
    pub fn new<G: WitnessGenerator<F, D, NUM_HASH_OUT_ELTS>>(
        generator: G,
    ) -> WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS> {
        WitnessGeneratorRef(Box::new(generator))
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize> PartialEq
    for WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>
{
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize> Eq
    for WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>
{
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize> Debug
    for WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0.id())
    }
}

/// Values generated by a generator invocation.
#[derive(Debug)]
pub struct GeneratedValues<F: Field> {
    pub target_values: Vec<(Target, F)>,
}

impl<F: Field> From<Vec<(Target, F)>> for GeneratedValues<F> {
    fn from(target_values: Vec<(Target, F)>) -> Self {
        Self { target_values }
    }
}

impl<F: Field> WitnessWrite<F> for GeneratedValues<F> {
    fn set_target(&mut self, target: Target, value: F) {
        self.target_values.push((target, value));
    }
}

impl<F: Field> GeneratedValues<F> {
    pub fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity).into()
    }

    pub fn empty() -> Self {
        Vec::new().into()
    }

    pub fn singleton_wire(wire: Wire, value: F) -> Self {
        Self::singleton_target(Target::Wire(wire), value)
    }

    pub fn singleton_target(target: Target, value: F) -> Self {
        vec![(target, value)].into()
    }

    pub fn singleton_extension_target<const D: usize>(
        et: ExtensionTarget<D>,
        value: F::Extension,
    ) -> Self
    where
        F: RichField + HasExtension<D>,
    {
        let mut witness = Self::with_capacity(D);
        witness.set_extension_target(et, value);
        witness
    }
}

/// A generator which runs once after a list of dependencies is present in the witness.
pub trait SimpleGenerator<
    F: RichField + HasExtension<D>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>: 'static + Send + Sync + Debug
{
    fn id(&self) -> String;

    fn dependencies(&self) -> Vec<Target>;

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>);

    fn adapter(self) -> SimpleGeneratorAdapter<F, Self, D, NUM_HASH_OUT_ELTS>
    where
        Self: Sized,
    {
        SimpleGeneratorAdapter {
            inner: self,
            _phantom: PhantomData,
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()>;

    fn deserialize(
        src: &mut Buffer,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct SimpleGeneratorAdapter<
    F: RichField + HasExtension<D>,
    SG: SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> + ?Sized,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
> {
    _phantom: PhantomData<F>,
    inner: SG,
}

impl<
        F: RichField + HasExtension<D>,
        SG: SimpleGenerator<F, D, NUM_HASH_OUT_ELTS>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    > WitnessGenerator<F, D, NUM_HASH_OUT_ELTS>
    for SimpleGeneratorAdapter<F, SG, D, NUM_HASH_OUT_ELTS>
{
    fn id(&self) -> String {
        self.inner.id()
    }

    fn watch_list(&self) -> Vec<Target> {
        self.inner.dependencies()
    }

    fn run(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) -> bool {
        if witness.contains_all(&self.inner.dependencies()) {
            self.inner.run_once(witness, out_buffer);
            true
        } else {
            false
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        self.inner.serialize(dst, common_data)
    }

    fn deserialize(
        src: &mut Buffer,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        Ok(Self {
            inner: SG::deserialize(src, common_data)?,
            _phantom: PhantomData,
        })
    }
}

/// A generator which copies one wire to another.
#[derive(Debug, Default)]
pub struct CopyGenerator {
    pub(crate) src: Target,
    pub(crate) dst: Target,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for CopyGenerator
{
    fn id(&self) -> String {
        "CopyGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.src]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let value = witness.get_target(self.src);
        out_buffer.set_target(self.dst, value);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target(self.src)?;
        dst.write_target(self.dst)
    }

    fn deserialize(
        source: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let src = source.read_target()?;
        let dst = source.read_target()?;
        Ok(Self { src, dst })
    }
}

/// A generator for including a random value
#[derive(Debug, Default)]
pub struct RandomValueGenerator {
    pub(crate) target: Target,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for RandomValueGenerator
{
    fn id(&self) -> String {
        "RandomValueGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        Vec::new()
    }

    fn run_once(&self, _witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let random_value = F::rand();
        out_buffer.set_target(self.target, random_value);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target(self.target)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let target = src.read_target()?;
        Ok(Self { target })
    }
}

/// A generator for testing if a value equals zero
#[derive(Debug, Default)]
pub struct NonzeroTestGenerator {
    pub(crate) to_test: Target,
    pub(crate) dummy: Target,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for NonzeroTestGenerator
{
    fn id(&self) -> String {
        "NonzeroTestGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.to_test]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let to_test_value = witness.get_target(self.to_test);

        let dummy_value = if to_test_value == F::zero() {
            F::one()
        } else {
            to_test_value.inverse()
        };

        out_buffer.set_target(self.dummy, dummy_value);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target(self.to_test)?;
        dst.write_target(self.dummy)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let to_test = src.read_target()?;
        let dummy = src.read_target()?;
        Ok(Self { to_test, dummy })
    }
}

/// Generator used to fill an extra constant.
#[derive(Debug, Clone, Default)]
pub struct ConstantGenerator<F: Field> {
    pub row: usize,
    pub constant_index: usize,
    pub wire_index: usize,
    pub constant: F,
}

impl<F: Field> ConstantGenerator<F> {
    pub fn set_constant(&mut self, c: F) {
        self.constant = c;
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for ConstantGenerator<F>
{
    fn id(&self) -> String {
        "ConstantGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(&self, _witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        out_buffer.set_target(Target::wire(self.row, self.wire_index), self.constant);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.constant_index)?;
        dst.write_usize(self.wire_index)?;
        dst.write_field(self.constant)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        let constant_index = src.read_usize()?;
        let wire_index = src.read_usize()?;
        let constant = src.read_field()?;
        Ok(Self {
            row,
            constant_index,
            wire_index,
            constant,
        })
    }
}
