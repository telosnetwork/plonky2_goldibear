#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::ops::Range;

use p3_field::{AbstractExtensionField, AbstractField};

use plonky2_field::extension_algebra::ExtensionAlgebra;
use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon2_babybear::SPONGE_WIDTH;
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

use super::poseidon2_babybear::INTERNAL_DIAG_SHIFTS;

/// Poseidon2 BabyBear internal
#[derive(Clone, Debug, Default)]
pub struct Poseidon2InternalPermutationGate<F: RichField + HasExtension<D>, const D: usize>(
    PhantomData<F>,
);

impl<F: RichField + HasExtension<D>, const D: usize> Poseidon2InternalPermutationGate<F, D>
where
    
{
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    pub(crate) const fn wires_input(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        i * D..(i + 1) * D
    }

    pub(crate) const fn wires_output(i: usize) -> Range<usize> {
        assert!(i < SPONGE_WIDTH);
        (SPONGE_WIDTH + i) * D..(SPONGE_WIDTH + i + 1) * D
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for Poseidon2InternalPermutationGate<F, D>
where
    F: HasExtension<D>,
    
{
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(
        _src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        Ok(Poseidon2InternalPermutationGate::new())
    }

    fn eval_unfiltered(
        &self,
        vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<<F as HasExtension<D>>::Extension> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut state = inputs;
        state
            .iter_mut()
            .for_each(|x| *x = x.scalar_mul(F::Extension::from_canonical_u32(943718400)));
        let part_sum: ExtensionAlgebra<F, D> = state
            .iter()
            .skip(1)
            .fold(ExtensionAlgebra::<F, D>::zero(), |acc, x| acc + x.clone());
        let full_sum = part_sum.clone() + state[0].clone();
        state[0] = part_sum.clone() - state[0].clone();

        for i in 0..INTERNAL_DIAG_SHIFTS.len() {
            state[i + 1] = full_sum.clone()
                + state[i + 1]
                    .clone()
                    .scalar_mul(F::Extension::from_canonical_u32(
                        1 << INTERNAL_DIAG_SHIFTS[i],
                    ));
        }

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(state)
            .flat_map(|(out, computed_out)| (out - computed_out).to_basefield_array())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut state = inputs;
        state
            .iter_mut()
            .for_each(|x| *x *= F::Extension::from_canonical_u32(943718400));
        let part_sum: F::Extension = state
            .iter()
            .skip(1)
            .fold(F::Extension::zero(), |acc, x| acc + x.clone());
        let full_sum = part_sum.clone() + state[0].clone();
        state[0] = part_sum.clone() - state[0].clone();

        for i in 0..INTERNAL_DIAG_SHIFTS.len() {
            state[i + 1] = full_sum.clone()
                + state[i + 1].clone()
                    * F::Extension::from_canonical_u32(1 << INTERNAL_DIAG_SHIFTS[i]);
        }

        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext(Self::wires_output(i));
            let basefield_array: [F; D] =
                <<F as HasExtension<D>>::Extension as AbstractExtensionField<F>>::as_base_slice(
                    &(out - state[i]),
                )
                .try_into()
                .unwrap();
            yield_constr.many(basefield_array);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut state = inputs;
        let my_const = builder.constant_extension(F::Extension::from_canonical_u32(943718400));
        state
            .iter_mut()
            .for_each(|x| *x = builder.scalar_mul_ext_algebra(my_const, *x));
        let part_sum: ExtensionAlgebraTarget<D> = state
            .iter()
            .skip(1)
            .fold(builder.zero_ext_algebra(), |acc, x| {
                builder.add_ext_algebra(acc, *x)
            });
        let full_sum = builder.add_ext_algebra(part_sum, state[0]);
        state[0] = builder.sub_ext_algebra(part_sum, state[0]);

        for i in 0..INTERNAL_DIAG_SHIFTS.len() {
            let shift = builder.constant_extension(F::Extension::from_canonical_u32(
                1 << INTERNAL_DIAG_SHIFTS[i],
            ));
            state[i + 1] = builder.scalar_mul_add_ext_algebra(shift, state[i + 1], full_sum);
        }

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(state)
            .flat_map(|(out, computed_out)| {
                builder
                    .sub_ext_algebra(out, computed_out)
                    .to_ext_target_array()
            })
            .collect()
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        let gen = Poseidon2InternalPermutationGenerator { row };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        2 * D * SPONGE_WIDTH
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * D
    }
}

#[derive(Clone, Debug, Default)]
pub struct Poseidon2InternalPermutationGenerator<const D: usize> {
    row: usize,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for Poseidon2InternalPermutationGenerator<D>
where
    
{
    fn id(&self) -> String {
        "Poseidon2InternalPermutationGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .flat_map(|i| {
                Target::wires_from_range(
                    self.row,
                    Poseidon2InternalPermutationGate::<F, D>::wires_input(i),
                )
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_ext_target =
            |wire_range| ExtensionTarget::<D>::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_ext_target(wire_range));

        let inputs: [F::Extension; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| get_local_ext(Poseidon2InternalPermutationGate::<F, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut state = inputs;
        state
            .iter_mut()
            .for_each(|x| *x *= F::Extension::from_canonical_u32(943718400));
        let part_sum: F::Extension = state
            .iter()
            .skip(1)
            .fold(F::Extension::zero(), |acc, x| acc + x.clone());
        let full_sum = part_sum.clone() + state[0].clone();
        state[0] = part_sum.clone() - state[0].clone();

        for i in 0..INTERNAL_DIAG_SHIFTS.len() {
            state[i + 1] = full_sum.clone()
                + state[i + 1].clone()
                    * F::Extension::from_canonical_u32(1 << INTERNAL_DIAG_SHIFTS[i]);
        }

        for (i, &out) in state.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_ext_target(Poseidon2InternalPermutationGate::<F, D>::wires_output(i)),
                out,
            );
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self { row })
    }
}

#[cfg(test)]
mod tests {
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon2_internal_permutation::Poseidon2InternalPermutationGate;
    use crate::hash::hash_types::BABYBEAR_NUM_HASH_OUT_ELTS;
    use crate::plonk::config::{GenericConfig, Poseidon2BabyBearConfig};

    #[test]
    fn low_degree() {
        const D: usize = 4;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = Poseidon2InternalPermutationGate::<F, D>::new();
        test_low_degree::<F, Poseidon2InternalPermutationGate<F, D>, D, NUM_HASH_OUT_ELTS>(gate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 4;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = Poseidon2InternalPermutationGate::new();
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}
