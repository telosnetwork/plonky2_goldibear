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
use crate::hash::poseidon_goldilocks::{PoseidonGoldilocks, SPONGE_WIDTH};
use crate::iop::ext_target::{ExtensionAlgebraTarget, ExtensionTarget};
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Poseidon MDS Gate
#[derive(Debug, Default)]
pub struct PoseidonMdsGate<F: RichField + HasExtension<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + HasExtension<D>, const D: usize> PoseidonMdsGate<F, D> {
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

    // Following are methods analogous to ones in `Poseidon`, but for extension algebras.

    /// Same as `mds_row_shf` for an extension algebra of `F`.
    fn mds_row_shf_algebra(
        r: usize,
        v: &[ExtensionAlgebra<F, D>; SPONGE_WIDTH],
    ) -> ExtensionAlgebra<F, D> {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = ExtensionAlgebra::zero();

        for i in 0..SPONGE_WIDTH {
            let coeff = <<F as HasExtension<D>>::Extension as AbstractField>::from_canonical_u64(
                PoseidonGoldilocks::MDS_MATRIX_CIRC[i],
            );
            res += v[(i + r) % SPONGE_WIDTH].scalar_mul(coeff);
        }
        {
            let coeff = <<F as HasExtension<D>>::Extension as AbstractField>::from_canonical_u64(
                PoseidonGoldilocks::MDS_MATRIX_DIAG[r],
            );
            res += v[r].scalar_mul(coeff);
        }

        res
    }

    /// Same as `mds_row_shf_recursive` for an extension algebra of `F`.
    fn mds_row_shf_algebra_circuit<const NUM_HASH_OUT_ELTS: usize>(
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        r: usize,
        v: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> ExtensionAlgebraTarget<D> {
        debug_assert!(r < SPONGE_WIDTH);
        let mut res = builder.zero_ext_algebra();

        for i in 0..SPONGE_WIDTH {
            let coeff =
                builder.constant_extension(<F as HasExtension<D>>::Extension::from_canonical_u64(
                    PoseidonGoldilocks::MDS_MATRIX_CIRC[i],
                ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[(i + r) % SPONGE_WIDTH], res);
        }
        {
            let coeff =
                builder.constant_extension(<F as HasExtension<D>>::Extension::from_canonical_u64(
                    PoseidonGoldilocks::MDS_MATRIX_DIAG[r],
                ));
            res = builder.scalar_mul_add_ext_algebra(coeff, v[r], res);
        }

        res
    }

    /// Same as `mds_layer` for an extension algebra of `F`.
    fn mds_layer_algebra(
        state: &[ExtensionAlgebra<F, D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebra<F, D>; SPONGE_WIDTH] {
        let mut result = [ExtensionAlgebra::zero(); SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_algebra(r, state);
        }

        result
    }

    /// Same as `mds_layer_recursive` for an extension algebra of `F`.
    fn mds_layer_algebra_circuit<const NUM_HASH_OUT_ELTS: usize>(
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        state: &[ExtensionAlgebraTarget<D>; SPONGE_WIDTH],
    ) -> [ExtensionAlgebraTarget<D>; SPONGE_WIDTH] {
        let mut result = [builder.zero_ext_algebra(); SPONGE_WIDTH];

        for r in 0..SPONGE_WIDTH {
            result[r] = Self::mds_row_shf_algebra_circuit(builder, r, state);
        }

        result
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for PoseidonMdsGate<F, D>
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
        Ok(PoseidonMdsGate::new())
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

        let computed_outputs = Self::mds_layer_algebra(&inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
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

        let computed_outputs = PoseidonGoldilocks::mds_layer_field::<F, F::Extension>(&inputs);

        for i in 0..SPONGE_WIDTH {
            let out = vars.get_local_ext(Self::wires_output(i));
            let basefield_array: [F; D] =
                <<F as HasExtension<D>>::Extension as AbstractExtensionField<F>>::as_base_slice(
                    &(out - computed_outputs[i]),
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

        let computed_outputs = Self::mds_layer_algebra_circuit(builder, &inputs);

        (0..SPONGE_WIDTH)
            .map(|i| vars.get_local_ext_algebra(Self::wires_output(i)))
            .zip(computed_outputs)
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
        let gen = PoseidonMdsGenerator { row };
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
pub struct PoseidonMdsGenerator<const D: usize> {
    row: usize,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for PoseidonMdsGenerator<D>
{
    fn id(&self) -> String {
        "PoseidonMdsGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .flat_map(|i| {
                Target::wires_from_range(self.row, PoseidonMdsGate::<F, D>::wires_input(i))
            })
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let get_local_get_target =
            |wire_range| ExtensionTarget::<D>::from_range(self.row, wire_range);
        let get_local_ext =
            |wire_range| witness.get_extension_target(get_local_get_target(wire_range));

        let inputs: [_; SPONGE_WIDTH] = (0..SPONGE_WIDTH)
            .map(|i| get_local_ext(PoseidonMdsGate::<F, D>::wires_input(i)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let outputs = PoseidonGoldilocks::mds_layer_field(&inputs);

        for (i, &out) in outputs.iter().enumerate() {
            out_buffer.set_extension_target(
                get_local_get_target(PoseidonMdsGate::<F, D>::wires_output(i)),
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
    use plonky2_field::{GOLDILOCKS_EXTENSION_FIELD_DEGREE, GOLDILOCKS_NUM_HASH_OUT_ELTS};

    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::poseidon_goldilocks_mds::PoseidonMdsGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = PoseidonMdsGate::<F, D>::new();
        test_low_degree::<F, PoseidonMdsGate<F, D>, D, NUM_HASH_OUT_ELTS>(gate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = PoseidonMdsGate::new();
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}
