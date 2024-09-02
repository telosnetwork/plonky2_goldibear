#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use p3_field::TwoAdicField;
use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::WitnessGeneratorRef;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBaseBatch};
use crate::util::serialization::{Buffer, IoResult};

/// A gate which does nothing.
#[derive(Debug)]
pub struct NoopGate;

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize> Gate<F, D, NUM_HASH_OUT_ELTS> for NoopGate
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        "NoopGate".into()
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>) -> IoResult<Self> {
        Ok(Self)
    }

    fn eval_unfiltered(&self, _vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        Vec::new()
    }

    fn eval_unfiltered_base_batch(&self, _vars: EvaluationVarsBaseBatch<F, NUM_HASH_OUT_ELTS>) -> Vec<F> {
        Vec::new()
    }

    fn eval_unfiltered_circuit(
        &self,
        _builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        _vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        Vec::new()
    }

    fn generators(&self, _row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        0
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        0
    }

    fn num_constraints(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use p3_goldilocks::Goldilocks;

    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::noop::NoopGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<Goldilocks, _, 2, 4>(NoopGate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = 4;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(NoopGate)
    }
}
