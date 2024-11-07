#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use p3_field::{AbstractField, PackedField, TwoAdicField};

use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate for raising a value to a power.
#[derive(Clone, Debug, Default)]
pub struct ExponentiationGate<F: RichField + HasExtension<D>, const D: usize>
where
    F::Extension: TwoAdicField,
{
    pub num_power_bits: usize,
    pub _phantom: PhantomData<F>,
}

impl<F: RichField + HasExtension<D>, const D: usize> ExponentiationGate<F, D>
where
    F::Extension: TwoAdicField,
{
    pub const fn new(num_power_bits: usize) -> Self {
        Self {
            num_power_bits,
            _phantom: PhantomData,
        }
    }

    pub fn new_from_config(config: &CircuitConfig) -> Self {
        let num_power_bits = Self::max_power_bits(config.num_wires, config.num_routed_wires);
        Self::new(num_power_bits)
    }

    fn max_power_bits(num_wires: usize, num_routed_wires: usize) -> usize {
        // 2 wires are reserved for the base and output.
        let max_for_routed_wires = num_routed_wires - 2;
        let max_for_wires = (num_wires - 2) / 2;
        max_for_routed_wires.min(max_for_wires)
    }

    pub(crate) const fn wire_base(&self) -> usize {
        0
    }

    /// The `i`th bit of the exponent, in little-endian order.
    pub(crate) const fn wire_power_bit(&self, i: usize) -> usize {
        debug_assert!(i < self.num_power_bits);
        1 + i
    }

    pub const fn wire_output(&self) -> usize {
        1 + self.num_power_bits
    }

    pub(crate) const fn wire_intermediate_value(&self, i: usize) -> usize {
        debug_assert!(i < self.num_power_bits);
        2 + self.num_power_bits + i
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for ExponentiationGate<F, D>
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        format!("{self:?}<D={D}>")
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.num_power_bits)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let num_power_bits = src.read_usize()?;
        Ok(Self::new(num_power_bits))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>) -> Vec<F::Extension> {
        let base = vars.local_wires[self.wire_base()];

        let power_bits: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_power_bit(i)])
            .collect();
        let intermediate_values: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_intermediate_value(i)])
            .collect();

        let output = vars.local_wires[self.wire_output()];

        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );

        for i in 0..self.num_power_bits {
            let prev_intermediate_value = if i == 0 {
                <F::Extension as AbstractField>::one()
            } else {
                <F::Extension as AbstractField>::square(&intermediate_values[i - 1])
            };

            // power_bits is in LE order, but we accumulate in BE order.
            let cur_bit = power_bits[self.num_power_bits - i - 1];

            let not_cur_bit = <F::Extension as AbstractField>::one() - cur_bit;
            let computed_intermediate_value =
                prev_intermediate_value * (cur_bit * base + not_cur_bit);
            constraints.push(computed_intermediate_value - intermediate_values[i]);
        }

        constraints.push(output - intermediate_values[self.num_power_bits - 1]);

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(
        &self,
        vars_base: EvaluationVarsBaseBatch<F, NUM_HASH_OUT_ELTS>,
    ) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        let base = vars.local_wires[self.wire_base()];

        let power_bits: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_power_bit(i)])
            .collect();
        let intermediate_values: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_intermediate_value(i)])
            .collect();

        let output = vars.local_wires[self.wire_output()];

        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );

        let one = builder.one_extension();
        for i in 0..self.num_power_bits {
            let prev_intermediate_value = if i == 0 {
                one
            } else {
                builder.square_extension(intermediate_values[i - 1])
            };

            // power_bits is in LE order, but we accumulate in BE order.
            let cur_bit = power_bits[self.num_power_bits - i - 1];
            let mul_by = builder.select_ext_generalized(cur_bit, base, one);
            let intermediate_value_diff =
                builder.mul_sub_extension(prev_intermediate_value, mul_by, intermediate_values[i]);
            constraints.push(intermediate_value_diff);
        }

        let output_diff =
            builder.sub_extension(output, intermediate_values[self.num_power_bits - 1]);
        constraints.push(output_diff);

        constraints
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        let gen = ExponentiationGenerator::<F, D> {
            row,
            gate: self.clone(),
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        self.wire_intermediate_value(self.num_power_bits - 1) + 1
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        4
    }

    fn num_constraints(&self) -> usize {
        self.num_power_bits + 1
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    PackedEvaluableBase<F, D, NUM_HASH_OUT_ELTS> for ExponentiationGate<F, D>
where
    F::Extension: TwoAdicField,
{
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let base = vars.local_wires[self.wire_base()];

        let power_bits: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_power_bit(i)])
            .collect();
        let intermediate_values: Vec<_> = (0..self.num_power_bits)
            .map(|i| vars.local_wires[self.wire_intermediate_value(i)])
            .collect();

        let output = vars.local_wires[self.wire_output()];

        for i in 0..self.num_power_bits {
            let prev_intermediate_value = if i == 0 {
                P::one()
            } else {
                intermediate_values[i - 1].square()
            };

            // power_bits is in LE order, but we accumulate in BE order.
            let cur_bit = power_bits[self.num_power_bits - i - 1];

            let not_cur_bit = P::one() - cur_bit;
            let computed_intermediate_value =
                prev_intermediate_value * (cur_bit * base + not_cur_bit);
            yield_constr.one(computed_intermediate_value - intermediate_values[i]);
        }

        yield_constr.one(output - intermediate_values[self.num_power_bits - 1]);
    }
}

#[derive(Debug, Default)]
pub struct ExponentiationGenerator<F: RichField + HasExtension<D>, const D: usize>
where
    F::Extension: TwoAdicField,
{
    row: usize,
    gate: ExponentiationGate<F, D>,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for ExponentiationGenerator<F, D>
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        "ExponentiationGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let local_target = |column| Target::wire(self.row, column);

        let mut deps = Vec::with_capacity(self.gate.num_power_bits + 1);
        deps.push(local_target(self.gate.wire_base()));
        for i in 0..self.gate.num_power_bits {
            deps.push(local_target(self.gate.wire_power_bit(i)));
        }
        deps
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let get_local_wire = |column| witness.get_wire(local_wire(column));

        let num_power_bits = self.gate.num_power_bits;
        let base = get_local_wire(self.gate.wire_base());

        let power_bits = (0..num_power_bits)
            .map(|i| get_local_wire(self.gate.wire_power_bit(i)))
            .collect::<Vec<_>>();
        let mut intermediate_values = Vec::with_capacity(num_power_bits);

        let mut current_intermediate_value = F::one();
        for i in 0..num_power_bits {
            if power_bits[num_power_bits - i - 1] == F::one() {
                current_intermediate_value *= base;
            }
            intermediate_values.push(current_intermediate_value);
            current_intermediate_value *= current_intermediate_value;
        }

        for i in 0..num_power_bits {
            let intermediate_value_wire = local_wire(self.gate.wire_intermediate_value(i));
            out_buffer.set_wire(intermediate_value_wire, intermediate_values[i]);
        }

        let output_wire = local_wire(self.gate.wire_output());
        out_buffer.set_wire(output_wire, intermediate_values[num_power_bits - 1]);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)?;
        self.gate.serialize(dst, _common_data)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        let gate = ExponentiationGate::deserialize(src, _common_data)?;
        Ok(Self { row, gate })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_field::Field;
    use p3_goldilocks::Goldilocks;
    use rand::Rng;
    use rand::rngs::OsRng;

    use crate::field::types::Sample;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::{GOLDILOCKS_NUM_HASH_OUT_ELTS, HashOut};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::log2_ceil;

    use super::*;

    const MAX_POWER_BITS: usize = 17;

    #[test]
    fn wire_indices() {
        let gate = ExponentiationGate::<Goldilocks, 2> {
            num_power_bits: 5,
            _phantom: PhantomData,
        };

        assert_eq!(gate.wire_base(), 0);
        assert_eq!(gate.wire_power_bit(0), 1);
        assert_eq!(gate.wire_power_bit(4), 5);
        assert_eq!(gate.wire_output(), 6);
        assert_eq!(gate.wire_intermediate_value(0), 7);
        assert_eq!(gate.wire_intermediate_value(4), 11);
    }

    #[test]
    fn low_degree() {
        let config = CircuitConfig {
            num_wires: 120,
            num_routed_wires: 30,
            ..CircuitConfig::standard_recursion_config_gl()
        };

        test_low_degree::<Goldilocks, _, 2, 4>(ExponentiationGate::new_from_config(&config));
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(ExponentiationGate::new_from_config(
            &CircuitConfig::standard_recursion_config_gl(),
        ))
    }

    #[test]
    fn test_gate_constraint() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        type FF = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::FE;

        /// Returns the local wires for an exponentiation gate given the base, power, and power bit
        /// values.
        fn get_wires(base: F, power: u64) -> Vec<FF> {
            let mut power_bits = Vec::new();
            let mut cur_power = power;
            while cur_power > 0 {
                power_bits.push(cur_power % 2);
                cur_power /= 2;
            }

            let num_power_bits = power_bits.len();

            let power_bits_f: Vec<_> = power_bits
                .iter()
                .map(|b| F::from_canonical_u64(*b))
                .collect();

            let mut v = vec![base];
            v.extend(power_bits_f);

            let mut intermediate_values = Vec::new();
            let mut current_intermediate_value = F::one();
            for i in 0..num_power_bits {
                if power_bits[num_power_bits - i - 1] == 1 {
                    current_intermediate_value *= base;
                }
                intermediate_values.push(current_intermediate_value);
                current_intermediate_value *= current_intermediate_value;
            }
            let output_value = intermediate_values[num_power_bits - 1];
            v.push(output_value);
            v.extend(intermediate_values);

            v.iter().map(|&x| x.into()).collect::<Vec<_>>()
        }

        let mut rng = OsRng;

        let base = F::two();
        let power = rng.gen::<usize>() % (1 << MAX_POWER_BITS);
        let num_power_bits = log2_ceil(power + 1);
        let gate = ExponentiationGate::<F, D> {
            num_power_bits,
            _phantom: PhantomData,
        };

        let vars: EvaluationVars<'_, Goldilocks, 2, 4> = EvaluationVars {
            local_constants: &[],
            local_wires: &get_wires(base, power as u64),
            public_inputs_hash: &HashOut::rand(),
        };
        assert!(
            gate.eval_unfiltered(vars).iter().all(|x| x.is_zero()),
            "Gate constraints are not satisfied."
        );
    }
}
