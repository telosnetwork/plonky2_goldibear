use plonky2_field::extension_field::Extendable;
use plonky2_util::ceil_div_usize;

use super::arithmetic_u32::U32Target;
use crate::gates::comparison::ComparisonGate;
use crate::hash::hash_types::RichField;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    /// Returns true if a is less than or equal to b, considered as base-`2^num_bits` limbs of a large value.
    /// This range-checks its inputs.
    pub fn list_le(&mut self, a: Vec<Target>, b: Vec<Target>, num_bits: usize) -> BoolTarget {
        assert_eq!(
            a.len(),
            b.len(),
            "Comparison must be between same number of inputs and outputs"
        );
        let n = a.len();

        let chunk_bits = 2;
        let num_chunks = ceil_div_usize(num_bits, chunk_bits);

        let one = self.one();
        let mut result = one;
        for i in 0..n {
            let a_le_b_gate = ComparisonGate::new(num_bits, num_chunks);
            let a_le_b_gate_index = self.add_gate(a_le_b_gate.clone(), vec![]);
            self.connect(
                Target::wire(a_le_b_gate_index, a_le_b_gate.wire_first_input()),
                a[i],
            );
            self.connect(
                Target::wire(a_le_b_gate_index, a_le_b_gate.wire_second_input()),
                b[i],
            );
            let a_le_b_result = Target::wire(a_le_b_gate_index, a_le_b_gate.wire_result_bool());

            let b_le_a_gate = ComparisonGate::new(num_bits, num_chunks);
            let b_le_a_gate_index = self.add_gate(b_le_a_gate.clone(), vec![]);
            self.connect(
                Target::wire(b_le_a_gate_index, b_le_a_gate.wire_first_input()),
                b[i],
            );
            self.connect(
                Target::wire(b_le_a_gate_index, b_le_a_gate.wire_second_input()),
                a[i],
            );
            let b_le_a_result = Target::wire(b_le_a_gate_index, b_le_a_gate.wire_result_bool());

            let these_limbs_equal = self.mul(a_le_b_result, b_le_a_result);
            let these_limbs_less_than = self.sub(one, b_le_a_result);
            result = self.mul_add(these_limbs_equal, result, these_limbs_less_than);
        }

        // `result` being boolean is an invariant, maintained because its new value is always
        // `x * result + y`, where `x` and `y` are booleans that are not simultaneously true.
        BoolTarget::new_unsafe(result)
    }

    /// Helper function for comparing, specifically, lists of `U32Target`s.
    pub fn list_le_binary<const BITS: usize>(&mut self, a: Vec<BinaryTarget<BITS>>, b: Vec<BinaryTarget<BITS>>) -> BoolTarget {
        let a_targets = a.iter().map(|&t| t.0).collect();
        let b_targets = b.iter().map(|&t| t.0).collect();
        self.list_le(a_targets, b_targets, BITS)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use num::BigUint;
    use plonky2_field::field_types::Field;
    use rand::Rng;

    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    fn test_list_le(size: usize, num_bits: usize) -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let config = CircuitConfig::standard_recursion_config();
        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D>::new(config);

        let mut rng = rand::thread_rng();

        let lst1: Vec<u64> = (0..size)
            .map(|_| rng.gen_range(0..(1 << num_bits)))
            .collect();
        let lst2: Vec<u64> = (0..size)
            .map(|_| rng.gen_range(0..(1 << num_bits)))
            .collect();

        let a_biguint = BigUint::from_slice(
            &lst1
                .iter()
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect::<Vec<_>>(),
        );
        let b_biguint = BigUint::from_slice(
            &lst2
                .iter()
                .flat_map(|&x| [x as u32, (x >> 32) as u32])
                .collect::<Vec<_>>(),
        );

        let a = lst1
            .iter()
            .map(|&x| builder.constant(F::from_canonical_u64(x)))
            .collect();
        let b = lst2
            .iter()
            .map(|&x| builder.constant(F::from_canonical_u64(x)))
            .collect();

        let result = builder.list_le(a, b, num_bits);

        let expected_result = builder.constant_bool(a_biguint <= b_biguint);
        builder.connect(result.target, expected_result.target);

        let data = builder.build::<C>();
        let proof = data.prove(pw).unwrap();
        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_multiple_comparison() -> Result<()> {
        for size in [1, 3, 6] {
            for num_bits in [20, 32, 40, 44] {
                test_list_le(size, num_bits).unwrap();
            }
        }

        Ok(())
    }
}
