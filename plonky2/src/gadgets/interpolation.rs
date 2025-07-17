#[cfg(not(feature = "std"))]
use alloc::vec;

use plonky2_field::types::HasExtension;

use crate::gates::coset_interpolation::CosetInterpolationGate;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>
{
    /// Interpolates a polynomial, whose points are a coset of the multiplicative subgroup with the
    /// given size, and whose values are given. Returns the evaluation of the interpolant at
    /// `evaluation_point`.
    pub(crate) fn interpolate_coset(
        &mut self,
        gate: CosetInterpolationGate<F, D>,
        coset_shift: Target,
        values: &[ExtensionTarget<D>],
        evaluation_point: ExtensionTarget<D>,
    ) -> ExtensionTarget<D> {
        let row = self.num_gates();
        self.connect(coset_shift, Target::wire(row, gate.wire_shift()));
        for (i, &v) in values.iter().enumerate() {
            self.connect_extension(v, ExtensionTarget::from_range(row, gate.wires_value(i)));
        }
        self.connect_extension(
            evaluation_point,
            ExtensionTarget::from_range(row, gate.wires_evaluation_point()),
        );

        let eval = ExtensionTarget::from_range(row, gate.wires_evaluation_value());
        self.add_gate(gate, vec![]);

        eval
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use anyhow::Result;
    use p3_field::{cyclic_subgroup_coset_known_order, AbstractExtensionField, TwoAdicField};
    use plonky2_field::types::HasExtension;
    use plonky2_field::{GOLDILOCKS_EXTENSION_FIELD_DEGREE, GOLDILOCKS_NUM_HASH_OUT_ELTS};

    use crate::field::interpolation::interpolant;
    use crate::field::types::Sample;
    use crate::gates::coset_interpolation::CosetInterpolationGate;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    #[test]
    fn test_interpolate() -> Result<()> {
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        type FF = <F as HasExtension<D>>::Extension;
        let config = CircuitConfig::standard_recursion_config_gl();
        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);

        let subgroup_bits = 2;
        let len = 1 << subgroup_bits;
        let coset_shift = F::rand();
        let g = <F as TwoAdicField>::two_adic_generator(subgroup_bits);
        let points = cyclic_subgroup_coset_known_order::<F>(g, coset_shift, len);
        let values = FF::rand_vec(len);

        let homogeneous_points = points
            .zip(values.iter())
            .map(|(a, &b)| (<FF as AbstractExtensionField<F>>::from_base(a), b))
            .collect::<Vec<_>>();

        let true_interpolant = interpolant(&homogeneous_points);

        let z = FF::rand();
        let true_eval = true_interpolant.eval(z);

        let coset_shift_target = builder.constant(coset_shift);

        let value_targets = values
            .iter()
            .map(|&v| builder.constant_extension(v))
            .collect::<Vec<_>>();

        let zt = builder.constant_extension(z);

        let evals_coset_gates = (2..=4)
            .map(|max_degree| {
                builder.interpolate_coset(
                    CosetInterpolationGate::with_max_degree(subgroup_bits, max_degree),
                    coset_shift_target,
                    &value_targets,
                    zt,
                )
            })
            .collect::<Vec<_>>();
        let true_eval_target = builder.constant_extension(true_eval);
        for &eval_coset_gate in evals_coset_gates.iter() {
            builder.connect_extension(eval_coset_gate, true_eval_target);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }
}
