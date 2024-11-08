#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use p3_field::AbstractField;

use plonky2_field::types::HasExtension;

use crate::gates::gate::Gate;
use crate::gates::poseidon_goldilocks_mds::PoseidonMdsGate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon_goldilocks::{
    HALF_N_FULL_ROUNDS, N_FULL_ROUNDS_TOTAL, N_PARTIAL_ROUNDS, PoseidonGoldilocks, SPONGE_WIDTH,
};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Evaluates a full Poseidon permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Debug, Default)]
pub struct PoseidonGate<F: RichField + HasExtension<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + HasExtension<D>, const D: usize> PoseidonGate<F, D>
where

{
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub(crate) const fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub(crate) const fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub(crate) const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    const fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_FULL_0: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    const fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize = Self::START_FULL_0 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < N_PARTIAL_ROUNDS);
        Self::START_PARTIAL + round
    }

    const START_FULL_1: usize = Self::START_PARTIAL + N_PARTIAL_ROUNDS;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_1 + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_FULL_1 + SPONGE_WIDTH * HALF_N_FULL_ROUNDS
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for PoseidonGate<F, D>
where

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
        Ok(PoseidonGate::new())
    }

    fn eval_unfiltered(
        &self,
        vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<<F as HasExtension<D>>::Extension> {
        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints
            .push(swap * (swap - <<F as HasExtension<D>>::Extension as AbstractField>::one()));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [<F as HasExtension<D>>::Extension::one(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_field::<F, F::Extension>(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            PoseidonGoldilocks::sbox_layer_field::<F, F::Extension>(&mut state);
            state = PoseidonGoldilocks::mds_layer_field(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        PoseidonGoldilocks::partial_first_constant_layer(&mut state);
        state = PoseidonGoldilocks::mds_partial_layer_init(&state);
        for r in 0..(N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = PoseidonGoldilocks::sbox_monomial(sbox_in);
            state[0] += <F as HasExtension<D>>::Extension::from_canonical_u64(
                PoseidonGoldilocks::FAST_PARTIAL_ROUND_CONSTANTS[r],
            );
            state = PoseidonGoldilocks::mds_partial_layer_fast_field(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(N_PARTIAL_ROUNDS - 1)];
        constraints.push(state[0] - sbox_in);
        state[0] = PoseidonGoldilocks::sbox_monomial(sbox_in);
        state = PoseidonGoldilocks::mds_partial_layer_fast_field(&state, N_PARTIAL_ROUNDS - 1);
        round_ctr += N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_field(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            PoseidonGoldilocks::sbox_layer_field::<F, F::Extension>(&mut state);
            state = PoseidonGoldilocks::mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * (swap - F::one()));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::zero(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            PoseidonGoldilocks::sbox_layer(&mut state);
            state = PoseidonGoldilocks::mds_layer(&state);
            round_ctr += 1;
        }

        // Partial rounds.
        PoseidonGoldilocks::partial_first_constant_layer(&mut state);
        state = PoseidonGoldilocks::mds_partial_layer_init(&state);
        for r in 0..(N_PARTIAL_ROUNDS - 1) {
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = PoseidonGoldilocks::sbox_monomial(sbox_in);
            state[0] += F::from_canonical_u64(PoseidonGoldilocks::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = PoseidonGoldilocks::mds_partial_layer_fast(&state, r);
        }
        let sbox_in = vars.local_wires[Self::wire_partial_sbox(N_PARTIAL_ROUNDS - 1)];
        yield_constr.one(state[0] - sbox_in);
        state[0] = PoseidonGoldilocks::sbox_monomial(sbox_in);
        state = PoseidonGoldilocks::mds_partial_layer_fast(&state, N_PARTIAL_ROUNDS - 1);
        round_ctr += N_PARTIAL_ROUNDS;

        // Second set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            PoseidonGoldilocks::sbox_layer(&mut state);
            state = PoseidonGoldilocks::mds_layer(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        // The naive method is more efficient if we have enough routed wires for PoseidonMdsGate.
        let use_mds_gate = builder.config.num_routed_wires
            >= <PoseidonMdsGate<F, D> as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_wires(
                &PoseidonMdsGate::<F, D>::new(),
            );

        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        let mut round_ctr = 0;

        // First set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_circuit(builder, &mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            PoseidonGoldilocks::sbox_layer_circuit(builder, &mut state);
            state = PoseidonGoldilocks::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        // Partial rounds.
        if use_mds_gate {
            for r in 0..N_PARTIAL_ROUNDS {
                PoseidonGoldilocks::constant_layer_circuit(builder, &mut state, round_ctr);
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = PoseidonGoldilocks::sbox_monomial_circuit(builder, sbox_in);
                state = PoseidonGoldilocks::mds_layer_circuit(builder, &state);
                round_ctr += 1;
            }
        } else {
            PoseidonGoldilocks::partial_first_constant_layer_circuit(builder, &mut state);
            state = PoseidonGoldilocks::mds_partial_layer_init_circuit(builder, &state);
            for r in 0..(N_PARTIAL_ROUNDS - 1) {
                let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = PoseidonGoldilocks::sbox_monomial_circuit(builder, sbox_in);
                let c = PoseidonGoldilocks::FAST_PARTIAL_ROUND_CONSTANTS[r];
                let c = <F as HasExtension<D>>::Extension::from_canonical_u64(c);
                let c = builder.constant_extension(c);
                state[0] = builder.add_extension(state[0], c);
                state = PoseidonGoldilocks::mds_partial_layer_fast_circuit(builder, &state, r);
            }
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(N_PARTIAL_ROUNDS - 1)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = PoseidonGoldilocks::sbox_monomial_circuit(builder, sbox_in);
            state = PoseidonGoldilocks::mds_partial_layer_fast_circuit(
                builder,
                &state,
                N_PARTIAL_ROUNDS - 1,
            );
            round_ctr += N_PARTIAL_ROUNDS;
        }

        // Second set of full rounds.
        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_circuit(builder, &mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            PoseidonGoldilocks::sbox_layer_circuit(builder, &mut state);
            state = PoseidonGoldilocks::mds_layer_circuit(builder, &state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        let gen = PoseidonGenerator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        SPONGE_WIDTH * (N_FULL_ROUNDS_TOTAL - 1) + N_PARTIAL_ROUNDS + SPONGE_WIDTH + 1 + 4
    }
}

#[derive(Debug, Default)]
pub struct PoseidonGenerator<F: RichField + HasExtension<D>, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for PoseidonGenerator<F, D>
where

{
    fn id(&self) -> String {
        "PoseidonGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| PoseidonGate::<F, D>::wire_input(i))
            .chain(Some(PoseidonGate::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| witness.get_wire(local_wire(PoseidonGate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(PoseidonGate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::zero() || swap_value == F::one());

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_delta(i)), delta_i);
        }

        if swap_value == F::one() {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();
        let mut round_ctr = 0;

        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_field(&mut state, round_ctr);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(PoseidonGate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    );
                }
            }
            PoseidonGoldilocks::sbox_layer_field::<F, F>(&mut state);
            state = PoseidonGoldilocks::mds_layer_field(&state);
            round_ctr += 1;
        }

        PoseidonGoldilocks::partial_first_constant_layer(&mut state);
        state = PoseidonGoldilocks::mds_partial_layer_init(&state);
        for r in 0..(N_PARTIAL_ROUNDS - 1) {
            out_buffer.set_wire(
                local_wire(PoseidonGate::<F, D>::wire_partial_sbox(r)),
                state[0],
            );
            state[0] = PoseidonGoldilocks::sbox_monomial(state[0]);
            state[0] += F::from_canonical_u64(PoseidonGoldilocks::FAST_PARTIAL_ROUND_CONSTANTS[r]);
            state = PoseidonGoldilocks::mds_partial_layer_fast_field(&state, r);
        }
        out_buffer.set_wire(
            local_wire(PoseidonGate::<F, D>::wire_partial_sbox(
                N_PARTIAL_ROUNDS - 1,
            )),
            state[0],
        );
        state[0] = PoseidonGoldilocks::sbox_monomial(state[0]);
        state = PoseidonGoldilocks::mds_partial_layer_fast_field(&state, N_PARTIAL_ROUNDS - 1);
        round_ctr += N_PARTIAL_ROUNDS;

        for r in 0..HALF_N_FULL_ROUNDS {
            PoseidonGoldilocks::constant_layer_field(&mut state, round_ctr);
            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(PoseidonGate::<F, D>::wire_full_sbox_1(r, i)),
                    state[i],
                );
            }
            PoseidonGoldilocks::sbox_layer_field::<F, F>(&mut state);
            state = PoseidonGoldilocks::mds_layer_field(&state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(PoseidonGate::<F, D>::wire_output(i)), state[i]);
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
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_goldilocks::Goldilocks;

    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::hash_types::GOLDILOCKS_NUM_HASH_OUT_ELTS;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    use super::*;

    #[test]
    fn wire_indices() {
        type F = Goldilocks;
        const D: usize = 2;
        type Gate = PoseidonGate<F, D>;

        assert_eq!(Gate::wire_input(0), 0);
        assert_eq!(Gate::wire_input(11), 11);
        assert_eq!(Gate::wire_output(0), 12);
        assert_eq!(Gate::wire_output(11), 23);
        assert_eq!(Gate::WIRE_SWAP, 24);
        assert_eq!(Gate::wire_delta(0), 25);
        assert_eq!(Gate::wire_delta(3), 28);
        assert_eq!(Gate::wire_full_sbox_0(1, 0), 29);
        assert_eq!(Gate::wire_full_sbox_0(3, 0), 53);
        assert_eq!(Gate::wire_full_sbox_0(3, 11), 64);
        assert_eq!(Gate::wire_partial_sbox(0), 65);
        assert_eq!(Gate::wire_partial_sbox(21), 86);
        assert_eq!(Gate::wire_full_sbox_1(0, 0), 87);
        assert_eq!(Gate::wire_full_sbox_1(3, 0), 123);
        assert_eq!(Gate::wire_full_sbox_1(3, 11), 134);
    }

    #[test]
    fn generated_output() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let config = CircuitConfig {
            num_wires: 143,
            ..CircuitConfig::standard_recursion_config_gl()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = PoseidonGate<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..SPONGE_WIDTH)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs.set_wire(
            Wire {
                row,
                column: Gate::WIRE_SWAP,
            },
            F::zero(),
        );
        for i in 0..SPONGE_WIDTH {
            inputs.set_wire(
                Wire {
                    row,
                    column: Gate::wire_input(i),
                },
                permutation_inputs[i],
            );
        }

        let witness = generate_partial_witness(inputs, &circuit.prover_only, &circuit.common);

        let expected_outputs: [F; SPONGE_WIDTH] =
            PoseidonGoldilocks::poseidon(permutation_inputs.try_into().unwrap());
        for i in 0..SPONGE_WIDTH {
            let out = witness.get_wire(Wire {
                row: 0,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn low_degree() {
        type F = Goldilocks;
        let gate = PoseidonGate::<F, 2>::new();
        test_low_degree::<F, PoseidonGate<F, 2>, 2, 4>(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = PoseidonGate::<F, 2>::new();
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }
}
