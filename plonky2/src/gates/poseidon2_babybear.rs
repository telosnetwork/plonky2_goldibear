#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;
use core::usize;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64, TwoAdicField};
use plonky2_field::types::HasExtension;

use crate::{gates::gate::Gate, hash::poseidon2_babybear::Poseidon2BabyBearHash, iop::target::BoolTarget};
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon2_babybear::{
    EXTERNAL_CONSTANTS, HALF_N_FULL_ROUNDS, INTERNAL_CONSTANTS, N_FULL_ROUNDS_TOTAL,
    N_PARTIAL_ROUNDS, SPONGE_CAPACITY, SPONGE_WIDTH,
};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

use super::gate::GateRef;

const SBOX_EXP: u64 = 7;
pub(crate) const INTERNAL_DIAG_SHIFTS: [usize; SPONGE_WIDTH - 1] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15];
/// Evaluates a full Poseidon permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Clone, Debug, Default)]
pub struct Poseidon2BabyBearGate<F: RichField + HasExtension<D>, const D: usize> {
    num_ops: usize,
    _phantom: PhantomData<F>,
}

const ROUTED_WIRES_PER_OP: usize = 2 * SPONGE_WIDTH + 1;
const NON_ROUTED_WIRES_PER_OP: usize =
    SPONGE_CAPACITY + SPONGE_WIDTH * (N_FULL_ROUNDS_TOTAL - 1) + N_PARTIAL_ROUNDS;

impl<F: RichField + HasExtension<D>, const D: usize> Poseidon2BabyBearGate<F, D>
where
    F::Extension: TwoAdicField,
{
    pub fn new() -> Self {
        Self::new_from_config(&CircuitConfig::standard_recursion_config_bb_wide())
    }

    pub fn new_from_config(config: &CircuitConfig) -> Self {
        if BabyBear::ORDER_U64 != F::ORDER_U64 {
            panic!("The Poseidon2 BabyBear gate can be used only for the BabyBear field!")
        }
        let wires_per_op = ROUTED_WIRES_PER_OP + NON_ROUTED_WIRES_PER_OP;
        let num_ops =
            (config.num_wires / wires_per_op).min(config.num_routed_wires / ROUTED_WIRES_PER_OP);
        Self {
            num_ops,
            _phantom: PhantomData,
        }
    }
    /***************** START ROUTED WIRES ***********************/
    /// The wire index for the `i`th input to the permutation.
    pub(crate) const fn wire_input(op: usize, i: usize) -> usize {
        ROUTED_WIRES_PER_OP * op + i
    }

    /// The wire index for the `i`th output to the permutation.
    pub(crate) const fn wire_output(op: usize, i: usize) -> usize {
        ROUTED_WIRES_PER_OP * op + SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub(crate) const fn wire_swap(op: usize) -> usize {
        ROUTED_WIRES_PER_OP * op + 2 * SPONGE_WIDTH
    }

    /************** *******************/

    const fn start_delta(&self, op: usize) -> usize {
        self.num_ops * ROUTED_WIRES_PER_OP + op * NON_ROUTED_WIRES_PER_OP
    }

    /// A wire which stores `swap * (input[i + SPONGE_CAPACITY] - input[i])`; used to compute the swapped inputs.
    const fn wire_delta(&self, op: usize, i: usize) -> usize {
        assert!(i < SPONGE_CAPACITY);
        self.start_delta(op) + i
    }

    const fn start_full_0(&self, op: usize) -> usize {
        self.start_delta(op) + SPONGE_CAPACITY
    }

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    const fn wire_full_sbox_0(&self, op: usize, round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        self.start_full_0(op)  + SPONGE_WIDTH * (round - 1) + i
    }

    const fn start_partial(&self, op: usize) -> usize {self.start_full_0(op)  + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1)}

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    const fn wire_partial_sbox(&self, op: usize, round: usize) -> usize {
        debug_assert!(round < N_PARTIAL_ROUNDS);
        self.start_partial(op) + round
    }

    const fn start_full_1(&self, op: usize) -> usize {self.start_partial(op) + N_PARTIAL_ROUNDS}

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    const fn wire_full_sbox_1(&self, op: usize, round: usize, i: usize) -> usize {
        debug_assert!(round < HALF_N_FULL_ROUNDS);
        debug_assert!(i < SPONGE_WIDTH);
        self.start_full_1(op) + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    pub(crate) const fn end(&self, op: usize) -> usize {
        self.start_full_1(op) + SPONGE_WIDTH * HALF_N_FULL_ROUNDS
    }
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    Gate<F, D, NUM_HASH_OUT_ELTS> for Poseidon2BabyBearGate<F, D>
where
    F::Extension: TwoAdicField,
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
        Ok(Poseidon2BabyBearGate::new())
    }

    fn finalize(&self, builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>) {
        let gate_ref: GateRef<F, D, NUM_HASH_OUT_ELTS> = GateRef::new(self.clone());
        let gate_slot = builder.current_slots.entry(gate_ref.clone()).or_default();
        let slot = gate_slot.current_slot.get::<[F]>(&[]);
        if let Some(&(gate_idx, mut slot_idx)) = slot {
            let zero = builder.zero();
            while slot_idx < self.num_ops -1 {
                slot_idx += 1;
                builder.add_simple_generator(Poseidon2BabyBearGenerator::<F,D> {row: gate_idx, op: slot_idx, _phantom: PhantomData})
            }
        }
    }

    fn eval_unfiltered(
        &self,
        vars: EvaluationVars<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<<F as HasExtension<D>>::Extension> {
        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );
        for op in 0..self.num_ops {
            // Assert that `swap` is binary.
            let swap = vars.local_wires[Self::wire_swap(op)];
            constraints
                .push(swap * (swap - <<F as HasExtension<D>>::Extension as AbstractField>::one()));

            // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
            for i in 0..SPONGE_CAPACITY {
                let input_lhs = vars.local_wires[Self::wire_input(op, i)];
                let input_rhs = vars.local_wires[Self::wire_input(op, i + SPONGE_CAPACITY)];
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                constraints.push(swap * (input_rhs - input_lhs) - delta_i);
            }

            // Compute the possibly-swapped input layer.
            let mut state = [<F as HasExtension<D>>::Extension::one(); SPONGE_WIDTH];
            for i in 0..SPONGE_CAPACITY {
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                let input_lhs = Self::wire_input(op, i);
                let input_rhs = Self::wire_input(op, i + SPONGE_CAPACITY);
                state[i] = vars.local_wires[input_lhs] + delta_i;
                state[i + SPONGE_CAPACITY] = vars.local_wires[input_rhs] - delta_i;
            }
            for i in 2 * SPONGE_CAPACITY..SPONGE_WIDTH {
                state[i] = vars.local_wires[Self::wire_input(op, i)];
            }
            permute_external_mut(&mut state);

            for r in 0..HALF_N_FULL_ROUNDS {
                add_rc(&mut state, r);
                if r > 0 {
                    for i in 0..SPONGE_WIDTH {
                        let sbox_in = vars.local_wires[self.wire_full_sbox_0(op, r, i)];
                        constraints.push(state[i] - sbox_in);
                        state[i] = sbox_in;
                    }
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());
                permute_external_mut(&mut state);
            }

            // The internal rounds.
            // for r in 0..self.rounds_p {
            //     state[0] += AF::from_f(self.internal_constants[r]);
            //     state[0] = self.sbox_p(&state[0]);
            //     self.internal_linear_layer.permute_mut(state);
            // }
            for r in 0..N_PARTIAL_ROUNDS {
                state[0] += F::Extension::from_canonical_u32(INTERNAL_CONSTANTS[r]);
                let sbox_in = vars.local_wires[self.wire_partial_sbox(op, r)];
                constraints.push(state[0] - sbox_in);
                state[0] = sbox_in.exp_const_u64::<SBOX_EXP>();
                permute_internal_mut(&mut state);
            }

            // Second set of full rounds.
            // The second half of the external rounds.
            // for r in rounds_f_half..self.rounds_f {
            //     self.add_rc(state, &self.external_constants[r]);
            //     self.sbox(state);
            //     self.external_linear_layer.permute_mut(state);
            // }
            for r in HALF_N_FULL_ROUNDS..N_FULL_ROUNDS_TOTAL {
                add_rc(&mut state, r);
                for i in 0..SPONGE_WIDTH {
                    let sbox_in =
                        vars.local_wires[self.wire_full_sbox_1(op, r - HALF_N_FULL_ROUNDS, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());
                permute_external_mut(&mut state);
            }

            for i in 0..SPONGE_WIDTH {
                constraints.push(state[i] - vars.local_wires[Self::wire_output(op, i)]);
            }
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F, NUM_HASH_OUT_ELTS>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        for op in 0..self.num_ops {
            // Assert that `swap` is binary.
            let swap = vars.local_wires[Self::wire_swap(op)];
            yield_constr.one(swap * (swap - <F as AbstractField>::one()));

            // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
            for i in 0..SPONGE_CAPACITY {
                let input_lhs = vars.local_wires[Self::wire_input(op, i)];
                let input_rhs = vars.local_wires[Self::wire_input(op, i + SPONGE_CAPACITY)];
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
            }

            // Compute the possibly-swapped input layer.
            let mut state = [F::one(); SPONGE_WIDTH];
            for i in 0..SPONGE_CAPACITY {
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                let input_lhs = Self::wire_input(op, i);
                let input_rhs = Self::wire_input(op, i + SPONGE_CAPACITY);
                state[i] = vars.local_wires[input_lhs] + delta_i;
                state[i + SPONGE_CAPACITY] = vars.local_wires[input_rhs] - delta_i;
            }
            for i in 2 * SPONGE_CAPACITY..SPONGE_WIDTH {
                state[i] = vars.local_wires[Self::wire_input(op, i)];
            }

            permute_external_mut(&mut state);

            for r in 0..HALF_N_FULL_ROUNDS {
                add_rc(&mut state, r);
                if r > 0 {
                    for i in 0..SPONGE_WIDTH {
                        let sbox_in = vars.local_wires[self.wire_full_sbox_0(op, r, i)];
                        yield_constr.one(state[i] - sbox_in);
                        state[i] = sbox_in;
                    }
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());
                permute_external_mut(&mut state);
            }

            for r in 0..N_PARTIAL_ROUNDS {
                state[0] += F::from_canonical_u32(INTERNAL_CONSTANTS[r]);
                let sbox_in = vars.local_wires[self.wire_partial_sbox(op, r)];
                yield_constr.one(state[0] - sbox_in);
                state[0] = sbox_in.exp_const_u64::<SBOX_EXP>();
                permute_internal_mut(&mut state);
            }

            // Second set of full rounds.
            // The second half of the external rounds.
            for r in HALF_N_FULL_ROUNDS..N_FULL_ROUNDS_TOTAL {
                add_rc(&mut state, r);
                for i in 0..SPONGE_WIDTH {
                    let sbox_in =
                        vars.local_wires[self.wire_full_sbox_1(op, r - HALF_N_FULL_ROUNDS, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());
                permute_external_mut(&mut state);
            }

            for i in 0..SPONGE_WIDTH {
                yield_constr.one(state[i] - vars.local_wires[Self::wire_output(op, i)]);
            }
        }
    }
    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        vars: EvaluationTargets<D, NUM_HASH_OUT_ELTS>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(
            <Self as Gate<F, D, NUM_HASH_OUT_ELTS>>::num_constraints(&self),
        );
        for op in 0..self.num_ops {
            // Assert that `swap` is binary.
            let swap = vars.local_wires[Self::wire_swap(op)];
            constraints.push(builder.mul_sub_extension(swap, swap, swap));

            // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
            for i in 0..SPONGE_CAPACITY {
                let input_lhs = vars.local_wires[Self::wire_input(op, i)];
                let input_rhs = vars.local_wires[Self::wire_input(op, i + SPONGE_CAPACITY)];
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                let diff = builder.sub_extension(input_rhs, input_lhs);
                constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
            }

            // Compute the possibly-swapped input layer.
            let one = builder.one_extension();
            let mut state = [one; SPONGE_WIDTH];
            for i in 0..SPONGE_CAPACITY {
                let delta_i = vars.local_wires[self.wire_delta(op, i)];
                let input_lhs = Self::wire_input(op, i);
                let input_rhs = Self::wire_input(op, i + SPONGE_CAPACITY);
                state[i] = builder.add_extension(vars.local_wires[input_lhs], delta_i);
                state[i + SPONGE_CAPACITY] =
                    builder.sub_extension(vars.local_wires[input_rhs], delta_i);
            }
            for i in 2 * SPONGE_CAPACITY..SPONGE_WIDTH {
                state[i] = vars.local_wires[Self::wire_input(op, i)];
            }
            permute_external_mut_circuit(builder, &mut state);

            // First set of full rounds.
            for r in 0..HALF_N_FULL_ROUNDS {
                add_rc_circuit(builder, &mut state, r);
                if r > 0 {
                    for i in 0..SPONGE_WIDTH {
                        let sbox_in = vars.local_wires[self.wire_full_sbox_0(op, r, i)];
                        constraints.push(builder.sub_extension(state[i], sbox_in));
                        state[i] = sbox_in;
                    }
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = sbox_circuit(builder, state[i]));
                permute_external_mut_circuit(builder, &mut state);
            }

            // The internal rounds.
            for r in 0..N_PARTIAL_ROUNDS {
                state[0] = builder
                    .add_const_extension(state[0], F::from_canonical_u32(INTERNAL_CONSTANTS[r]));
                let sbox_in = vars.local_wires[self.wire_partial_sbox(op, r)];
                constraints.push(builder.sub_extension(state[0], sbox_in));
                state[0] = sbox_circuit(builder, sbox_in);
                permute_internal_mut_circuit(builder, &mut state);
            }

            // Second set of full rounds.
            // The second half of the external rounds.
            for r in HALF_N_FULL_ROUNDS..N_FULL_ROUNDS_TOTAL {
                add_rc_circuit(builder, &mut state, r);
                for i in 0..SPONGE_WIDTH {
                    let sbox_in =
                        vars.local_wires[self.wire_full_sbox_1(op, r - HALF_N_FULL_ROUNDS, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
                (0..SPONGE_WIDTH).for_each(|i| state[i] = sbox_circuit(builder, state[i]));
                permute_external_mut_circuit(builder, &mut state);
            }

            for i in 0..SPONGE_WIDTH {
                constraints.push(
                    builder.sub_extension(state[i], vars.local_wires[Self::wire_output(op, i)]),
                );
            }
        }

        constraints
    }

    fn generators(
        &self,
        row: usize,
        _local_constants: &[F],
    ) -> Vec<WitnessGeneratorRef<F, D, NUM_HASH_OUT_ELTS>> {
        (0..self.num_ops)
            .map(|op| {
                WitnessGeneratorRef::new(
                    Poseidon2BabyBearGenerator::<F, D> {
                        row,
                        op,
                        _phantom: PhantomData,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.end(self.num_ops - 1)
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        self.num_ops * (SPONGE_WIDTH * (N_FULL_ROUNDS_TOTAL - 1)
            + N_PARTIAL_ROUNDS
            + SPONGE_WIDTH
            + 1
            + SPONGE_CAPACITY)
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2BabyBearGenerator<F: RichField + HasExtension<D>, const D: usize> {
    row: usize,
    op: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for Poseidon2BabyBearGenerator<F, D>
where
    F::Extension: TwoAdicField,
{
    fn id(&self) -> String {
        "PoseidonGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| Poseidon2BabyBearGate::<F, D>::wire_input(self.op, i))
            .chain(Some(Poseidon2BabyBearGate::<F, D>::wire_swap(self.op)))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| {
                witness.get_wire(local_wire(Poseidon2BabyBearGate::<F, D>::wire_input(
                    self.op, i,
                )))
            })
            .collect::<Vec<_>>();
        let swap_value = witness.get_wire(local_wire(Poseidon2BabyBearGate::<F, D>::wire_swap(
            self.op,
        )));
        debug_assert!(swap_value == F::zero() || swap_value == F::one());

        let gate = Poseidon2BabyBearGate::<F, D>::new();
        for i in 0..SPONGE_CAPACITY {
            let delta_i = swap_value * (state[i + SPONGE_CAPACITY] - state[i]);
            out_buffer.set_wire(
                local_wire(gate.wire_delta(self.op, i)),
                delta_i,
            );
        }

        if swap_value == F::one() {
            for i in 0..SPONGE_CAPACITY {
                state.swap(i, SPONGE_CAPACITY + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();

        permute_external_mut(&mut state);
        let mut round_ctr = 0;

        for r in 0..HALF_N_FULL_ROUNDS {
            add_rc(&mut state, round_ctr);

            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(gate.wire_full_sbox_0(self.op, r, i)),
                        state[i],
                    );
                }
            }
            (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());

            permute_external_mut(&mut state);

            round_ctr += 1;
        }

        for r in 0..N_PARTIAL_ROUNDS {
            round_ctr += 1;
            state[0] += F::from_canonical_u32(INTERNAL_CONSTANTS[r]);

            out_buffer.set_wire(
                local_wire(gate.wire_partial_sbox(self.op, r)),
                state[0],
            );
            state[0] = state[0].exp_const_u64::<SBOX_EXP>();
            permute_internal_mut(&mut state);
        }

        for r in HALF_N_FULL_ROUNDS..N_FULL_ROUNDS_TOTAL {
            add_rc(&mut state, r);

            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(gate.wire_full_sbox_1(
                        self.op,
                        r - HALF_N_FULL_ROUNDS,
                        i,
                    )),
                    state[i],
                )
            }
            (0..SPONGE_WIDTH).for_each(|i| state[i] = state[i].exp_const_u64::<SBOX_EXP>());

            permute_external_mut(&mut state);

            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(
                local_wire(Poseidon2BabyBearGate::<F, D>::wire_output(self.op, i)),
                state[i],
            );
        }
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.op)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let row = src.read_usize()?;
        let op = src.read_usize()?;
        Ok(Self {
            row,
            op,
            _phantom: PhantomData,
        })
    }
}

fn sbox_circuit<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    input: ExtensionTarget<D>,
) -> ExtensionTarget<D>
where
    F::Extension: TwoAdicField,
{
    let x2 = builder.square_extension(input);
    let x3 = builder.mul_extension(input, x2);
    let x4 = builder.square_extension(x2);
    builder.mul_extension(x3, x4)
}

fn add_rc_circuit<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    round_idx: usize,
) where
    F::Extension: TwoAdicField,
{
    (0..SPONGE_WIDTH).for_each(|i| {
        state[i] = builder.add_const_extension(
            state[i],
            F::from_canonical_u32(EXTERNAL_CONSTANTS[round_idx][i]),
        )
    })
}

fn add_rc<F: AbstractField>(state: &mut [F; SPONGE_WIDTH], round_idx: usize) {
    assert!(round_idx < N_FULL_ROUNDS_TOTAL);
    (0..SPONGE_WIDTH)
        .for_each(|i| state[i] += F::from_canonical_u32(EXTERNAL_CONSTANTS[round_idx][i]))
}

fn permute_internal_mut_circuit<
    F: RichField + HasExtension<D>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
) where
    F::Extension: TwoAdicField,
{
    // let gate = Poseidon2InternalPermutationGate::<F, D>::new();
    // let row = builder.add_gate(gate, vec![]);
    // (0..SPONGE_WIDTH).for_each(|i| {
    //     builder.connect_extension(
    //         state[i],
    //         ExtensionTarget::<D>::from_range(row, Poseidon2InternalPermutationGate::<F,D>::wires_input(i))
    //     )
    // });
    // *state = (0..SPONGE_WIDTH).map(|i| {
    //     ExtensionTarget::<D>::from_range(row, Poseidon2InternalPermutationGate::<F,D>::wires_output(i))
    // }).collect_vec().try_into().unwrap();
    state
        .iter_mut()
        .for_each(|x| *x = builder.mul_const_extension(F::from_canonical_u32(943718400), *x));
    let part_sum = builder.add_many_extension(state[1..].into_iter());
    let full_sum = builder.add_extension(part_sum, state[0]);
    state[0] = builder.sub_extension(part_sum, state[0]);
    (0..SPONGE_WIDTH - 1).for_each(|i| {
        let shift = F::from_canonical_usize(1 << INTERNAL_DIAG_SHIFTS[i]);
        state[i + 1] = builder.mul_const_add_extension(shift, state[i + 1], full_sum);
    });
}

// /// Implements multiplication by the diffusion matrix 1 + Diag(vec) using a delayed reduction strategy.
// fn permute_state(state: &mut [MontyField31<FP>; WIDTH]) {
//     let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
//     let full_sum = part_sum + (state[0].value as u64);
//     let s0 = part_sum + (-state[0]).value as u64;
//     state[0] = MontyField31::new_monty(monty_reduce::<FP>(s0));

//     for i in 0..Self::INTERNAL_DIAG_SHIFTS.as_ref().len() {
//         let si =
//             full_sum + ((state[i + 1].value as u64) << Self::INTERNAL_DIAG_SHIFTS.as_ref()[i]);
//         state[i + 1] = MontyField31::new_monty(monty_reduce::<FP>(si));
//     }
// }
fn permute_internal_mut<AF: AbstractField>(state: &mut [AF; SPONGE_WIDTH]) {
    state
        .iter_mut()
        .for_each(|x| *x *= AF::from_canonical_u32(943718400));
    let part_sum: AF = state
        .iter()
        .skip(1)
        .fold(AF::zero(), |acc, x| acc + x.clone());
    let full_sum = part_sum.clone() + state[0].clone();
    state[0] = part_sum.clone() - state[0].clone();

    for i in 0..INTERNAL_DIAG_SHIFTS.len() {
        state[i + 1] = full_sum.clone()
            + state[i + 1].clone() * AF::from_canonical_u32(1 << INTERNAL_DIAG_SHIFTS[i]);
    }
}

fn permute_external_mut<AF: AbstractField, const WIDTH: usize>(state: &mut [AF; WIDTH]) {
    assert_eq!(WIDTH % 4, 0);
    for i in (0..WIDTH).step_by(4) {
        // Would be nice to find a better way to do this.
        let mut state_4 = [
            state[i].clone(),
            state[i + 1].clone(),
            state[i + 2].clone(),
            state[i + 3].clone(),
        ];
        apply_mat4(&mut state_4);
        state[i..i + 4].clone_from_slice(&state_4);
    }
    // Now, we apply the outer circulant matrix (to compute the y_i values).

    // We first precompute the four sums of every four elements.
    let sums: [AF; 4] = core::array::from_fn(|k| {
        (0..WIDTH)
            .step_by(4)
            .map(|j| state[j + k].clone())
            .sum::<AF>()
    });

    // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
    // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
    for i in 0..WIDTH {
        state[i] += sums[i % 4].clone();
    }
}

fn permute_external_mut_circuit<
    F: RichField + HasExtension<D>,
    const D: usize,
    const WIDTH: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    state: &mut [ExtensionTarget<D>; WIDTH],
) where
    F::Extension: TwoAdicField,
{
    assert_eq!(WIDTH % 4, 0);
    for i in (0..WIDTH).step_by(4) {
        // Would be nice to find a better way to do this.
        let mut state_4 = [
            state[i].clone(),
            state[i + 1].clone(),
            state[i + 2].clone(),
            state[i + 3].clone(),
        ];
        apply_mat4_circuit(builder, &mut state_4);
        state[i..i + 4].clone_from_slice(&state_4);
    }
    // Now, we apply the outer circulant matrix (to compute the y_i values).

    // We first precompute the four sums of every four elements.
    let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
        builder.add_many_extension((0..WIDTH).step_by(4).map(|j| state[j + k].clone()))
    });

    // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
    // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
    for i in 0..WIDTH {
        state[i] = builder.add_extension(state[i], sums[i % 4]);
    }
}

fn apply_mat4_circuit<
    F: RichField + HasExtension<D>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    x: &mut [ExtensionTarget<D>; 4],
) where
    F::Extension: TwoAdicField,
{
    // let gate = ApplyMat4Gate::<F, D>::new_from_config(&builder.config);
    // let (row, op) = builder.find_slot(gate, &[], &[]);
    // (0..4).for_each(|i| {
    //     builder.connect_extension(
    //         x[i],
    //         ExtensionTarget::<D>::from_range(row, ApplyMat4Gate::<F,D>::wires_input(op, i))
    //     )
    // });
    // *x = [0, 1, 2, 3].map(|i| {
    //     ExtensionTarget::<D>::from_range(row, ApplyMat4Gate::<F,D>::wires_output(op, i))
    // });
    let t01 = builder.add_extension(x[0], x[1]); //x[0].clone() + x[1].clone();
    let t23 = builder.add_extension(x[2], x[3]); //x[2].clone() + x[3].clone();
    let t0123 = builder.add_extension(t01, t23); //t01.clone() + t23.clone();
    let t01123 = builder.add_extension(t0123, x[1]); //t0123.clone() + x[1].clone();
    let t01233 = builder.add_extension(t0123, x[3]); //t0123.clone() + x[3].clone();
    x[3] = builder.mul_const_add_extension(F::two(), x[0], t01233); //t01233.clone() + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = builder.mul_const_add_extension(F::two(), x[2], t01123); //t01123.clone() + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = builder.add_extension(t01123, t01); //t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = builder.add_extension(t01233, t23); //t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

fn apply_mat4<AF>(x: &mut [AF; 4])
where
    AF: AbstractField,
{
    let t01 = x[0].clone() + x[1].clone();
    let t23 = x[2].clone() + x[3].clone();
    let t0123 = t01.clone() + t23.clone();
    let t01123 = t0123.clone() + x[1].clone();
    let t01233 = t0123.clone() + x[3].clone();
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    x[3] = t01233.clone() + x[0].double(); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = t01123.clone() + x[2].double(); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = t01123 + t01; // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = t01233 + t23; // x[0] + x[1] + 2*x[2] + 3*x[3]
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_baby_bear::BabyBear;
    use plonky2_field::types::Sample;

    use super::*;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::poseidon2_babybear::Permuter31;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, Poseidon2BabyBearConfig};

    #[test]
    fn wire_indices() {
        // type F = BabyBear;
        // const D: usize = 4;
        // type Gate = Poseidon2BabyBearGate<F, D>;

        // assert_eq!(Gate::wire_input(0), 0);
        // assert_eq!(Gate::wire_input(23), 23);
        // assert_eq!(Gate::wire_output(0), 24);
        // assert_eq!(Gate::wire_output(23), 47);
        // assert_eq!(Gate::wire_swap(op), 48);
        // assert_eq!(Gate::wire_delta(0), 49);
        // assert_eq!(Gate::wire_delta(7), 56);
        // assert_eq!(Gate::wire_full_sbox_0(1, 0), 57);
        // assert_eq!(Gate::wire_full_sbox_0(3, 0), 105);
        // assert_eq!(Gate::wire_full_sbox_0(3, 23), 128);
        // assert_eq!(Gate::wire_partial_sbox(0), 129);
        // assert_eq!(Gate::wire_partial_sbox(20), 149);
        // assert_eq!(Gate::wire_full_sbox_1(0, 0), 150);
        // assert_eq!(Gate::wire_full_sbox_1(3, 0), 222);
        // assert_eq!(Gate::wire_full_sbox_1(3, 23), 245);
    }

    #[test]
    fn generated_output() {
        const D: usize = 4;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = 8;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let config = CircuitConfig::standard_recursion_config_bb_wide();
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2BabyBearGate<F, D>;
        let gate = Gate::new();
        let (row, op) = builder.find_slot(gate, &[], &[]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..SPONGE_WIDTH)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs.set_wire(
            Wire {
                row,
                column: Gate::wire_swap(op),
            },
            F::zero(),
        );
        for i in 0..SPONGE_WIDTH {
            inputs.set_wire(
                Wire {
                    row,
                    column: Gate::wire_input(op, i),
                },
                permutation_inputs[i],
            );
        }

        let witness = generate_partial_witness(inputs, &circuit.prover_only, &circuit.common);

        let expected_outputs: [F; SPONGE_WIDTH] =
            <F as Permuter31>::permute(permutation_inputs.try_into().unwrap());
        for i in 0..SPONGE_WIDTH {
            let out = witness.get_wire(Wire {
                row,
                column: Gate::wire_output(op, i),
            });
            assert_eq!(out, expected_outputs[i]);
        }
    }

    #[test]
    fn low_degree() {
        type F = BabyBear;
        let gate = Poseidon2BabyBearGate::<F, 4>::new();
        test_low_degree::<F, Poseidon2BabyBearGate<F, 4>, 4, 8>(gate)
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 4;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = 8;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let gate = Poseidon2BabyBearGate::<F, D>::new();
        test_eval_fns::<F, C, _, D, NUM_HASH_OUT_ELTS>(gate)
    }

    #[test]
    fn test_permute_internal() {
        use p3_baby_bear::{BabyBearDiffusionMatrixParameters, BabyBearParameters};
        use p3_monty_31::DiffusionMatrixParameters;
        type F = BabyBear;

        let mut state: [F; SPONGE_WIDTH] = F::rand_array();
        let mut state_clone: [F; SPONGE_WIDTH] = state.clone();
        <BabyBearDiffusionMatrixParameters as DiffusionMatrixParameters<
            BabyBearParameters,
            SPONGE_WIDTH,
        >>::permute_state(&mut state);
        permute_internal_mut(&mut state_clone);
        assert_eq!(state, state_clone);
    }

    #[test]
    fn test_permute_internal_circuit() {
        const D: usize = 4;
        const NUM_HASH_OUT_ELTS: usize = 8;
        type F = BabyBear;
        type EF = <F as HasExtension<D>>::Extension;

        let mut state: [EF; SPONGE_WIDTH] = EF::rand_array();
        let config = CircuitConfig::standard_recursion_config_bb_wide();
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);
        let mut pw = PartialWitness::<F>::new();
        let mut state_target: [ExtensionTarget<D>; SPONGE_WIDTH] = builder
            .add_virtual_extension_targets(SPONGE_WIDTH)
            .try_into()
            .unwrap();
        pw.set_extension_targets(&state_target, &state);
        permute_internal_mut::<EF>(&mut state);
        permute_internal_mut_circuit(&mut builder, &mut state_target);
        pw.set_extension_targets(&state_target, &state);
        let data = builder.build::<Poseidon2BabyBearConfig>();
        let proof = data.prove(pw);
        data.verify(proof.unwrap()).unwrap();
    }

    #[test]
    fn test_permute_external_circuit() {
        const D: usize = 4;
        const NUM_HASH_OUT_ELTS: usize = 8;
        type F = BabyBear;
        type EF = <F as HasExtension<D>>::Extension;

        let mut state: [EF; SPONGE_WIDTH] = EF::rand_array();
        let config = CircuitConfig::standard_recursion_config_gl();
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);
        let mut pw = PartialWitness::<F>::new();
        let mut state_target: [ExtensionTarget<D>; SPONGE_WIDTH] = builder
            .add_virtual_extension_targets(SPONGE_WIDTH)
            .try_into()
            .unwrap();
        pw.set_extension_targets(&state_target, &state);
        permute_external_mut::<EF, SPONGE_WIDTH>(&mut state);
        permute_external_mut_circuit(&mut builder, &mut state_target);
        // This should cause failure if permute_external_mut_circuit and permute_external_mut are not consistent.
        pw.set_extension_targets(&state_target, &state);
        let data = builder.build::<Poseidon2BabyBearConfig>();
        let proof = data.prove(pw);
        data.verify(proof.unwrap()).unwrap();
    }
}
