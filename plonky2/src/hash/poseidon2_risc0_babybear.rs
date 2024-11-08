use core::fmt::Debug;
use std::marker::PhantomData;

use lazy_static::lazy_static;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField64};
use p3_poseidon2;
use p3_poseidon2::{DiffusionPermutation, Poseidon2, Poseidon2ExternalMatrixHL};
use p3_symmetric::Permutation;

use crate::field::types::HasExtension;
use crate::gates::poseidon2_risc0_babybear::Poseidon2R0BabyBearGate;
use crate::hash::hash_types::HashOut;
use crate::hash::hashing::{compress, PlonkyPermutation};
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

use super::hash_types::RichField;

pub(crate) const HALF_N_FULL_ROUNDS: usize = 4;
pub(crate) const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub(crate) const N_PARTIAL_ROUNDS: usize = 21;
pub(crate) const SPONGE_RATE: usize = 16;
pub(crate) const SPONGE_CAPACITY: usize = 8;
pub const SPONGE_WIDTH: usize = 24;

#[rustfmt::skip]
pub(crate) const EXTERNAL_CONSTANTS: [[u32; SPONGE_WIDTH]; N_FULL_ROUNDS_TOTAL] = [
    [0x0FA20C37, 0x0795BB97, 0x12C60B9C, 0x0EABD88E, 0x096485CA, 0x07093527, 0x1B1D4E50, 0x30A01ACE,
        0x3BD86F5A, 0x69AF7C28, 0x3F94775F, 0x731560E8, 0x465A0ECD, 0x574EF807, 0x62FD4870, 0x52CCFE44,
        0x14772B14, 0x4DEDF371, 0x260ACD7C, 0x1F51DC58, 0x75125532, 0x686A4D7B, 0x54BAC179, 0x31947706,
    ],
    [
        0x29799D3B, 0x6E01AE90, 0x203A7A64, 0x4F7E25BE, 0x72503F77, 0x45BD3B69, 0x769BD6B4, 0x5A867F08,
        0x4FDBA082, 0x251C4318, 0x28F06201, 0x6788C43A, 0x4C6D6A99, 0x357784A8, 0x2ABAF051, 0x770F7DE6,
        0x1794B784, 0x4796C57A, 0x724B7A10, 0x449989A7, 0x64935CF1, 0x59E14AAC, 0x0E620BB8, 0x3AF5A33B,
    ],
    [
        0x4465CC0E, 0x019DF68F, 0x4AF8D068, 0x08784F82, 0x0CEFDEAE, 0x6337A467, 0x32FA7A16, 0x486F62D6,
        0x386A7480, 0x20F17C4A, 0x54E50DA8, 0x2012CF03, 0x5FE52950, 0x09AFB6CD, 0x2523044E, 0x5C54D0EF,
        0x71C01F3C, 0x60B2C4FB, 0x4050B379, 0x5E6A70A5, 0x418543F5, 0x71DEBE56, 0x1AAD2994, 0x3368A483,
    ],
    [
        0x07A86F3A, 0x5EA43FF1, 0x2443780E, 0x4CE444F7, 0x146F9882, 0x3132B089, 0x197EA856, 0x667030C3,
        0x2317D5DC, 0x0C2C48A7, 0x56B2DF66, 0x67BD81E9, 0x4FCDFB19, 0x4BAAEF32, 0x0328D30A, 0x6235760D,
        0x12432912, 0x0A49E258, 0x030E1B70, 0x48CAEB03, 0x49E4D9E9, 0x1051B5C6, 0x6A36DBBE, 0x4CFF27A5,
    ],
    [
        0x032959AD, 0x2B18AF6A, 0x55D3DC8C, 0x43BD26C8, 0x0C41595F, 0x7048D2E2, 0x00DB8983, 0x2AF563D7,
        0x6E84758F, 0x611D64E1, 0x1F9977E2, 0x64163A0A, 0x5C5FC27B, 0x02E22561, 0x3A2D75DB, 0x1BA7B71A,
        0x34343F64, 0x7406B35D, 0x19DF8299, 0x6FF4480A, 0x514A81C8, 0x57AB52CE, 0x6AD69F52, 0x3E0C0E0D,
    ],
    [
        0x48126114, 0x2A9D62CC, 0x17441F23, 0x485762BB, 0x2F218674, 0x06FDC64A, 0x0861B7F2, 0x3B36EEE6,
        0x70A11040, 0x04B31737, 0x3722A872, 0x2A351C63, 0x623560DC, 0x62584AB2, 0x382C7C04, 0x3BF9EDC7,
        0x0E38FE51, 0x376F3B10, 0x5381E178, 0x3AFC61C7, 0x5C1BCB4D, 0x6643CE1F, 0x2D0AF1C1, 0x08F583CC,
    ],
    [
        0x5D6FF60F, 0x6324C1E5, 0x74412FB7, 0x70C0192E, 0x0B72F141, 0x4067A111, 0x57388C4F, 0x351009EC,
        0x0974C159, 0x539A58B3, 0x038C0CFF, 0x476C0392, 0x3F7BC15F, 0x4491DD2C, 0x4D1FEF55, 0x04936AE3,
        0x58214DD4, 0x683C6AAD, 0x1B42F16B, 0x6DC79135, 0x2D4E71EC, 0x3E2946EA, 0x59DCE8DB, 0x6CEE892A,
    ],
    [
        0x47F07350, 0x7106CE93, 0x3BD4A7A9, 0x2BFE636A, 0x430011E9, 0x001CD66A, 0x307FAF5B, 0x0D9EF3FE,
        0x6D40043A, 0x2E8F470C, 0x1B6865E8, 0x0C0E6C01, 0x4D41981F, 0x423B9D3D, 0x410408CC, 0x263F0884,
        0x5311BBD0, 0x4DAE58D8, 0x30401CEA, 0x09AFA575, 0x4B3D5B42, 0x63AC0B37, 0x5FE5BB14, 0x5244E9D4,
    ]
];

pub(crate) const INTERNAL_CONSTANTS: [u32; N_PARTIAL_ROUNDS] = [
    0x1DA78EC2, 0x730B0924, 0x3EB56CF3, 0x5BD93073, 0x37204C97, 0x51642D89, 0x66E943E8, 0x1A3E72DE,
    0x70BEB1E9, 0x30FF3B3F, 0x4240D1C4, 0x12647B8D, 0x65D86965, 0x49EF4D7C, 0x47785697, 0x46B3969F,
    0x5C7B7A0E, 0x7078FC60, 0x4F22D482, 0x482A9AEE, 0x6BEB839D,
];

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabyBearR0<F: AbstractField> {
    _phantom: PhantomData<F>,
}

impl<F: AbstractField> DiffusionMatrixBabyBearR0<F> {}

pub const M_INT_DIAG_HZN: &[u32; 24] = &[
    0x409133f0, 0x1667a8a1, 0x06a6c7b6, 0x6f53160e, 0x273b11d1, 0x03176c5d, 0x72f9bbf9, 0x73ceba91,
    0x5cdef81d, 0x01393285, 0x46daee06, 0x065d7ba6, 0x52d72d6f, 0x05dd05e0, 0x3bab4b63, 0x6ada3842,
    0x2fc5fbec, 0x770d61b0, 0x5715aae9, 0x03ef0e90, 0x75b6c770, 0x242adf5f, 0x00d0ca4c, 0x36c0e388,
];

impl<F: Clone + AbstractField + Sync, const WIDTH: usize> Permutation<[F; WIDTH]>
    for DiffusionMatrixBabyBearR0<F>
{
    fn permute_mut(&self, input: &mut [F; WIDTH]) {
        let sum: F = input.iter().fold(F::zero(), |acc, x| acc + x.clone());
        for i in 0..SPONGE_WIDTH {
            input[i] = sum.clone() + F::from_canonical_u32(M_INT_DIAG_HZN[i]) * input[i].clone();
        }
    }
}

impl<F: Clone + AbstractField + Sync, const WIDTH: usize> DiffusionPermutation<F, WIDTH>
    for DiffusionMatrixBabyBearR0<F>
{
}

lazy_static! {
    pub static ref poseidon2_r0: Poseidon2<BabyBear, Poseidon2ExternalMatrixHL, DiffusionMatrixBabyBearR0<BabyBear>,/* DiffusionMatrixBabyBear,*/ SPONGE_WIDTH, 7> = {
        Poseidon2::new(
            N_FULL_ROUNDS_TOTAL,
            EXTERNAL_CONSTANTS
                .map(|arr| arr.map(BabyBear::from_canonical_u32))
                .to_vec(),
            Poseidon2ExternalMatrixHL,
            N_PARTIAL_ROUNDS,
            INTERNAL_CONSTANTS
                .map(BabyBear::from_canonical_u32)
                .to_vec(),
            DiffusionMatrixBabyBearR0::<BabyBear>::default(),
        )
    };
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Poseidon31R0Permutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for Poseidon31R0Permutation<T> {}

impl<T> AsRef<[T]> for Poseidon31R0Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

pub(crate) trait Permuter31R0: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl Permuter31R0 for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}
#[derive(Debug)]
pub struct Poseidon31R0;

impl<T: Copy + Debug + Default + Eq + Permuter31R0 + Send + Sync> PlonkyPermutation<T>
    for Poseidon31R0Permutation<T>
{
    const RATE: usize = SPONGE_RATE;
    const WIDTH: usize = SPONGE_WIDTH;

    fn new<I: IntoIterator<Item = T>>(elts: I) -> Self {
        let mut perm = Self {
            state: [T::default(); SPONGE_WIDTH],
        };
        perm.set_from_iter(elts, 0);
        perm
    }

    fn set_elt(&mut self, elt: T, idx: usize) {
        self.state[idx] = elt;
    }

    fn set_from_slice(&mut self, elts: &[T], start_idx: usize) {
        let begin = start_idx;
        let end = start_idx + elts.len();
        self.state[begin..end].copy_from_slice(elts);
    }

    fn set_from_iter<I: IntoIterator<Item = T>>(&mut self, elts: I, start_idx: usize) {
        for (s, e) in self.state[start_idx..].iter_mut().zip(elts) {
            *s = e;
        }
    }

    fn permute(&mut self) {
        self.state = T::permute(self.state);
    }

    fn squeeze(&self) -> &[T] {
        &self.state[..Self::RATE]
    }
}

impl<F: PrimeField64> Permuter31R0 for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut res = input
            .map(|x| Self::as_canonical_u64(&x))
            .map(BabyBear::from_canonical_u64);
        poseidon2_r0.permute_mut(&mut res);
        res.map(|x| <BabyBear as PrimeField64>::as_canonical_u64(&x))
            .map(Self::from_canonical_u64)
    }
}
/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2R0BabyBearHash;
impl<F: RichField> Hasher<F> for Poseidon2R0BabyBearHash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F, 8>;
    type Permutation = Poseidon31R0Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        let mut perm = Self::Permutation::new(core::iter::repeat(F::zero()));

        // Absorb all input chunks.
        for input_chunk in input.chunks(Self::Permutation::RATE) {
            perm.set_from_slice(input_chunk, 0);
            for i in input_chunk.len()..Self::Permutation::RATE {
                perm.set_elt(F::zero(), i);
            }
            perm.permute();
        }

        // Squeeze until we have the desired number of outputs.
        let mut outputs = Vec::new();
        loop {
            for &item in perm.squeeze() {
                outputs.push(item);
                if outputs.len() == SPONGE_CAPACITY {
                    return HashOut::from_vec(outputs);
                }
            }
            perm.permute();
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let res = compress::<F, Self::Permutation, 8>(left, right);
        res
    }
}

impl<F: RichField> AlgebraicHasher<F, 8> for Poseidon2R0BabyBearHash {
    type AlgebraicPermutation = Poseidon31R0Permutation<Target>;

    #[allow(unused)]
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D, 8>,
    ) -> Self::AlgebraicPermutation
    where
        F: HasExtension<D>,

    {
        let gate_type: Poseidon2R0BabyBearGate<F, D> = Poseidon2R0BabyBearGate::<F, D>::new();
        let (row, op) = builder.find_slot(gate_type.clone(), &[], &[]);

        let swap_wire = Poseidon2R0BabyBearGate::<F, D>::wire_swap(op);
        let swap_wire = Target::wire(row, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = Poseidon2R0BabyBearGate::<F, D>::wire_input(op, i);
            let in_wire = Target::wire(row, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..SPONGE_WIDTH)
                .map(|i| Target::wire(row, Poseidon2R0BabyBearGate::<F, D>::wire_output(op, i))),
        )
    }

    fn hash_n_to_m_no_pad_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D, 8>,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target>
    where
        F: HasExtension<D>,
            {
        let zero = builder.zero();
        let mut state = Self::AlgebraicPermutation::new(core::iter::repeat(zero));

        // Absorb all input chunks.
        for input_chunk in inputs.chunks(Self::AlgebraicPermutation::RATE) {
            // Overwrite the first r elements with the inputs. This differs from a standard sponge,
            // where we would xor or add in the inputs. This is a well-known variant, though,
            // sometimes called "overwrite mode".
            state.set_from_slice(input_chunk, 0);
            for i in input_chunk.len()..Self::AlgebraicPermutation::RATE {
                state.set_elt(zero, i);
            }
            state = builder.permute::<Self>(state);
        }

        // Squeeze until we have the desired number of outputs.
        let mut outputs = Vec::with_capacity(num_outputs);
        loop {
            for &s in state.squeeze() {
                outputs.push(s);
                if outputs.len() == num_outputs {
                    return outputs;
                }
            }
            state = builder.permute::<Self>(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;

    use crate::field::types::Sample;
    use crate::hash::hash_types::BABYBEAR_NUM_HASH_OUT_ELTS;
    use crate::plonk::circuit_builder::CircuitBuilder;

    use super::{poseidon2_r0, Poseidon2R0BabyBearHash, SPONGE_WIDTH};

    #[test]
    fn test_against_r0_values() {
        let input: &mut [BabyBear; SPONGE_WIDTH] = &mut [
            0x00000000, 0x00000001, 0x00000002, 0x00000003, 0x00000004, 0x00000005, 0x00000006,
            0x00000007, 0x00000008, 0x00000009, 0x0000000A, 0x0000000B, 0x0000000C, 0x0000000D,
            0x0000000E, 0x0000000F, 0x00000010, 0x00000011, 0x00000012, 0x00000013, 0x00000014,
            0x00000015, 0x00000016, 0x00000017,
        ]
        .map(BabyBear::from_canonical_u32);

        let expected: [BabyBear; SPONGE_WIDTH] = [
            0x2ed3e23d, 0x12921fb0, 0x0e659e79, 0x61d81dc9, 0x32bae33b, 0x62486ae3, 0x1e681b60,
            0x24b91325, 0x2a2ef5b9, 0x50e8593e, 0x5bc818ec, 0x10691997, 0x35a14520, 0x2ba6a3c5,
            0x279d47ec, 0x55014e81, 0x5953a67f, 0x2f403111, 0x6b8828ff, 0x1801301f, 0x2749207a,
            0x3dc9cf21, 0x3c985ba2, 0x57a99864,
        ]
        .map(BabyBear::from_canonical_u32);

        poseidon2_r0.permute_mut(input);

        assert_eq!(*input, expected);
    }

    #[test]
    fn test_poseidon2_r0_babybear() {
        use crate::iop::witness::{PartialWitness, WitnessWrite};
        use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
        use crate::plonk::config::Poseidon2BabyBearConfig;
        type F = BabyBear;
        const D: usize = 4;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type H = Poseidon2R0BabyBearHash;
        type C = Poseidon2BabyBearConfig;
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(
            CircuitConfig::standard_recursion_config_bb_wide(),
        );
        let vec = F::rand_vec(NUM_HASH_OUT_ELTS * 3);
        let vec_target = builder.add_virtual_targets(NUM_HASH_OUT_ELTS * 3);
        builder.hash_or_noop::<H>(vec_target.clone());
        // builder.hash_or_noop::<H>(vec_target.clone());

        let mut pw = PartialWitness::<F>::new();
        pw.set_target_arr(&vec_target, &vec);
        let data: CircuitData<F, C, D, NUM_HASH_OUT_ELTS> = builder.build();
        let proof = data.prove(pw);
        data.verify(proof.unwrap()).unwrap();
    }
}
