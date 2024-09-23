use core::fmt::Debug;

use lazy_static::lazy_static;
use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_field::{AbstractField, PrimeField64, TwoAdicField};
use p3_poseidon2;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::Permutation;
use plonky2_field::types::HasExtension;

use super::hash_types::{HashOut, RichField};
use super::hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation};
use crate::gates::poseidon2_babybear::Poseidon2BabyBearGate;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::{AlgebraicHasher, Hasher};

pub(crate) const HALF_N_FULL_ROUNDS: usize = 4;
pub(crate) const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub(crate) const N_PARTIAL_ROUNDS: usize = 13;
pub(crate) const SPONGE_RATE: usize = 8;
pub(crate) const SPONGE_CAPACITY: usize = 8;
pub const SPONGE_WIDTH: usize = 16;


#[rustfmt::skip]
pub(crate) const EXTERNAL_CONSTANTS: [[u32; SPONGE_WIDTH]; N_FULL_ROUNDS_TOTAL] = [
    [
        1321363468, 285374923, 858595076, 131742120, 550898981, 109281027, 1548327248, 299186948,
        1198120888, 1302311359, 568137078, 1484856917, 1301979945, 725688886, 941758026, 323341913,
    ],
    [
        1049323172, 822409348, 1406080127, 1279024384, 214862539, 904628921, 1320747287, 11578228,
        1036373712, 1474430466, 1430509860, 111174484, 1124450171, 85382027, 679880882, 243277213,
    ],
    [
        1338495990, 1523013347, 1841068573, 578194469, 47683837, 1790441672, 1628061601,
        1716216090, 1635810049, 1115145248, 1117524270, 678640014, 1962751651, 1367401392,
        11688709, 1950824358,
    ],
    [
        528649031, 1937116923, 1460949223, 1193074357, 1221801411, 1183923117, 433505619,
        1928933309, 505759755, 285671663, 1047265910, 909281502, 1258966486, 864761693, 307024510,
        504858517,
    ],
    [
        1467478033, 1754565867, 432187324, 1452390672, 881974300, 550050336, 1447309270, 939419487,
        1783112406, 1166910332, 107514714, 580516863, 2003318760, 854475946, 934896823, 994783668,
    ],
    [
        1841107561, 438269126, 1550523825, 913322122, 600932628, 583000098, 1262690949, 105797869,
        277542016, 170491952, 365854467, 1479645308, 1457660602, 1635879552, 499155053, 741227047,
    ],
    [
        651389942, 464828001, 89696107, 360044673, 230330371, 1773129416, 1380150763, 745014723,
        793475694, 1361274828, 1443741698, 51616650, 731414218, 1087554954, 1273943885, 311581717,
    ],
    [
        702702762, 1473247301, 132108357, 1348260424, 476775430, 1438949459, 2434448, 1349232398,
        1954471898, 1762138591, 1271221795, 1593266476, 864488771, 139147729, 1053373910,
        422842363,
    ],
];

pub(crate) const INTERNAL_CONSTANTS: [u32; N_PARTIAL_ROUNDS] = [
    402771160, 320708227, 1122772462, 100431997, 202594011, 1226485372, 1088619034, 64118538,
    109828860, 724723599, 1662837151, 797753907, 1075635743,
];

lazy_static! {
    pub static ref poseidon2: Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, SPONGE_WIDTH, 7> = {
        Poseidon2::new(
            N_FULL_ROUNDS_TOTAL,
            EXTERNAL_CONSTANTS
                .map(|arr| arr.map(BabyBear::from_canonical_u32))
                .to_vec(),
            Poseidon2ExternalMatrixGeneral,
            N_PARTIAL_ROUNDS,
            INTERNAL_CONSTANTS
                .map(BabyBear::from_canonical_u32)
                .to_vec(),
            DiffusionMatrixBabyBear::default(),
        )
    };
}

#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Poseidon31Permutation<T> {
    state: [T; SPONGE_WIDTH],
}

impl<T: Eq> Eq for Poseidon31Permutation<T> {}

impl<T> AsRef<[T]> for Poseidon31Permutation<T> {
    fn as_ref(&self) -> &[T] {
        &self.state
    }
}

pub(crate) trait Permuter31: Sized {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH];
}

impl Permuter31 for Target {
    fn permute(_input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        panic!("Call `permute_swapped()` instead of `permute()`");
    }
}
#[derive(Debug)]
pub struct Poseidon31;

impl<T: Copy + Debug + Default + Eq + Permuter31 + Send + Sync> PlonkyPermutation<T>
    for Poseidon31Permutation<T>
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

impl<F: RichField> Permuter31 for F {
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        let mut res = input
            .map(|x| <F as PrimeField64>::as_canonical_u64(&x))
            .map(BabyBear::from_canonical_u64);
        poseidon2.permute_mut(&mut res);
        res.map(|x| <BabyBear as PrimeField64>::as_canonical_u64(&x))
            .map(F::from_canonical_u64)
    }
}

/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon2BabyBearHash;
impl<F: RichField> Hasher<F> for Poseidon2BabyBearHash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<F, 8>;
    type Permutation = Poseidon31Permutation<F>;

    fn hash_no_pad(input: &[F]) -> Self::Hash {
        hash_n_to_hash_no_pad::<F, Self::Permutation, 8>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        let res = compress::<F, Self::Permutation, 8>(left, right);
        res
    }
}

impl<F: RichField> AlgebraicHasher<F, 8> for Poseidon2BabyBearHash {
    type AlgebraicPermutation = Poseidon31Permutation<Target>;

    // TODO: first define the Poseidon2_BabyBear Gates and then this method.
    #[allow(unused)]
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D, 8>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + HasExtension<D>,
        <F as HasExtension<D>>::Extension: TwoAdicField,
    {
        let gate_type: Poseidon2BabyBearGate<F, D> = Poseidon2BabyBearGate::<F, D>::new();
        let (row, op) = builder.find_slot(gate_type.clone(), &[], &[]);

        let swap_wire = Poseidon2BabyBearGate::<F, D>::wire_swap(op);
        let swap_wire = Target::wire(row, swap_wire);
        builder.connect(swap.target, swap_wire);

        // Route input wires.
        let inputs = inputs.as_ref();
        for i in 0..SPONGE_WIDTH {
            let in_wire = Poseidon2BabyBearGate::<F, D>::wire_input(op, i);
            let in_wire = Target::wire(row, in_wire);
            builder.connect(inputs[i], in_wire);
        }

        // Collect output wires.
        Self::AlgebraicPermutation::new(
            (0..SPONGE_WIDTH)
                .map(|i| Target::wire(row, Poseidon2BabyBearGate::<F, D>::wire_output(op, i))),
        )
    }
}

#[test]
fn test_poseidon2_babybear() {
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use plonky2_field::types::Sample;
    use crate::plonk::config::Poseidon2BabyBearConfig;
    type F = BabyBear;
    const D: usize = 4;
    const NUM_HASH_OUT_ELTS: usize = 8;
    type H = Poseidon2BabyBearHash;
    type C = Poseidon2BabyBearConfig;
    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(
        CircuitConfig::standard_recursion_config_bb_wide(),
    );
    let vec = F::rand_vec(NUM_HASH_OUT_ELTS * 3);
    let res = H::hash_or_noop(&vec);
    let vec_target = builder.add_virtual_targets(NUM_HASH_OUT_ELTS * 3);
    let res_target = builder.hash_or_noop::<H>(vec_target.clone());

    let mut pw = PartialWitness::<F>::new();
    pw.set_target_arr(&vec_target, &vec);
    pw.set_hash_target(res_target, res);
    let data: CircuitData<F,C,D,NUM_HASH_OUT_ELTS> = builder.build();
    let proof = data.prove(pw);
    data.verify(proof.unwrap()).unwrap();

}
