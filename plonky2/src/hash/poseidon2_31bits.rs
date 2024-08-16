use core::fmt::Debug;
use p3_babybear::BabyBear;
use p3_field::{AbstractField, TwoAdicField};
use p3_poseidon2;
use p3_symmetric::Permutation;
use plonky2_field::types::HasExtension;

use crate::{iop::target::{BoolTarget, Target}, plonk::config::{AlgebraicHasher, Hasher}};
use crate::plonk::circuit_builder::CircuitBuilder;
use super::{hash_types::{HashOut, RichField}, hashing::{compress, hash_n_to_hash_no_pad, PlonkyPermutation}};

const SPONGE_RATE: usize = 8;
const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = 24;

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

impl Permuter31 for BabyBear 
{
    fn permute(input: [Self; SPONGE_WIDTH]) -> [Self; SPONGE_WIDTH] {
        type F = BabyBear;
        use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
        use p3_babybear::{DiffusionMatrixBabyBear};
        let rounds_f = 8;
        let external_constants = [
            [
                1321363468, 285374923, 858595076, 131742120, 550898981, 109281027, 1548327248,299186948,
                1198120888, 1302311359, 568137078,1484856917, 1301979945, 725688886, 941758026, 323341913,
                1049323172, 822409348, 1406080127,1279024384,214862539, 904628921, 1320747287,11578228,
            ],
            [
                1036373712, 1474430466,1430509860, 111174484,1124450171, 85382027,679880882,243277213,
                1338495990,1523013347,1841068573,578194469,47683837,1790441672,1628061601,1716216090,
                1635810049,1115145248,1117524270,678640014,1962751651,1367401392,11688709,1950824358,
            ],
            [
                528649031,1937116923,1460949223,1193074357,1221801411,1183923117,433505619,1928933309,
                505759755,285671663,1047265910,909281502,1258966486,864761693,307024510,504858517,
                1467478033,1754565867,432187324,1452390672,881974300,550050336,1447309270,939419487,
            ],
            [
                1783112406,1166910332,107514714,580516863,2003318760,854475946,934896823,994783668,
                1841107561,438269126,1550523825,913322122,600932628,583000098,1262690949,105797869,
                277542016,170491952,365854467,1479645308,1457660602,1635879552,499155053,741227047,
            ],
            [
                651389942,464828001,89696107,360044673,230330371,1773129416,1380150763,745014723,
                793475694,1361274828,1443741698,51616650,731414218,1087554954,1273943885,311581717,
                702702762,1473247301,132108357,1348260424,476775430,1438949459,2434448,1349232398,
            ],
            [
                1954471898,1762138591,1271221795,1593266476,864488771,139147729,1053373910,422842363,
                402771160,320708227,1122772462,100431997,202594011,1226485372,1088619034,64118538,
                109828860,724723599,1662837151,797753907,1075635743,1575373462,365777825,604117715,
            ],
            [
                1952887467,1576774110,1060853314,1306115009,164487850,1937929749,1736073259,1774187999,
                1808417874,1401859259,560553519,1224550565,1406884940,967539390,1612868776,1866059303,
                1842816770,1640443045,187350672,913519914,908692598,84796829,1097450478,1663405354,
            ],
            [
                1506039113,1051738250,1926136353,1936901654,879697270,1118890060,656192020,258274710,
                770459103,41927072,1016302271,1639635057,1851773725,1865852628,1414217486,418545569,
                876943762,605521983,1364698550,787157278,352400855,895034787,858954246,1175237388,
            ],
        ].map(|arr| arr.map(F::from_canonical_u32)).to_vec();
        let external_linear_layer = Poseidon2ExternalMatrixGeneral;
        let rounds_p = 21;
        let internal_constants = [
            617787195,1986707395,1246478145,26078679,58345221,661023325,880140881,252492041,
            1756164978,583975464,373781724,1493388549,1965581821,105804973,1379083274,969091717,
            143950039,1011062373,1888518418,744902302,1685783724,
        ].map(F::from_canonical_u32).to_vec();
        let internal_linear_layer = DiffusionMatrixBabyBear::default();
        let poseidon2: Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 24, 7> =
        Poseidon2::new(rounds_f, external_constants, external_linear_layer, rounds_p, internal_constants, internal_linear_layer);
        let mut res = input;
        poseidon2.permute_mut(&mut res);
        res
    }}

/// Poseidon hash function.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Poseidon31Hash;
impl Hasher<BabyBear> for Poseidon31Hash {
    const HASH_SIZE: usize = 4 * 8;
    type Hash = HashOut<BabyBear>;
    type Permutation = Poseidon31Permutation<BabyBear>;

    fn hash_no_pad(input: &[BabyBear]) -> Self::Hash {
        hash_n_to_hash_no_pad::<BabyBear, Self::Permutation>(input)
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash {
        compress::<BabyBear, Self::Permutation>(left, right)
    }
}

impl AlgebraicHasher<BabyBear> for Poseidon31Hash {
    type AlgebraicPermutation = Poseidon31Permutation<Target>;

    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<BabyBear, D>,
    ) -> Self::AlgebraicPermutation
    where
        BabyBear: RichField + HasExtension<D>,
        <BabyBear as HasExtension<D>>::Extension: TwoAdicField,
    {
        todo!()
    }
}