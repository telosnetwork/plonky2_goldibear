use core::arch::x86_64::*;
use core::fmt;
use core::fmt::{Debug, Formatter};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_goldilocks::Goldilocks;
use crate::ops::Square;
use crate::packed::PackedField;
use p3_field::{AbstractField, Field, PrimeField64};


/// AVX512 Goldilocks Field
///
/// Ideally `Avx512Goldilocks` would wrap `__m512i`. Unfortunately, `__m512i` has an alignment
/// of 64B, which would preclude us from casting `[Goldilocks; 8]` (alignment 8B) to
/// `Avx512Goldilocks`. We need to ensure that `Avx512Goldilocks` has the same alignment as
/// `Goldilocks`. Thus we wrap `[Goldilocks; 8]` and use the `new` and `get` methods to
/// convert to and from `__m512i`.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct Avx512Goldilocks(pub [Goldilocks; 8]);

impl Avx512Goldilocks {
    #[inline]
    fn new(x: __m512i) -> Self {
        unsafe { transmute(x) }
    }
    #[inline]
    fn get(&self) -> __m512i {
        unsafe { transmute(*self) }
    }
}

impl Add<Self> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(unsafe { add(self.get(), rhs.get()) })
    }
}
impl Add<Goldilocks> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Goldilocks) -> Self {
        self + Self::from(rhs)
    }
}
impl Add<Avx512Goldilocks> for Goldilocks {
    type Output = Avx512Goldilocks;
    #[inline]
    fn add(self, rhs: Self::Output) -> Self::Output {
        Self::Output::from(self) + rhs
    }
}
impl AddAssign<Self> for Avx512Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl AddAssign<Goldilocks> for Avx512Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Goldilocks) {
        *self = *self + rhs;
    }
}

impl Debug for Avx512Goldilocks {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({:?})", self.get())
    }
}

impl Default for Avx512Goldilocks {
    #[inline]
    fn default() -> Self {
        Self::zeros()
    }
}

impl Div<Goldilocks> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Goldilocks) -> Self {
        self * rhs.inverse()
    }
}
impl DivAssign<Goldilocks> for Avx512Goldilocks {
    #[inline]
    fn div_assign(&mut self, rhs: Goldilocks) {
        *self *= rhs.inverse();
    }
}

impl From<Goldilocks> for Avx512Goldilocks {
    fn from(x: Goldilocks) -> Self {
        Self([x; 8])
    }
}

impl Mul<Self> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(unsafe { mul(self.get(), rhs.get()) })
    }
}
impl Mul<Goldilocks> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Goldilocks) -> Self {
        self * Self::from(rhs)
    }
}
impl Mul<Avx512Goldilocks> for Goldilocks {
    type Output = Avx512Goldilocks;
    #[inline]
    fn mul(self, rhs: Avx512Goldilocks) -> Self::Output {
        Self::Output::from(self) * rhs
    }
}
impl MulAssign<Self> for Avx512Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl MulAssign<Goldilocks> for Avx512Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Goldilocks) {
        *self = *self * rhs;
    }
}

impl Neg for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(unsafe { neg(self.get()) })
    }
}

impl Product for Avx512Goldilocks {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ones())
    }
}

unsafe impl PackedField for Avx512Goldilocks {
    const WIDTH: usize = 8;

    type Scalar = Goldilocks;

    fn zeros() -> Self{
        Self([Goldilocks::zero();8])
    }

    fn ones() -> Self{
        Self([Goldilocks::one();8])
    }

    fn twos() -> Self{
        Self([Goldilocks::two();8])
    }

    #[inline]
    fn from_slice(slice: &[Self::Scalar]) -> &Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &*slice.as_ptr().cast() }
    }
    #[inline]
    fn from_slice_mut(slice: &mut [Self::Scalar]) -> &mut Self {
        assert_eq!(slice.len(), Self::WIDTH);
        unsafe { &mut *slice.as_mut_ptr().cast() }
    }
    #[inline]
    fn as_slice(&self) -> &[Self::Scalar] {
        &self.0[..]
    }
    #[inline]
    fn as_slice_mut(&mut self) -> &mut [Self::Scalar] {
        &mut self.0[..]
    }

    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.get(), other.get());
        let (res0, res1) = match block_len {
            1 => unsafe { interleave1(v0, v1) },
            2 => unsafe { interleave2(v0, v1) },
            4 => unsafe { interleave4(v0, v1) },
            8 => (v0, v1),
            _ => panic!("unsupported block_len"),
        };
        (Self::new(res0), Self::new(res1))
    }
}

impl Square for Avx512Goldilocks {
    #[inline]
    fn square(&self) -> Self {
        Self::new(unsafe { square(self.get()) })
    }
}

impl Sub<Self> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(unsafe { sub(self.get(), rhs.get()) })
    }
}
impl Sub<Goldilocks> for Avx512Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Goldilocks) -> Self {
        self - Self::from(rhs)
    }
}
impl Sub<Avx512Goldilocks> for Goldilocks {
    type Output = Avx512Goldilocks;
    #[inline]
    fn sub(self, rhs: Avx512Goldilocks) -> Self::Output {
        Self::Output::from(self) - rhs
    }
}
impl SubAssign<Self> for Avx512Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl SubAssign<Goldilocks> for Avx512Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Goldilocks) {
        *self = *self - rhs;
    }
}

impl Sum for Avx512Goldilocks {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zeros())
    }
}

const FIELD_ORDER: __m512i = unsafe { transmute([Goldilocks::ORDER_U64; 8]) };
const EPSILON: __m512i = unsafe { transmute([Goldilocks::ORDER_U64.wrapping_neg(); 8]) };

#[inline]
unsafe fn canonicalize(x: __m512i) -> __m512i {
    let mask = _mm512_cmpge_epu64_mask(x, FIELD_ORDER);
    _mm512_mask_sub_epi64(x, mask, x, FIELD_ORDER)
}

#[inline]
unsafe fn add_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    let res_wrapped = _mm512_add_epi64(x, y);
    let mask = _mm512_cmplt_epu64_mask(res_wrapped, y); // mask set if add overflowed
    let res = _mm512_mask_sub_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER);
    res
}

#[inline]
unsafe fn sub_no_double_overflow_64_64(x: __m512i, y: __m512i) -> __m512i {
    let mask = _mm512_cmplt_epu64_mask(x, y); // mask set if sub will underflow (x < y)
    let res_wrapped = _mm512_sub_epi64(x, y);
    let res = _mm512_mask_add_epi64(res_wrapped, mask, res_wrapped, FIELD_ORDER);
    res
}

#[inline]
unsafe fn add(x: __m512i, y: __m512i) -> __m512i {
    add_no_double_overflow_64_64(x, canonicalize(y))
}

#[inline]
unsafe fn sub(x: __m512i, y: __m512i) -> __m512i {
    sub_no_double_overflow_64_64(x, canonicalize(y))
}

#[inline]
unsafe fn neg(y: __m512i) -> __m512i {
    _mm512_sub_epi64(FIELD_ORDER, canonicalize(y))
}

const LO_32_BITS_MASK: __mmask16 = unsafe { transmute(0b0101010101010101u16) };

#[inline]
unsafe fn mul64_64(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    // We want to move the high 32 bits to the low position. The multiplication instruction ignores
    // the high 32 bits, so it's ok to just duplicate it into the low position. This duplication can
    // be done on port 5; bitshifts run on port 0, competing with multiplication.
    //   This instruction is only provided for 32-bit floats, not integers. Idk why Intel makes the
    // distinction; the casts are free and it guarantees that the exact bit pattern is preserved.
    // Using a swizzle instruction of the wrong domain (float vs int) does not increase latency
    // since Haswell.
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));
    let y_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(y)));

    // All four pairwise multiplications
    let mul_ll = _mm512_mul_epu32(x, y);
    let mul_lh = _mm512_mul_epu32(x, y_hi);
    let mul_hl = _mm512_mul_epu32(x_hi, y);
    let mul_hh = _mm512_mul_epu32(x_hi, y_hi);

    // Bignum addition
    // Extract high 32 bits of mul_ll and add to mul_hl. This cannot overflow.
    let mul_ll_hi = _mm512_srli_epi64::<32>(mul_ll);
    let t0 = _mm512_add_epi64(mul_hl, mul_ll_hi);
    // Extract low 32 bits of t0 and add to mul_lh. Again, this cannot overflow.
    // Also, extract high 32 bits of t0 and add to mul_hh.
    let t0_lo = _mm512_and_si512(t0, EPSILON);
    let t0_hi = _mm512_srli_epi64::<32>(t0);
    let t1 = _mm512_add_epi64(mul_lh, t0_lo);
    let t2 = _mm512_add_epi64(mul_hh, t0_hi);
    // Lastly, extract the high 32 bits of t1 and add to t2.
    let t1_hi = _mm512_srli_epi64::<32>(t1);
    let res_hi = _mm512_add_epi64(t2, t1_hi);

    // Form res_lo by combining the low half of mul_ll with the low half of t1 (shifted into high
    // position).
    let t1_lo = _mm512_castps_si512(_mm512_moveldup_ps(_mm512_castsi512_ps(t1)));
    let res_lo = _mm512_mask_blend_epi32(LO_32_BITS_MASK, t1_lo, mul_ll);

    (res_hi, res_lo)
}

#[inline]
unsafe fn square64(x: __m512i) -> (__m512i, __m512i) {
    // Get high 32 bits of x. See comment in mul64_64_s.
    let x_hi = _mm512_castps_si512(_mm512_movehdup_ps(_mm512_castsi512_ps(x)));

    // All pairwise multiplications.
    let mul_ll = _mm512_mul_epu32(x, x);
    let mul_lh = _mm512_mul_epu32(x, x_hi);
    let mul_hh = _mm512_mul_epu32(x_hi, x_hi);

    // Bignum addition, but mul_lh is shifted by 33 bits (not 32).
    let mul_ll_hi = _mm512_srli_epi64::<33>(mul_ll);
    let t0 = _mm512_add_epi64(mul_lh, mul_ll_hi);
    let t0_hi = _mm512_srli_epi64::<31>(t0);
    let res_hi = _mm512_add_epi64(mul_hh, t0_hi);

    // Form low result by adding the mul_ll and the low 31 bits of mul_lh (shifted to the high
    // position).
    let mul_lh_lo = _mm512_slli_epi64::<33>(mul_lh);
    let res_lo = _mm512_add_epi64(mul_ll, mul_lh_lo);

    (res_hi, res_lo)
}

#[inline]
unsafe fn reduce128(x: (__m512i, __m512i)) -> __m512i {
    let (hi0, lo0) = x;
    let hi_hi0 = _mm512_srli_epi64::<32>(hi0);
    let lo1 = sub_no_double_overflow_64_64(lo0, hi_hi0);
    let t1 = _mm512_mul_epu32(hi0, EPSILON);
    let lo2 = add_no_double_overflow_64_64(lo1, t1);
    lo2
}

#[inline]
unsafe fn mul(x: __m512i, y: __m512i) -> __m512i {
    reduce128(mul64_64(x, y))
}

#[inline]
unsafe fn square(x: __m512i) -> __m512i {
    reduce128(square64(x))
}

#[inline]
unsafe fn interleave1(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let a = _mm512_unpacklo_epi64(x, y);
    let b = _mm512_unpackhi_epi64(x, y);
    (a, b)
}

const INTERLEAVE2_IDX_A: __m512i = unsafe {
    transmute([
        0o00u64, 0o01u64, 0o10u64, 0o11u64, 0o04u64, 0o05u64, 0o14u64, 0o15u64,
    ])
};
const INTERLEAVE2_IDX_B: __m512i = unsafe {
    transmute([
        0o02u64, 0o03u64, 0o12u64, 0o13u64, 0o06u64, 0o07u64, 0o16u64, 0o17u64,
    ])
};

#[inline]
unsafe fn interleave2(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let a = _mm512_permutex2var_epi64(x, INTERLEAVE2_IDX_A, y);
    let b = _mm512_permutex2var_epi64(x, INTERLEAVE2_IDX_B, y);
    (a, b)
}

#[inline]
unsafe fn interleave4(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let a = _mm512_shuffle_i64x2::<0x44>(x, y);
    let b = _mm512_shuffle_i64x2::<0xee>(x, y);
    (a, b)
}

#[cfg(test)]
mod tests {
    use crate::arch::x86_64::avx512_goldilocks_field::Avx512Goldilocks;
    use crate::goldilocks_field::Goldilocks;
    use crate::ops::Square;
    use crate::packed::PackedField;
    use crate::types::Field;

    fn test_vals_a() -> [Goldilocks; 8] {
        [
            Goldilocks::from_noncanonical_u64(14479013849828404771),
            Goldilocks::from_noncanonical_u64(9087029921428221768),
            Goldilocks::from_noncanonical_u64(2441288194761790662),
            Goldilocks::from_noncanonical_u64(5646033492608483824),
            Goldilocks::from_noncanonical_u64(2779181197214900072),
            Goldilocks::from_noncanonical_u64(2989742820063487116),
            Goldilocks::from_noncanonical_u64(727880025589250743),
            Goldilocks::from_noncanonical_u64(3803926346107752679),
        ]
    }
    fn test_vals_b() -> [Goldilocks; 8] {
        [
            Goldilocks::from_noncanonical_u64(17891926589593242302),
            Goldilocks::from_noncanonical_u64(11009798273260028228),
            Goldilocks::from_noncanonical_u64(2028722748960791447),
            Goldilocks::from_noncanonical_u64(7929433601095175579),
            Goldilocks::from_noncanonical_u64(6632528436085461172),
            Goldilocks::from_noncanonical_u64(2145438710786785567),
            Goldilocks::from_noncanonical_u64(11821483668392863016),
            Goldilocks::from_noncanonical_u64(15638272883309521929),
        ]
    }

    #[test]
    fn test_add() {
        let a_arr = test_vals_a();
        let b_arr = test_vals_b();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_b = *Avx512Goldilocks::from_slice(&b_arr);
        let packed_res = packed_a + packed_b;
        let arr_res = packed_res.as_slice();

        let expected = a_arr.iter().zip(b_arr).map(|(&a, b)| a + b);
        for (exp, &res) in expected.zip(arr_res) {
            assert_eq!(res, exp);
        }
    }

    #[test]
    fn test_mul() {
        let a_arr = test_vals_a();
        let b_arr = test_vals_b();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_b = *Avx512Goldilocks::from_slice(&b_arr);
        let packed_res = packed_a * packed_b;
        let arr_res = packed_res.as_slice();

        let expected = a_arr.iter().zip(b_arr).map(|(&a, b)| a * b);
        for (exp, &res) in expected.zip(arr_res) {
            assert_eq!(res, exp);
        }
    }

    #[test]
    fn test_square() {
        let a_arr = test_vals_a();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_res = packed_a.square();
        let arr_res = packed_res.as_slice();

        let expected = a_arr.iter().map(|&a| a.square());
        for (exp, &res) in expected.zip(arr_res) {
            assert_eq!(res, exp);
        }
    }

    #[test]
    fn test_neg() {
        let a_arr = test_vals_a();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_res = -packed_a;
        let arr_res = packed_res.as_slice();

        let expected = a_arr.iter().map(|&a| -a);
        for (exp, &res) in expected.zip(arr_res) {
            assert_eq!(res, exp);
        }
    }

    #[test]
    fn test_sub() {
        let a_arr = test_vals_a();
        let b_arr = test_vals_b();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_b = *Avx512Goldilocks::from_slice(&b_arr);
        let packed_res = packed_a - packed_b;
        let arr_res = packed_res.as_slice();

        let expected = a_arr.iter().zip(b_arr).map(|(&a, b)| a - b);
        for (exp, &res) in expected.zip(arr_res) {
            assert_eq!(res, exp);
        }
    }

    #[test]
    fn test_interleave_is_involution() {
        let a_arr = test_vals_a();
        let b_arr = test_vals_b();

        let packed_a = *Avx512Goldilocks::from_slice(&a_arr);
        let packed_b = *Avx512Goldilocks::from_slice(&b_arr);
        {
            // Interleave, then deinterleave.
            let (x, y) = packed_a.interleave(packed_b, 1);
            let (res_a, res_b) = x.interleave(y, 1);
            assert_eq!(res_a.as_slice(), a_arr);
            assert_eq!(res_b.as_slice(), b_arr);
        }
        {
            let (x, y) = packed_a.interleave(packed_b, 2);
            let (res_a, res_b) = x.interleave(y, 2);
            assert_eq!(res_a.as_slice(), a_arr);
            assert_eq!(res_b.as_slice(), b_arr);
        }
        {
            let (x, y) = packed_a.interleave(packed_b, 4);
            let (res_a, res_b) = x.interleave(y, 4);
            assert_eq!(res_a.as_slice(), a_arr);
            assert_eq!(res_b.as_slice(), b_arr);
        }
        {
            let (x, y) = packed_a.interleave(packed_b, 8);
            let (res_a, res_b) = x.interleave(y, 8);
            assert_eq!(res_a.as_slice(), a_arr);
            assert_eq!(res_b.as_slice(), b_arr);
        }
    }

    #[test]
    fn test_interleave() {
        let in_a: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(00),
            Goldilocks::from_noncanonical_u64(01),
            Goldilocks::from_noncanonical_u64(02),
            Goldilocks::from_noncanonical_u64(03),
            Goldilocks::from_noncanonical_u64(04),
            Goldilocks::from_noncanonical_u64(05),
            Goldilocks::from_noncanonical_u64(06),
            Goldilocks::from_noncanonical_u64(07),
        ];
        let in_b: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(10),
            Goldilocks::from_noncanonical_u64(11),
            Goldilocks::from_noncanonical_u64(12),
            Goldilocks::from_noncanonical_u64(13),
            Goldilocks::from_noncanonical_u64(14),
            Goldilocks::from_noncanonical_u64(15),
            Goldilocks::from_noncanonical_u64(16),
            Goldilocks::from_noncanonical_u64(17),
        ];
        let int1_a: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(00),
            Goldilocks::from_noncanonical_u64(10),
            Goldilocks::from_noncanonical_u64(02),
            Goldilocks::from_noncanonical_u64(12),
            Goldilocks::from_noncanonical_u64(04),
            Goldilocks::from_noncanonical_u64(14),
            Goldilocks::from_noncanonical_u64(06),
            Goldilocks::from_noncanonical_u64(16),
        ];
        let int1_b: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(01),
            Goldilocks::from_noncanonical_u64(11),
            Goldilocks::from_noncanonical_u64(03),
            Goldilocks::from_noncanonical_u64(13),
            Goldilocks::from_noncanonical_u64(05),
            Goldilocks::from_noncanonical_u64(15),
            Goldilocks::from_noncanonical_u64(07),
            Goldilocks::from_noncanonical_u64(17),
        ];
        let int2_a: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(00),
            Goldilocks::from_noncanonical_u64(01),
            Goldilocks::from_noncanonical_u64(10),
            Goldilocks::from_noncanonical_u64(11),
            Goldilocks::from_noncanonical_u64(04),
            Goldilocks::from_noncanonical_u64(05),
            Goldilocks::from_noncanonical_u64(14),
            Goldilocks::from_noncanonical_u64(15),
        ];
        let int2_b: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(02),
            Goldilocks::from_noncanonical_u64(03),
            Goldilocks::from_noncanonical_u64(12),
            Goldilocks::from_noncanonical_u64(13),
            Goldilocks::from_noncanonical_u64(06),
            Goldilocks::from_noncanonical_u64(07),
            Goldilocks::from_noncanonical_u64(16),
            Goldilocks::from_noncanonical_u64(17),
        ];
        let int4_a: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(00),
            Goldilocks::from_noncanonical_u64(01),
            Goldilocks::from_noncanonical_u64(02),
            Goldilocks::from_noncanonical_u64(03),
            Goldilocks::from_noncanonical_u64(10),
            Goldilocks::from_noncanonical_u64(11),
            Goldilocks::from_noncanonical_u64(12),
            Goldilocks::from_noncanonical_u64(13),
        ];
        let int4_b: [Goldilocks; 8] = [
            Goldilocks::from_noncanonical_u64(04),
            Goldilocks::from_noncanonical_u64(05),
            Goldilocks::from_noncanonical_u64(06),
            Goldilocks::from_noncanonical_u64(07),
            Goldilocks::from_noncanonical_u64(14),
            Goldilocks::from_noncanonical_u64(15),
            Goldilocks::from_noncanonical_u64(16),
            Goldilocks::from_noncanonical_u64(17),
        ];

        let packed_a = *Avx512Goldilocks::from_slice(&in_a);
        let packed_b = *Avx512Goldilocks::from_slice(&in_b);
        {
            let (x1, y1) = packed_a.interleave(packed_b, 1);
            assert_eq!(x1.as_slice(), int1_a);
            assert_eq!(y1.as_slice(), int1_b);
        }
        {
            let (x2, y2) = packed_a.interleave(packed_b, 2);
            assert_eq!(x2.as_slice(), int2_a);
            assert_eq!(y2.as_slice(), int2_b);
        }
        {
            let (x4, y4) = packed_a.interleave(packed_b, 4);
            assert_eq!(x4.as_slice(), int4_a);
            assert_eq!(y4.as_slice(), int4_b);
        }
        {
            let (x8, y8) = packed_a.interleave(packed_b, 8);
            assert_eq!(x8.as_slice(), in_a);
            assert_eq!(y8.as_slice(), in_b);
        }
    }
}
