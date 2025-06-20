#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::util::serialization::{Buffer, IoResult, Read, Write};

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>
{
    /// Checks that `x < 2^n_log` using a `BaseSumGate`.
    pub fn range_check(&mut self, x: Target, n_log: usize) {
        self.split_le(x, n_log);
    }

    /// Returns the first `num_low_bits` little-endian bits of `x`.
    /// Assume that F::ORDER = 2^EXP0 - 2^EXP1 + 1
    pub fn low_bits(
        &mut self,
        x: Target,
        num_low_bits: usize,
        are_noncanonical_indices_ok: bool,
        num_bits: usize,
    ) -> Vec<BoolTarget> {
        assert!(num_bits <= F::EXP0);
        let mut res = self.split_le(x, num_bits);
        if !are_noncanonical_indices_ok {
            let one = self.one();
            let (lo_bits, hi_bits) = res.split_at(F::EXP1);
            let lo_bits_sum = self.add_many(lo_bits.iter().map(|b| b.target));
            let hi_bits_sum = self.add_many(hi_bits.iter().map(|b| b.target));
            let hi_bits_sum_minus_exp0_plus_exp1 =
                self.add_const(hi_bits_sum, F::from_canonical_usize(F::EXP0 - F::EXP1));
            let y = self.inverse_or_zero(hi_bits_sum_minus_exp0_plus_exp1);
            let maybe_0 = self.arithmetic(
                F::one(),
                -F::one(),
                hi_bits_sum_minus_exp0_plus_exp1,
                y,
                one,
            );
            let must_be_0 = self.mul(maybe_0, lo_bits_sum);
            self.assert_zero(must_be_0);
        }
        res.truncate(num_low_bits);
        res
    }

    /// Returns `(a,b)` such that `x = a + 2^n_log * b` with `a < 2^n_log`.
    /// `x` is assumed to be range-checked for having `num_bits` bits.
    pub fn split_low_high(&mut self, x: Target, n_log: usize, num_bits: usize) -> (Target, Target) {
        let low = self.add_virtual_target();
        let high = self.add_virtual_target();

        self.add_simple_generator(LowHighGenerator {
            integer: x,
            n_log,
            low,
            high,
        });

        self.range_check(low, n_log);
        self.range_check(high, num_bits - n_log);

        let pow2 = self.constant(F::from_canonical_u64(1 << n_log));
        let comp_x = self.mul_add(high, pow2, low);
        self.connect(x, comp_x);

        (low, high)
    }

    pub fn assert_bool(&mut self, b: BoolTarget) {
        let z = self.mul_sub(b.target, b.target, b.target);
        let zero = self.zero();
        self.connect(z, zero);
    }
}

#[derive(Debug, Default)]
pub struct LowHighGenerator {
    integer: Target,
    n_log: usize,
    low: Target,
    high: Target,
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    SimpleGenerator<F, D, NUM_HASH_OUT_ELTS> for LowHighGenerator
{
    fn id(&self) -> String {
        "LowHighGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![self.integer]
    }

    fn run_once(&self, witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        let integer_value = witness.get_target(self.integer).as_canonical_u64();
        let low = integer_value & ((1 << self.n_log) - 1);
        let high = integer_value >> self.n_log;

        out_buffer.set_target(self.low, F::from_canonical_u64(low));
        out_buffer.set_target(self.high, F::from_canonical_u64(high));
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target(self.integer)?;
        dst.write_usize(self.n_log)?;
        dst.write_target(self.low)?;
        dst.write_target(self.high)
    }

    fn deserialize(
        src: &mut Buffer,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let integer = src.read_target()?;
        let n_log = src.read_usize()?;
        let low = src.read_target()?;
        let high = src.read_target()?;
        Ok(Self {
            integer,
            n_log,
            low,
            high,
        })
    }
}
