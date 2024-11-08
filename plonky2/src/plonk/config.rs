//! Hashing configuration to be used when building a circuit.
//!
//! This module defines a [`Hasher`] trait as well as its recursive
//! counterpart [`AlgebraicHasher`] for in-circuit hashing. It also
//! provides concrete configurations, one fully recursive leveraging
//! the Poseidon hash function both internally and natively, and one
//! mixing Poseidon internally and truncated Keccak externally.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use p3_baby_bear::BabyBear;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_field::extension::BinomialExtensionField;
use p3_goldilocks::Goldilocks;
use serde::de::DeserializeOwned;
use serde::Serialize;

use plonky2_field::types::HasExtension;

use crate::hash::hash_types::{HashOut, HashOutTarget, RichField};
use crate::hash::hashing::PlonkyPermutation;
use crate::hash::keccak::KeccakHash;
use crate::hash::poseidon2_babybear::Poseidon2BabyBearHash;
use crate::hash::poseidon_goldilocks::Poseidon64Hash;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;

pub trait GenericHashOut<F: Field>:
    Copy + Clone + Debug + Eq + PartialEq + Send + Sync + Serialize + DeserializeOwned
{
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;

    fn to_vec(&self) -> Vec<F>;
}

/// Trait for hash functions.
pub trait Hasher<F: RichField>: Sized + Copy + Debug + Eq + PartialEq {
    /// Size of `Hash` in bytes.
    const HASH_SIZE: usize;

    /// Hash Output
    type Hash: GenericHashOut<F>;

    /// Permutation used in the sponge construction.
    type Permutation: PlonkyPermutation<F>;

    /// Hash a message without any padding step. Note that this can enable length-extension attacks.
    /// However, it is still collision-resistant in cases where the input has a fixed length.
    fn hash_no_pad(input: &[F]) -> Self::Hash;

    /// Pad the message using the `pad10*1` rule, then hash it.
    fn hash_pad(input: &[F]) -> Self::Hash {
        let mut padded_input = input.to_vec();
        padded_input.push(F::one());
        while (padded_input.len() + 1) % Self::Permutation::RATE != 0 {
            padded_input.push(F::zero());
        }
        padded_input.push(F::one());
        Self::hash_no_pad(&padded_input)
    }

    /// Hash the slice if necessary to reduce its length to ~256 bits. If it already fits, this is a
    /// no-op.
    fn hash_or_noop(inputs: &[F]) -> Self::Hash {
        if inputs.len() <= F::NUM_HASH_OUT_ELTS {
            let mut inputs_bytes = vec![0u8; Self::HASH_SIZE];
            let mut idx = 0;
            for el in inputs {
                for b in el.to_bytes() {
                    inputs_bytes[idx] = b;
                    idx += 1;
                }
            }
            Self::Hash::from_bytes(&inputs_bytes)
        } else {
            Self::hash_no_pad(inputs)
        }
    }

    fn two_to_one(left: Self::Hash, right: Self::Hash) -> Self::Hash;
}

/// Trait for algebraic hash functions, built from a permutation using the sponge construction.
pub trait AlgebraicHasher<F: RichField, const NUM_HASH_OUT_ELTS: usize>:
    Hasher<F, Hash = HashOut<F, NUM_HASH_OUT_ELTS>>
{
    type AlgebraicPermutation: PlonkyPermutation<Target>;

    /// Circuit to conditionally swap two chunks of the inputs (useful in verifying Merkle proofs),
    /// then apply the permutation.
    fn permute_swapped<const D: usize>(
        inputs: Self::AlgebraicPermutation,
        swap: BoolTarget,
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Self::AlgebraicPermutation
    where
        F: RichField + HasExtension<D>,
        F::Extension: TwoAdicField;

    fn hash_or_noop_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        inputs: Vec<Target>,
    ) -> HashOutTarget<NUM_HASH_OUT_ELTS>
    where
        F: RichField + HasExtension<D>,

    {
        let zero = builder.zero();
        if inputs.len() <= NUM_HASH_OUT_ELTS {
            HashOutTarget::from_partial(&inputs, zero)
        } else {
            Self::hash_n_to_hash_no_pad_circuit::<D>(builder, inputs)
        }
    }

    fn hash_n_to_hash_no_pad_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        inputs: Vec<Target>,
    ) -> HashOutTarget<NUM_HASH_OUT_ELTS>
    where
        F: RichField + HasExtension<D>,

    {
        HashOutTarget::from_vec(Self::hash_n_to_m_no_pad_circuit::<D>(
            builder,
            inputs,
            NUM_HASH_OUT_ELTS,
        ))
    }

    fn hash_n_to_m_no_pad_circuit<const D: usize>(
        builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
        inputs: Vec<Target>,
        num_outputs: usize,
    ) -> Vec<Target>
    where
        F: RichField + HasExtension<D>,

    {
        let zero = builder.zero();
        let mut state = Self::AlgebraicPermutation::new(core::iter::repeat(zero));

        // Absorb all input chunks.
        for input_chunk in inputs.chunks(Self::AlgebraicPermutation::RATE) {
            // Overwrite the first r elements with the inputs. This differs from a standard sponge,
            // where we would xor or add in the inputs. This is a well-known variant, though,
            // sometimes called "overwrite mode".
            state.set_from_slice(input_chunk, 0);
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

/// Generic configuration trait.
pub trait GenericConfig<const D: usize, const NUM_HASH_OUT_ELTS: usize>:
    Debug + Clone + Sync + Sized + Send + Eq + PartialEq
{
    /// Main field.
    type F: RichField + HasExtension<D>;
    /// Field extension of degree D of the main field.
    type FE: ExtensionField<Self::F>;
    /// Hash function used for building Merkle trees.
    type Hasher: Hasher<Self::F>;
    /// Algebraic hash function used for the challenger and hashing public inputs.
    type InnerHasher: AlgebraicHasher<Self::F, NUM_HASH_OUT_ELTS>;
}

/// Configuration using Poseidon over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct PoseidonGoldilocksConfig;
impl GenericConfig<2, 4> for PoseidonGoldilocksConfig {
    type F = Goldilocks;
    type FE = BinomialExtensionField<Self::F, 2>;
    type Hasher = Poseidon64Hash;
    type InnerHasher = Poseidon64Hash;
}

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Serialize)]
pub struct Poseidon2BabyBearConfig;
impl GenericConfig<4, 8> for Poseidon2BabyBearConfig {
    type F = BabyBear;
    type FE = BinomialExtensionField<Self::F, 4>;
    type Hasher = Poseidon2BabyBearHash;
    type InnerHasher = Poseidon2BabyBearHash;
}

/// Configuration using truncated Keccak over the Goldilocks field.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct KeccakGoldilocksConfig;
impl GenericConfig<2, 4> for KeccakGoldilocksConfig {
    type F = Goldilocks;
    type FE = BinomialExtensionField<Self::F, 2>;
    type Hasher = KeccakHash<25>;
    type InnerHasher = Poseidon64Hash;
}
