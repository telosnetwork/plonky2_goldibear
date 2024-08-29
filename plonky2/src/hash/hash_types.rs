#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use anyhow::ensure;
use p3_babybear::BabyBear;
use p3_field::{Field, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::field::types::Sample;
use crate::iop::target::Target;
use crate::plonk::config::GenericHashOut;

/// A prime order field with the features we need to use it as a base field in our argument system.
pub trait RichField: PrimeField64 + Sample + TwoAdicField {}

impl RichField for Goldilocks {}
impl RichField for BabyBear {}

//pub const NUM_HASH_OUT_ELTS: usize = 4;

/// Represents a ~256 bit hash output.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HashOut<F: Field, const NUM_HASH_OUT_ELTS: usize> {
    pub elements: [F; NUM_HASH_OUT_ELTS],
}

impl<F: Field, const NUM_HASH_OUT_ELTS: usize> HashOut<F, NUM_HASH_OUT_ELTS> {
    pub fn zero() -> Self {
        Self {
            elements: [F::zero(); NUM_HASH_OUT_ELTS],
        }
    }

    // TODO: Switch to a TryFrom impl.
    pub fn from_vec(elements: Vec<F>) -> Self {
        debug_assert!(elements.len() == NUM_HASH_OUT_ELTS);
        Self {
            elements: elements.try_into().unwrap(),
        }
    }

    pub fn from_partial(elements_in: &[F]) -> Self {
        let mut elements = [F::zero(); NUM_HASH_OUT_ELTS];
        elements[0..elements_in.len()].copy_from_slice(elements_in);
        Self { elements }
    }
}

impl<F: Field, const NUM_HASH_OUT_ELTS: usize> From<[F; NUM_HASH_OUT_ELTS]> for HashOut<F, NUM_HASH_OUT_ELTS> {
    fn from(elements: [F; NUM_HASH_OUT_ELTS]) -> Self {
        Self { elements }
    }
}

impl<F: Field, const NUM_HASH_OUT_ELTS: usize> TryFrom<&[F]> for HashOut<F, NUM_HASH_OUT_ELTS> {
    type Error = anyhow::Error;

    fn try_from(elements: &[F]) -> Result<Self, Self::Error> {
        ensure!(elements.len() == NUM_HASH_OUT_ELTS);
        Ok(Self {
            elements: elements.try_into().unwrap(),
        })
    }
}

impl<F, const NUM_HASH_OUT_ELTS: usize> Sample for HashOut<F, NUM_HASH_OUT_ELTS>
where
    F: Field + Sample,
{
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        Self {
            elements: (0..NUM_HASH_OUT_ELTS).map(|_|{F::sample(rng)}).collect::<Vec<F>>().try_into().unwrap(),
        }
    }
}

impl<F: RichField, const NUM_HASH_OUT_ELTS: usize> GenericHashOut<F> for HashOut<F, NUM_HASH_OUT_ELTS> {
    fn to_bytes(&self) -> Vec<u8> {
        self.elements
            .into_iter()
            .flat_map(|x| x.as_canonical_u64().to_le_bytes())
            .collect()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        HashOut {
            elements: bytes
                .chunks(8)
                .take(NUM_HASH_OUT_ELTS)
                .map(|x| F::from_canonical_u64(u64::from_le_bytes(x.try_into().unwrap())))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }

    fn to_vec(&self) -> Vec<F> {
        self.elements.to_vec()
    }
}

impl<F: Field, const NUM_HASH_OUT_ELTS: usize> Default for HashOut<F, NUM_HASH_OUT_ELTS> {
    fn default() -> Self {
        Self::zero()
    }
}

/// Represents a ~256 bit hash output.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HashOutTarget<const NUM_HASH_OUT_ELTS: usize> {
    pub elements: [Target; NUM_HASH_OUT_ELTS],
}

impl<const NUM_HASH_OUT_ELTS: usize> HashOutTarget<NUM_HASH_OUT_ELTS> {
    // TODO: Switch to a TryFrom impl.
    pub fn from_vec(elements: Vec<Target>) -> Self {
        debug_assert!(elements.len() == NUM_HASH_OUT_ELTS);
        Self {
            elements: elements.try_into().unwrap(),
        }
    }

    pub fn from_partial(elements_in: &[Target], zero: Target) -> Self {
        let mut elements = [zero; NUM_HASH_OUT_ELTS];
        elements[0..elements_in.len()].copy_from_slice(elements_in);
        Self { elements }
    }
}

impl<const NUM_HASH_OUT_ELTS: usize> From<[Target; NUM_HASH_OUT_ELTS]> for HashOutTarget<NUM_HASH_OUT_ELTS> {
    fn from(elements: [Target; NUM_HASH_OUT_ELTS]) -> Self {
        Self { elements }
    }
}

impl<const NUM_HASH_OUT_ELTS: usize> TryFrom<&[Target]> for HashOutTarget<NUM_HASH_OUT_ELTS> {
    type Error = anyhow::Error;

    fn try_from(elements: &[Target]) -> Result<Self, Self::Error> {
        ensure!(elements.len() == NUM_HASH_OUT_ELTS);
        Ok(Self {
            elements: elements.try_into().unwrap(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleCapTarget<const NUM_HASH_OUT_ELTS: usize>(pub Vec<HashOutTarget<NUM_HASH_OUT_ELTS>>);

/// Hash consisting of a byte array.
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct BytesHash<const N: usize>(pub [u8; N]);

impl<const N: usize> Sample for BytesHash<N> {
    #[inline]
    fn sample<R>(rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
    {
        let mut buf = [0; N];
        rng.fill_bytes(&mut buf);
        Self(buf)
    }
}

impl<F: RichField, const N: usize> GenericHashOut<F> for BytesHash<N> {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_vec()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.try_into().unwrap())
    }

    fn to_vec(&self) -> Vec<F> {
        self.0
            // Chunks of 7 bytes since 8 bytes would allow collisions.
            .chunks(7)
            .map(|bytes| {
                let mut arr = [0; 8];
                arr[..bytes.len()].copy_from_slice(bytes);
                F::from_canonical_u64(u64::from_le_bytes(arr))
            })
            .collect()
    }
}

impl<const N: usize> Serialize for BytesHash<N> {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        todo!()
    }
}

impl<'de, const N: usize> Deserialize<'de> for BytesHash<N> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        todo!()
    }
}
