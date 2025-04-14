#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::mem::size_of;

use anyhow::ensure;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use static_assertions::const_assert;

use crate::field::types::Sample;
use crate::iop::target::Target;
use crate::plonk::config::GenericHashOut;
use crate::util::serialization::{IoResult, Read, Write};

/// A prime order field with the features we need to use it as a base field in our argument system.
/// We assume F::ORDER = 2^EXP0 - 2^EXP1 + 1, where 64 >= EXP0 > EXP1 >= 1.
pub trait RichField: PrimeField64 + Sample + TwoAdicField {
    const EXP0: usize;
    const EXP1: usize;
    const NUM_HASH_OUT_ELTS: usize;
    fn read_from_buffer<T: Read + ?Sized>(reader: &mut T) -> IoResult<Self>;
    fn write_to_buffer<T: Write + ?Sized>(&self, writer: &mut T) -> IoResult<()>;
    fn to_bytes(&self) -> Vec<u8>;
    fn hash_out_elements_from_bytes(bytes: &[u8]) -> Vec<Self>;
}

pub const GOLDILOCKS_NUM_HASH_OUT_ELTS: usize = 4;

impl RichField for Goldilocks {
    const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;

    fn read_from_buffer<T: Read + ?Sized>(reader: &mut T) -> IoResult<Self> {
        let mut buf = [0; size_of::<u64>()];
        reader.read_exact(&mut buf)?;
        Ok(Goldilocks::from_canonical_u64(u64::from_le_bytes(buf)))
    }

    fn write_to_buffer<T: Write + ?Sized>(&self, writer: &mut T) -> IoResult<()> {
        writer.write_all(&self.as_canonical_u64().to_le_bytes())
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.as_canonical_u64().to_le_bytes().to_vec()
    }

    fn hash_out_elements_from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks(8)
            .take(Self::NUM_HASH_OUT_ELTS)
            .map(|x| Goldilocks::from_canonical_u64(u64::from_le_bytes(x.try_into().unwrap())))
            .collect::<Vec<_>>()
    }

    const EXP0: usize = 64;

    const EXP1: usize = 32;
}
const_assert!(
    Goldilocks::ORDER_U64
        == ((1u128 << Goldilocks::EXP0) - (1u128 << Goldilocks::EXP1) + 1u128) as u64
);

pub const BABYBEAR_NUM_HASH_OUT_ELTS: usize = 8;

impl RichField for BabyBear {
    const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;

    fn read_from_buffer<T: Read + ?Sized>(reader: &mut T) -> IoResult<Self> {
        let mut buf = [0; size_of::<u32>()];
        reader.read_exact(&mut buf)?;
        Ok(BabyBear::from_canonical_u32(u32::from_le_bytes(buf)))
    }

    fn write_to_buffer<T: Write + ?Sized>(&self, writer: &mut T) -> IoResult<()> {
        writer.write_all(&self.as_canonical_u32().to_le_bytes())
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.as_canonical_u32().to_le_bytes().to_vec()
    }

    fn hash_out_elements_from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks(4)
            .take(Self::NUM_HASH_OUT_ELTS)
            .map(|x| {
                BabyBear::from_canonical_u32(
                    u32::from_le_bytes(x.try_into().unwrap()) % Self::ORDER_U32,
                )
            })
            .collect::<Vec<_>>()
    }

    const EXP0: usize = 31;

    const EXP1: usize = 27;
}
const_assert!(
    BabyBear::ORDER_U64 == ((1u128 << BabyBear::EXP0) - (1u128 << BabyBear::EXP1) + 1u128) as u64
);
//pub const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;

/// Represents a ~256 bit hash output.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct HashOut<F: Field, const NUM_HASH_OUT_ELTS: usize> {
    #[serde(with = "generic_arrays")]
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

impl<F: Field, const NUM_HASH_OUT_ELTS: usize> From<[F; NUM_HASH_OUT_ELTS]>
    for HashOut<F, NUM_HASH_OUT_ELTS>
{
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
            elements: (0..NUM_HASH_OUT_ELTS)
                .map(|_| F::sample(rng))
                .collect::<Vec<F>>()
                .try_into()
                .unwrap(),
        }
    }
}

impl<F: RichField, const NUM_HASH_OUT_ELTS: usize> GenericHashOut<F>
    for HashOut<F, NUM_HASH_OUT_ELTS>
{
    fn to_bytes(&self) -> Vec<u8> {
        self.elements
            .into_iter()
            .flat_map(|x| x.to_bytes())
            .collect()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        HashOut {
            elements: F::hash_out_elements_from_bytes(bytes).try_into().unwrap(),
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

impl<const NUM_HASH_OUT_ELTS: usize> From<[Target; NUM_HASH_OUT_ELTS]>
    for HashOutTarget<NUM_HASH_OUT_ELTS>
{
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
pub struct MerkleCapTarget<const NUM_HASH_OUT_ELTS: usize>(
    pub Vec<HashOutTarget<NUM_HASH_OUT_ELTS>>,
);

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

mod generic_arrays {
    #[cfg(not(feature = "std"))]
    use alloc::{format, vec::Vec};
    use core::marker::PhantomData;

    use serde::de::{SeqAccess, Visitor};
    use serde::ser::SerializeTuple;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
    where
        T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut data = Vec::with_capacity(N);
            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}
