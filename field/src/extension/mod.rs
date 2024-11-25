use alloc::vec::Vec;

use p3_field::{AbstractExtensionField, Field};

use crate::types::HasExtension;

/// Flatten the slice by sending every extension field element to its D-sized canonical representation.
pub fn flatten<F: HasExtension<D>, const D: usize>(l: &[F::Extension]) -> Vec<F>
where
    F: Field,
{
    l.iter().flat_map(|x| x.as_base_slice().to_vec()).collect()
}

/// Batch every D-sized chunks into extension field elements.
pub fn unflatten<F: HasExtension<D>, const D: usize>(l: &[F]) -> Vec<F::Extension>
where
    F: Field,
{
    debug_assert_eq!(l.len() % D, 0);
    l.chunks_exact(D)
        .map(|c| <F::Extension as AbstractExtensionField<F>>::from_base_slice(c))
        .collect()
}
