use alloc::vec::Vec;

use p3_field::{AbstractExtensionField, Field};

/// Flatten the slice by sending every extension field element to its D-sized canonical representation.
pub fn flatten<F, EF: AbstractExtensionField<F>>(l: &[EF]) -> Vec<F>
where
    F: Field ,
{
    l.iter()
        .flat_map(|x| x.as_base_slice().to_vec())
        .collect()
}

/// Batch every D-sized chunks into extension field elements.
pub fn unflatten<F, EF: AbstractExtensionField<F>>(l: &[F]) -> Vec<EF>
where
    F: Field ,
{
    debug_assert_eq!(l.len() % EF::D, 0);
    l.chunks_exact(EF::D)
        .map(|c| EF::from_base_slice(c))
        .collect()
}
