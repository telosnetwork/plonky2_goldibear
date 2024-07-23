use alloc::vec::Vec;

use num::bigint::BigUint;
use p3_field::Field;

/// Finds a set of shifts that result in unique cosets for the multiplicative subgroup of size
/// `2^subgroup_bits`.
pub fn get_unique_coset_shifts<F: Field>(subgroup_size: usize, num_shifts: usize) -> Vec<F> {
    // From Lagrange's theorem.
    let num_cosets = (F::order() - 1u32) / (subgroup_size as u32);
    assert!(
        BigUint::from(num_shifts) <= num_cosets,
        "The subgroup does not have enough distinct cosets"
    );

    // Let g be a generator of the entire multiplicative group. Let n be the order of the subgroup.
    // The subgroup can be written as <g^(|F*| / n)>. We can use g^0, ..., g^(num_shifts - 1) as our
    // shifts, since g^i <g^(|F*| / n)> are distinct cosets provided i < |F*| / n, which we checked.
    F::generator().powers().take(num_shifts).collect()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_goldilocks::Goldilocks;
    extern crate std;
    use std::collections::HashSet;

    use p3_field::TwoAdicField;

    use crate::cosets::get_unique_coset_shifts;

    fn cyclic_subgroup_coset_known_order<F: TwoAdicField>(
        generator: F,
        shift: F,
        order: usize,
    ) -> Vec<F> {
        let subgroup: Vec<F> = generator.powers().take(order).collect();
        subgroup.into_iter().map(|x| x * shift).collect()
    }
    #[test]
    fn distinct_cosets() {
        type F = Goldilocks;
        const SUBGROUP_BITS: usize = 5;
        const NUM_SHIFTS: usize = 50;

        let generator = F::two_adic_generator(SUBGROUP_BITS);
        let subgroup_size = 1 << SUBGROUP_BITS;

        let shifts = get_unique_coset_shifts::<F>(subgroup_size, NUM_SHIFTS);

        let mut union = HashSet::new();
        for shift in shifts {
            let coset = cyclic_subgroup_coset_known_order(generator, shift, subgroup_size);
            assert!(
                coset.into_iter().all(|x| union.insert(x)),
                "Duplicate element!"
            );
        }
    }
}
