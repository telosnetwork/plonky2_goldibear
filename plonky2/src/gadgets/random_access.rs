#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use plonky2_field::types::HasExtension;

use crate::gates::random_access::RandomAccessGate;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::VerifierCircuitTarget;
use crate::util::log2_strict;

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>
{
    /// Checks that a `Target` matches a vector at a particular index.
    pub fn random_access(&mut self, access_index: Target, v: Vec<Target>) -> Target {
        let vec_size = v.len();
        let bits = log2_strict(vec_size);
        debug_assert!(vec_size > 0);
        if vec_size == 1 {
            return v[0];
        }
        let claimed_element = self.add_virtual_target();

        let dummy_gate = RandomAccessGate::<F, D>::new_from_config(&self.config, bits);
        let (row, copy) = self.find_slot(dummy_gate, &[], &[]);

        v.iter().enumerate().for_each(|(i, &val)| {
            self.connect(val, Target::wire(row, dummy_gate.wire_list_item(i, copy)));
        });
        self.connect(
            access_index,
            Target::wire(row, dummy_gate.wire_access_index(copy)),
        );
        self.connect(
            claimed_element,
            Target::wire(row, dummy_gate.wire_claimed_element(copy)),
        );

        claimed_element
    }

    /// Like `random_access`, but with `ExtensionTarget`s rather than simple `Target`s.
    pub fn random_access_extension(
        &mut self,
        access_index: Target,
        v: Vec<ExtensionTarget<D>>,
    ) -> ExtensionTarget<D> {
        let selected: Vec<_> = (0..D)
            .map(|i| self.random_access(access_index, v.iter().map(|et| et.0[i]).collect()))
            .collect();

        ExtensionTarget(selected.try_into().unwrap())
    }

    /// Like `random_access`, but with `HashOutTarget`s rather than simple `Target`s.
    pub fn random_access_hash(
        &mut self,
        access_index: Target,
        v: Vec<HashOutTarget<NUM_HASH_OUT_ELTS>>,
    ) -> HashOutTarget<NUM_HASH_OUT_ELTS> {
        let selected = core::array::from_fn(|i| {
            self.random_access(
                access_index,
                v.iter().map(|hash| hash.elements[i]).collect(),
            )
        });
        selected.into()
    }

    /// Like `random_access`, but with `MerkleCapTarget`s rather than simple `Target`s.
    pub fn random_access_merkle_cap(
        &mut self,
        access_index: Target,
        v: Vec<MerkleCapTarget<NUM_HASH_OUT_ELTS>>,
    ) -> MerkleCapTarget<NUM_HASH_OUT_ELTS> {
        let cap_size = v[0].0.len();
        assert!(v.iter().all(|cap| cap.0.len() == cap_size));

        let selected = (0..cap_size)
            .map(|i| self.random_access_hash(access_index, v.iter().map(|cap| cap.0[i]).collect()))
            .collect();
        MerkleCapTarget(selected)
    }

    /// Like `random_access`, but with `VerifierCircuitTarget`s rather than simple `Target`s.
    pub fn random_access_verifier_data(
        &mut self,
        access_index: Target,
        v: Vec<VerifierCircuitTarget<NUM_HASH_OUT_ELTS>>,
    ) -> VerifierCircuitTarget<NUM_HASH_OUT_ELTS> {
        let constants_sigmas_caps = v.iter().map(|vk| vk.constants_sigmas_cap.clone()).collect();
        let circuit_digests = v.iter().map(|vk| vk.circuit_digest).collect();
        let constants_sigmas_cap =
            self.random_access_merkle_cap(access_index, constants_sigmas_caps);
        let circuit_digest = self.random_access_hash(access_index, circuit_digests);
        VerifierCircuitTarget {
            constants_sigmas_cap,
            circuit_digest,
        }
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use p3_field::AbstractField;
    use plonky2_field::{GOLDILOCKS_EXTENSION_FIELD_DEGREE, GOLDILOCKS_NUM_HASH_OUT_ELTS};

    use super::*;
    use crate::field::types::Sample;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::plonk::verifier::verify;

    fn test_random_access_given_len(len_log: usize) -> Result<()> {
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        type FF = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::FE;
        let len = 1 << len_log;
        let config = CircuitConfig::standard_recursion_config_gl();
        let pw = PartialWitness::new();
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);
        let vec = FF::rand_vec(len);
        let v: Vec<_> = vec.iter().map(|x| builder.constant_extension(*x)).collect();

        for i in 0..len {
            let it = builder.constant(<F as AbstractField>::from_canonical_usize(i));
            let elem = builder.constant_extension(vec[i]);
            let res = builder.random_access_extension(it, v.clone());
            builder.connect_extension(elem, res);
        }

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;

        verify(proof, &data.verifier_only, &data.common)
    }

    #[test]
    fn test_random_access() -> Result<()> {
        for len_log in 1..3 {
            test_random_access_given_len(len_log)?;
        }
        Ok(())
    }
}
