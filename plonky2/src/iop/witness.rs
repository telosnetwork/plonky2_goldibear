#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use hashbrown::HashMap;
use itertools::{zip_eq, Itertools};
use p3_field::{AbstractExtensionField, Field};
use plonky2_field::types::HasExtension;

use crate::fri::structure::{FriOpenings, FriOpeningsTarget};
use crate::fri::witness_util::set_fri_proof_target;
use crate::hash::hash_types::{HashOut, HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::target::{BoolTarget, Target};
use crate::iop::wire::Wire;
use crate::plonk::circuit_data::{VerifierCircuitTarget, VerifierOnlyCircuitData};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::proof::{Proof, ProofTarget, ProofWithPublicInputs, ProofWithPublicInputsTarget};

pub trait WitnessWrite<F: Field> {
    fn set_target(&mut self, target: Target, value: F);

    fn set_hash_target<const NUM_HASH_OUT_ELTS: usize>(
        &mut self,
        ht: HashOutTarget<NUM_HASH_OUT_ELTS>,
        value: HashOut<F, NUM_HASH_OUT_ELTS>,
    ) {
        ht.elements
            .iter()
            .zip(value.elements)
            .for_each(|(&t, x)| self.set_target(t, x));
    }

    fn set_cap_target<H: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>, const NUM_HASH_OUT_ELTS: usize>(
        &mut self,
        ct: &MerkleCapTarget<NUM_HASH_OUT_ELTS>,
        value: &MerkleCap<F, H>,
    ) where
        F: RichField,
    {
        for (ht, h) in ct.0.iter().zip(&value.0) {
            self.set_hash_target(*ht, *h);
        }
    }

    fn set_extension_target<const D: usize>(&mut self, et: ExtensionTarget<D>, value: F::Extension)
    where
        F: RichField + HasExtension<D>,
    {
        self.set_target_arr(&et.0, value.as_base_slice());
    }

    fn set_target_arr(&mut self, targets: &[Target], values: &[F]) {
        zip_eq(targets, values).for_each(|(&target, &value)| self.set_target(target, value));
    }

    fn set_extension_targets<const D: usize>(
        &mut self,
        ets: &[ExtensionTarget<D>],
        values: &[F::Extension],
    ) where
        F: RichField + HasExtension<D>,
    {
        debug_assert_eq!(ets.len(), values.len());
        ets.iter()
            .zip(values)
            .for_each(|(&et, &v)| self.set_extension_target(et, v));
    }

    fn set_bool_target(&mut self, target: BoolTarget, value: bool) {
        self.set_target(target.target, F::from_bool(value))
    }

    /// Set the targets in a `ProofWithPublicInputsTarget` to their corresponding values in a
    /// `ProofWithPublicInputs`.
    fn set_proof_with_pis_target<
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        &mut self,
        proof_with_pis_target: &ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
        proof_with_pis: &ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    ) where
        F: RichField + HasExtension<D>,
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        let ProofWithPublicInputs {
            proof,
            public_inputs,
        } = proof_with_pis;
        let ProofWithPublicInputsTarget {
            proof: pt,
            public_inputs: pi_targets,
        } = proof_with_pis_target;

        // Set public inputs.
        for (&pi_t, &pi) in pi_targets.iter().zip_eq(public_inputs) {
            self.set_target(pi_t, pi);
        }

        self.set_proof_target(pt, proof);
    }

    /// Set the targets in a `ProofTarget` to their corresponding values in a `Proof`.
    fn set_proof_target<
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        &mut self,
        proof_target: &ProofTarget<D, NUM_HASH_OUT_ELTS>,
        proof: &Proof<F, C, D, NUM_HASH_OUT_ELTS>,
    ) where
        F: RichField + HasExtension<D>,
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        self.set_cap_target(&proof_target.wires_cap, &proof.wires_cap);
        self.set_cap_target(
            &proof_target.plonk_zs_partial_products_cap,
            &proof.plonk_zs_partial_products_cap,
        );
        self.set_cap_target(&proof_target.quotient_polys_cap, &proof.quotient_polys_cap);

        self.set_fri_openings(
            &proof_target.openings.to_fri_openings(),
            &proof.openings.to_fri_openings(),
        );

        set_fri_proof_target(self, &proof_target.opening_proof, &proof.opening_proof);
    }

    fn set_fri_openings<const D: usize>(
        &mut self,
        fri_openings_target: &FriOpeningsTarget<D>,
        fri_openings: &FriOpenings<F, D>,
    ) where
        F: RichField + HasExtension<D>,
    {
        for (batch_target, batch) in fri_openings_target
            .batches
            .iter()
            .zip_eq(&fri_openings.batches)
        {
            self.set_extension_targets(&batch_target.values, &batch.values);
        }
    }

    fn set_verifier_data_target<
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        &mut self,
        vdt: &VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
        vd: &VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
    ) where
        F: RichField + HasExtension<D>,
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        self.set_cap_target(&vdt.constants_sigmas_cap, &vd.constants_sigmas_cap);
        self.set_hash_target(vdt.circuit_digest, vd.circuit_digest);
    }

    fn set_wire(&mut self, wire: Wire, value: F) {
        self.set_target(Target::Wire(wire), value)
    }

    fn set_wires<W>(&mut self, wires: W, values: &[F])
    where
        W: IntoIterator<Item = Wire>,
    {
        // If we used itertools, we could use zip_eq for extra safety.
        for (wire, &value) in wires.into_iter().zip(values) {
            self.set_wire(wire, value);
        }
    }

    fn set_ext_wires<W, const D: usize>(&mut self, wires: W, value: F::Extension)
    where
        F: RichField + HasExtension<D>,

        W: IntoIterator<Item = Wire>,
    {
        self.set_wires(wires, value.as_base_slice());
    }

    fn extend<I: Iterator<Item = (Target, F)>>(&mut self, pairs: I) {
        for (t, v) in pairs {
            self.set_target(t, v);
        }
    }
}

/// A witness holds information on the values of targets in a circuit.
pub trait Witness<F: Field>: WitnessWrite<F> {
    fn try_get_target(&self, target: Target) -> Option<F>;

    fn get_target(&self, target: Target) -> F {
        self.try_get_target(target).unwrap()
    }

    fn get_targets(&self, targets: &[Target]) -> Vec<F> {
        targets.iter().map(|&t| self.get_target(t)).collect()
    }

    fn get_extension_target<const D: usize>(&self, et: ExtensionTarget<D>) -> F::Extension
    where
        F: RichField + HasExtension<D>,
    {
        F::Extension::from_base_slice(&self.get_targets(&et.to_target_array()))
    }

    fn get_extension_targets<const D: usize>(&self, ets: &[ExtensionTarget<D>]) -> Vec<F::Extension>
    where
        F: RichField + HasExtension<D>,
    {
        ets.iter()
            .map(|&et| self.get_extension_target(et))
            .collect()
    }

    fn get_bool_target(&self, target: BoolTarget) -> bool {
        let value = self.get_target(target.target);
        if value.is_zero() {
            return false;
        }
        if value.is_one() {
            return true;
        }
        panic!("not a bool")
    }

    fn get_hash_target<const NUM_HASH_OUT_ELTS: usize>(
        &self,
        ht: HashOutTarget<NUM_HASH_OUT_ELTS>,
    ) -> HashOut<F, NUM_HASH_OUT_ELTS> {
        HashOut {
            elements: self.get_targets(&ht.elements).try_into().unwrap(),
        }
    }

    fn get_merkle_cap_target<H, const NUM_HASH_OUT_ELTS: usize>(
        &self,
        cap_target: MerkleCapTarget<NUM_HASH_OUT_ELTS>,
    ) -> MerkleCap<F, H>
    where
        F: RichField,
        H: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        let cap = cap_target
            .0
            .iter()
            .map(|hash_target| self.get_hash_target(*hash_target))
            .collect();
        MerkleCap(cap)
    }

    fn get_wire(&self, wire: Wire) -> F {
        self.get_target(Target::Wire(wire))
    }

    fn try_get_wire(&self, wire: Wire) -> Option<F> {
        self.try_get_target(Target::Wire(wire))
    }

    fn contains(&self, target: Target) -> bool {
        self.try_get_target(target).is_some()
    }

    fn contains_all(&self, targets: &[Target]) -> bool {
        targets.iter().all(|&t| self.contains(t))
    }
}

#[derive(Clone, Debug)]
pub struct MatrixWitness<F: Field> {
    pub(crate) wire_values: Vec<Vec<F>>,
}

impl<F: Field> MatrixWitness<F> {
    pub fn get_wire(&self, gate: usize, input: usize) -> F {
        self.wire_values[input][gate]
    }
}

#[derive(Clone, Debug, Default)]
pub struct PartialWitness<F: Field> {
    pub target_values: HashMap<Target, F>,
}

impl<F: Field> PartialWitness<F> {
    pub fn new() -> Self {
        Self {
            target_values: HashMap::new(),
        }
    }
}

impl<F: Field> WitnessWrite<F> for PartialWitness<F> {
    fn set_target(&mut self, target: Target, value: F) {
        let opt_old_value = self.target_values.insert(target, value);
        if let Some(old_value) = opt_old_value {
            assert_eq!(
                value, old_value,
                "Target {target:?} was set twice with different values: {old_value} != {value}"
            );
        }
    }
}

impl<F: Field> Witness<F> for PartialWitness<F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        self.target_values.get(&target).copied()
    }
}

/// `PartitionWitness` holds a disjoint-set forest of the targets respecting a circuit's copy constraints.
/// The value of a target is defined to be the value of its root in the forest.
#[derive(Clone, Debug)]
pub struct PartitionWitness<'a, F: Field> {
    pub values: Vec<Option<F>>,
    pub representative_map: &'a [usize],
    pub num_wires: usize,
    pub degree: usize,
}

impl<'a, F: Field> PartitionWitness<'a, F> {
    pub fn new(num_wires: usize, degree: usize, representative_map: &'a [usize]) -> Self {
        Self {
            values: vec![None; representative_map.len()],
            representative_map,
            num_wires,
            degree,
        }
    }

    /// Set a `Target`. On success, returns the representative index of the newly-set target. If the
    /// target was already set, returns `None`.
    pub fn set_target_returning_rep(&mut self, target: Target, value: F) -> Option<usize> {
        let rep_index = self.representative_map[self.target_index(target)];
        let rep_value = &mut self.values[rep_index];
        if let Some(old_value) = *rep_value {
            assert_eq!(
                value, old_value,
                "Partition containing {target:?} was set twice with different values: {old_value} != {value}"
            );
            None
        } else {
            *rep_value = Some(value);
            Some(rep_index)
        }
    }

    pub(crate) fn target_index(&self, target: Target) -> usize {
        target.index(self.num_wires, self.degree)
    }

    pub fn full_witness(self) -> MatrixWitness<F> {
        let mut wire_values = vec![vec![F::zero(); self.degree]; self.num_wires];
        for i in 0..self.degree {
            for j in 0..self.num_wires {
                let t = Target::Wire(Wire { row: i, column: j });
                if let Some(x) = self.try_get_target(t) {
                    wire_values[j][i] = x;
                }
            }
        }

        MatrixWitness { wire_values }
    }
}

impl<F: Field> WitnessWrite<F> for PartitionWitness<'_, F> {
    fn set_target(&mut self, target: Target, value: F) {
        self.set_target_returning_rep(target, value);
    }
}

impl<F: Field> Witness<F> for PartitionWitness<'_, F> {
    fn try_get_target(&self, target: Target) -> Option<F> {
        let rep_index = self.representative_map[self.target_index(target)];
        self.values[rep_index]
    }
}
