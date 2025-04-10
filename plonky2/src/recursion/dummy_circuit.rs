#[cfg(not(feature = "std"))]
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};

use hashbrown::HashMap;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::HasExtension;
use plonky2_util::ceil_div_usize;

use crate::fri::proof::{FriProof, FriProofTarget};
use crate::gadgets::polynomial::PolynomialCoeffsExtTarget;
use crate::gates::noop::NoopGate;
use crate::hash::hash_types::{HashOutTarget, MerkleCapTarget, RichField};
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::generator::{GeneratedValues, SimpleGenerator};
use crate::iop::target::Target;
use crate::iop::witness::{PartialWitness, PartitionWitness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{
    CircuitData, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use crate::plonk::config::{AlgebraicHasher, GenericConfig, GenericHashOut, Hasher};
use crate::plonk::proof::{
    OpeningSet, OpeningSetTarget, Proof, ProofTarget, ProofWithPublicInputs,
    ProofWithPublicInputsTarget,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Creates a dummy proof which is suitable for use as a base proof in a cyclic recursion tree.
/// Such a base proof will not actually be verified, so most of its data is arbitrary. However, its
/// public inputs which encode the cyclic verification key must be set properly, and this method
/// takes care of that. It also allows the user to specify any other public inputs which should be
/// set in this base proof.
pub fn cyclic_base_proof<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    verifier_data: &VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
    mut nonzero_public_inputs: HashMap<usize, F>,
) -> ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>
where
    F: RichField + HasExtension<D>,

    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    C::Hasher: AlgebraicHasher<C::F, NUM_HASH_OUT_ELTS>,
{
    let pis_len = common_data.num_public_inputs;
    let cap_elements = common_data.config.fri_config.num_cap_elements();
    let start_vk_pis = pis_len - 4 - 4 * cap_elements;

    // Add the cyclic verifier data public inputs.
    nonzero_public_inputs.extend((start_vk_pis..).zip(verifier_data.circuit_digest.elements));
    for i in 0..cap_elements {
        let start = start_vk_pis + 4 + 4 * i;
        nonzero_public_inputs
            .extend((start..).zip(verifier_data.constants_sigmas_cap.0[i].elements));
    }

    // TODO: A bit wasteful to build a dummy circuit here. We could potentially use a proof that
    // just consists of zeros, apart from public inputs.
    dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(
        &dummy_circuit::<F, C, D, NUM_HASH_OUT_ELTS>(common_data),
        nonzero_public_inputs,
    )
    .unwrap()
}

/// Generate a proof for a dummy circuit. The `public_inputs` parameter let the caller specify
/// certain public inputs (identified by their indices) which should be given specific values.
/// The rest will default to zero.
pub(crate) fn dummy_proof<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    circuit: &CircuitData<F, C, D, NUM_HASH_OUT_ELTS>,
    nonzero_public_inputs: HashMap<usize, F>,
) -> anyhow::Result<ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>>
where
{
    let mut pw = PartialWitness::new();
    for i in 0..circuit.common.num_public_inputs {
        let pi = nonzero_public_inputs.get(&i).copied().unwrap_or_default();
        pw.set_target(circuit.prover_only.public_inputs[i], pi);
    }
    circuit.prove(pw)
}

/// Generate a circuit matching a given `CommonCircuitData`.
pub(crate) fn dummy_circuit<
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    const D: usize,
    const NUM_HASH_OUT_ELTS: usize,
>(
    common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
) -> CircuitData<F, C, D, NUM_HASH_OUT_ELTS>
where
{
    let config = common_data.config.clone();
    assert!(
        !common_data.config.zero_knowledge,
        "Degree calculation can be off if zero-knowledge is on."
    );

    // Number of `NoopGate`s to add to get a circuit of size `degree` in the end.
    // Need to account for public input hashing, a `PublicInputGate` and a `ConstantGate`.
    let degree = common_data.degree();
    let num_noop_gate = degree - ceil_div_usize(common_data.num_public_inputs, 8) - 2;

    let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config);
    for _ in 0..num_noop_gate {
        builder.add_gate(NoopGate, vec![]);
    }
    for gate in &common_data.gates {
        builder.add_gate_to_gate_set(gate.clone());
    }
    for _ in 0..common_data.num_public_inputs {
        builder.add_virtual_public_input();
    }

    let circuit = builder.build::<C>();
    assert_eq!(&circuit.common, common_data);
    circuit
}

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>
{
    pub(crate) fn dummy_proof_and_vk<
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension> + 'static,
    >(
        &mut self,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> anyhow::Result<(
        ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
        VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    )>
    where
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        let dummy_circuit = dummy_circuit::<F, C, D, NUM_HASH_OUT_ELTS>(common_data);
        let dummy_proof_with_pis =
            dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&dummy_circuit, HashMap::new())?;
        let dummy_proof_with_pis_target = self.add_virtual_proof_with_pis(common_data);
        let dummy_verifier_data_target =
            self.add_virtual_verifier_data(self.config.fri_config.cap_height);

        self.add_simple_generator(DummyProofGenerator {
            proof_with_pis_target: dummy_proof_with_pis_target.clone(),
            proof_with_pis: dummy_proof_with_pis,
            verifier_data_target: dummy_verifier_data_target.clone(),
            verifier_data: dummy_circuit.verifier_only,
        });

        Ok((dummy_proof_with_pis_target, dummy_verifier_data_target))
    }
}

#[derive(Debug)]
pub struct DummyProofGenerator<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize>
where
    F: RichField + HasExtension<D>,

    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
{
    pub(crate) proof_with_pis_target: ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
    pub(crate) proof_with_pis: ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    pub(crate) verifier_data_target: VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
    pub(crate) verifier_data: VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
}

impl<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize> Default
    for DummyProofGenerator<F, C, D, NUM_HASH_OUT_ELTS>
where
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
{
    fn default() -> Self {
        let proof_with_pis_target = ProofWithPublicInputsTarget {
            proof: ProofTarget {
                wires_cap: MerkleCapTarget(vec![]),
                plonk_zs_partial_products_cap: MerkleCapTarget(vec![]),
                quotient_polys_cap: MerkleCapTarget(vec![]),
                openings: OpeningSetTarget::default(),
                opening_proof: FriProofTarget {
                    commit_phase_merkle_caps: vec![],
                    query_round_proofs: vec![],
                    final_poly: PolynomialCoeffsExtTarget(vec![]),
                    pow_witness: Target::default(),
                },
            },
            public_inputs: vec![],
        };

        let proof_with_pis = ProofWithPublicInputs {
            proof: Proof {
                wires_cap: MerkleCap(vec![]),
                plonk_zs_partial_products_cap: MerkleCap(vec![]),
                quotient_polys_cap: MerkleCap(vec![]),
                openings: OpeningSet::default(),
                opening_proof: FriProof {
                    commit_phase_merkle_caps: vec![],
                    query_round_proofs: vec![],
                    final_poly: PolynomialCoeffs { coeffs: vec![] },
                    pow_witness: F::zero(),
                },
            },
            public_inputs: vec![],
        };

        let verifier_data_target = VerifierCircuitTarget {
            constants_sigmas_cap: MerkleCapTarget(vec![]),
            circuit_digest: HashOutTarget {
                elements: [Target::default(); NUM_HASH_OUT_ELTS],
            },
        };

        let verifier_data =
            VerifierOnlyCircuitData {
                constants_sigmas_cap: MerkleCap(vec![]),
                circuit_digest: <<C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::Hasher as Hasher<
                    C::F,
                >>::Hash::from_bytes(&vec![
                    0;
                    <<C as GenericConfig<
                        D,
                        NUM_HASH_OUT_ELTS,
                    >>::Hasher as Hasher<
                        C::F,
                    >>::HASH_SIZE
                ]),
            };

        Self {
            proof_with_pis_target,
            proof_with_pis,
            verifier_data_target,
            verifier_data,
        }
    }
}

impl<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize> SimpleGenerator<F, D, NUM_HASH_OUT_ELTS>
    for DummyProofGenerator<F, C, D, NUM_HASH_OUT_ELTS>
where
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension> + 'static,
    C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
{
    fn id(&self) -> String {
        "DummyProofGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        vec![]
    }

    fn run_once(&self, _witness: &PartitionWitness<F>, out_buffer: &mut GeneratedValues<F>) {
        out_buffer.set_proof_with_pis_target(&self.proof_with_pis_target, &self.proof_with_pis);
        out_buffer.set_verifier_data_target(&self.verifier_data_target, &self.verifier_data);
    }

    fn serialize(
        &self,
        dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<()> {
        dst.write_target_proof_with_public_inputs(&self.proof_with_pis_target)?;
        dst.write_proof_with_public_inputs(&self.proof_with_pis)?;
        dst.write_target_verifier_circuit(&self.verifier_data_target)?;
        dst.write_verifier_only_circuit_data(&self.verifier_data)
    }

    fn deserialize(
        src: &mut Buffer,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> IoResult<Self> {
        let proof_with_pis_target = src.read_target_proof_with_public_inputs()?;
        let proof_with_pis = src.read_proof_with_public_inputs(common_data)?;
        let verifier_data_target = src.read_target_verifier_circuit()?;
        let verifier_data = src.read_verifier_only_circuit_data()?;
        Ok(Self {
            proof_with_pis_target,
            proof_with_pis,
            verifier_data_target,
            verifier_data,
        })
    }
}
