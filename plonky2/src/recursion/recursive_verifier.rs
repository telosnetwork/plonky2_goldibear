#[cfg(not(feature = "std"))]
use alloc::vec;

use plonky2_field::types::HasExtension;

use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CommonCircuitData, VerifierCircuitTarget};
use crate::plonk::config::{AlgebraicHasher, GenericConfig};
use crate::plonk::plonk_common::salt_size;
use crate::plonk::proof::{
    OpeningSetTarget, ProofChallengesTarget, ProofTarget, ProofWithPublicInputsTarget,
};
use crate::plonk::vanishing_poly::eval_vanishing_poly_circuit;
use crate::plonk::vars::EvaluationTargets;
use crate::util::reducing::ReducingFactorTarget;
use crate::with_context;

impl<F: RichField + HasExtension<D>, const D: usize, const NUM_HASH_OUT_ELTS: usize>
    CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>
{
    /// Recursively verifies an inner proof.
    pub fn verify_proof<C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>>(
        &mut self,
        proof_with_pis: &ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS>,
        inner_verifier_data: &VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
        inner_common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) where
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        assert_eq!(
            proof_with_pis.public_inputs.len(),
            inner_common_data.num_public_inputs
        );
        let public_inputs_hash =
            self.hash_n_to_hash_no_pad::<C::InnerHasher>(proof_with_pis.public_inputs.clone());
        let challenges = proof_with_pis.get_challenges::<F, C>(
            self,
            public_inputs_hash,
            inner_verifier_data.circuit_digest,
            inner_common_data,
        );

        self.verify_proof_with_challenges::<C>(
            &proof_with_pis.proof,
            public_inputs_hash,
            challenges,
            inner_verifier_data,
            inner_common_data,
        );
    }

    /// Recursively verifies an inner proof.
    fn verify_proof_with_challenges<
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    >(
        &mut self,
        proof: &ProofTarget<D, NUM_HASH_OUT_ELTS>,
        public_inputs_hash: HashOutTarget<NUM_HASH_OUT_ELTS>,
        challenges: ProofChallengesTarget<D>,
        inner_verifier_data: &VerifierCircuitTarget<NUM_HASH_OUT_ELTS>,
        inner_common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) where
        C::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        let one = self.one_extension();

        let local_constants = &proof.openings.constants;
        let local_wires = &proof.openings.wires;
        let vars = EvaluationTargets {
            local_constants,
            local_wires,
            public_inputs_hash: &public_inputs_hash,
        };
        let local_zs = &proof.openings.plonk_zs;
        let next_zs = &proof.openings.plonk_zs_next;
        let local_lookup_zs = &proof.openings.lookup_zs;
        let next_lookup_zs = &proof.openings.next_lookup_zs;
        let s_sigmas = &proof.openings.plonk_sigmas;
        let partial_products = &proof.openings.partial_products;

        let zeta_pow_deg =
            self.exp_power_of_2_extension(challenges.plonk_zeta, inner_common_data.degree_bits());
        let vanishing_polys_zeta = with_context!(
            self,
            "evaluate the vanishing polynomial at our challenge point, zeta.",
            eval_vanishing_poly_circuit::<F, D, NUM_HASH_OUT_ELTS>(
                self,
                inner_common_data,
                challenges.plonk_zeta,
                zeta_pow_deg,
                vars,
                local_zs,
                next_zs,
                local_lookup_zs,
                next_lookup_zs,
                partial_products,
                s_sigmas,
                &challenges.plonk_betas,
                &challenges.plonk_gammas,
                &challenges.plonk_alphas,
                &challenges.plonk_deltas,
            )
        );

        with_context!(self, "check vanishing and quotient polynomials.", {
            let quotient_polys_zeta = &proof.openings.quotient_polys;
            let mut scale = ReducingFactorTarget::new(zeta_pow_deg);
            let z_h_zeta = self.sub_extension(zeta_pow_deg, one);
            for (i, chunk) in quotient_polys_zeta
                .chunks(inner_common_data.quotient_degree_factor)
                .enumerate()
            {
                let recombined_quotient = scale.reduce(chunk, self);
                let computed_vanishing_poly = self.mul_extension(z_h_zeta, recombined_quotient);
                self.connect_extension(vanishing_polys_zeta[i], computed_vanishing_poly);
            }
        });

        let merkle_caps = &[
            inner_verifier_data.constants_sigmas_cap.clone(),
            proof.wires_cap.clone(),
            proof.plonk_zs_partial_products_cap.clone(),
            proof.quotient_polys_cap.clone(),
        ];

        let fri_instance = inner_common_data.get_fri_instance_target(self, challenges.plonk_zeta);
        with_context!(
            self,
            "verify FRI proof",
            self.verify_fri_proof::<C>(
                &fri_instance,
                &proof.openings.to_fri_openings(),
                &challenges.fri_challenges,
                merkle_caps,
                &proof.opening_proof,
                &inner_common_data.fri_params,
            )
        );
    }

    pub fn add_virtual_proof_with_pis(
        &mut self,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> ProofWithPublicInputsTarget<D, NUM_HASH_OUT_ELTS> {
        let proof = self.add_virtual_proof(common_data);
        let public_inputs = self.add_virtual_targets(common_data.num_public_inputs);
        ProofWithPublicInputsTarget {
            proof,
            public_inputs,
        }
    }

    fn add_virtual_proof(
        &mut self,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> ProofTarget<D, NUM_HASH_OUT_ELTS> {
        let config = &common_data.config;
        let fri_params = &common_data.fri_params;
        let cap_height = fri_params.config.cap_height;

        let salt = salt_size(common_data.fri_params.hiding);
        let num_leaves_per_oracle = &mut vec![
            common_data.num_preprocessed_polys(),
            config.num_wires + salt,
            common_data.num_zs_partial_products_polys() + common_data.num_all_lookup_polys() + salt,
        ];

        if common_data.num_quotient_polys() > 0 {
            num_leaves_per_oracle.push(common_data.num_quotient_polys() + salt);
        }

        ProofTarget {
            wires_cap: self.add_virtual_cap(cap_height),
            plonk_zs_partial_products_cap: self.add_virtual_cap(cap_height),
            quotient_polys_cap: self.add_virtual_cap(cap_height),
            openings: self.add_opening_set(common_data),
            opening_proof: self.add_virtual_fri_proof(num_leaves_per_oracle, fri_params),
        }
    }

    fn add_opening_set(
        &mut self,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> OpeningSetTarget<D> {
        let config = &common_data.config;
        let num_challenges = config.num_challenges;
        let total_partial_products = num_challenges * common_data.num_partial_products;
        let has_lookup = common_data.num_lookup_polys != 0;
        let num_lookups = if has_lookup {
            common_data.num_all_lookup_polys()
        } else {
            0
        };
        OpeningSetTarget {
            constants: self.add_virtual_extension_targets(common_data.num_constants),
            plonk_sigmas: self.add_virtual_extension_targets(config.num_routed_wires),
            wires: self.add_virtual_extension_targets(config.num_wires),
            plonk_zs: self.add_virtual_extension_targets(num_challenges),
            plonk_zs_next: self.add_virtual_extension_targets(num_challenges),
            lookup_zs: self.add_virtual_extension_targets(num_lookups),
            next_lookup_zs: self.add_virtual_extension_targets(num_lookups),
            partial_products: self.add_virtual_extension_targets(total_partial_products),
            quotient_polys: self.add_virtual_extension_targets(common_data.num_quotient_polys()),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{sync::Arc, vec};
    #[cfg(feature = "std")]
    use std::sync::Arc;

    use anyhow::Result;
    use itertools::Itertools;
    use log::{info, Level};
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractField, PrimeField64};
    use p3_goldilocks::Goldilocks;
    use plonky2_field::{
        BABYBEAR_EXTENSION_FIELD_DEGREE, BABYBEAR_NUM_HASH_OUT_ELTS,
        GOLDILOCKS_EXTENSION_FIELD_DEGREE, GOLDILOCKS_NUM_HASH_OUT_ELTS,
    };
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use crate::fri::reduction_strategies::FriReductionStrategy;
    use crate::fri::FriConfig;
    use crate::gadgets::lookup::{OTHER_TABLE, TIP5_TABLE};
    use crate::gates::gate::GateRef;
    use crate::gates::lookup_table::LookupTable;
    use crate::gates::noop::NoopGate;
    use crate::gates::poseidon2_babybear::Poseidon2BabyBearGate;
    use crate::gates::poseidon_goldilocks::PoseidonGate;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_data::{CircuitConfig, VerifierOnlyCircuitData};
    use crate::plonk::config::{
        KeccakGoldilocksConfig, Poseidon2BabyBearConfig, PoseidonGoldilocksConfig,
    };
    use crate::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
    use crate::plonk::prover::prove;
    use crate::plonk::verifier::verify;
    use crate::recursion::regression_test_data::{
        RECURSIVE_VERIFIER_GL_COMMON_DATA, RECURSIVE_VERIFIER_GL_PROOF,
        RECURSIVE_VERIFIER_GL_VERIFIER_DATA,
    };
    use crate::util::proving_process_info::ProvingProcessInfo;
    use crate::util::serialization::DefaultGateSerializer;

    #[test]
    fn test_recursive_verifier_gl() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let config = CircuitConfig::standard_recursion_zk_config_gl();

        let (proof, vd, common_data) = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config, 4_000)?;
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    /// This test is strictly related to [`test_recursive_verifier_gl`], but it is meant to verify that the recursive verifier circuit representation
    /// (e.g. the circuit digest) doesn't change unexpectedly.
    /// For this reason, it uses the verifier/common data and the proof produced by the previous test and uses them validate the proof.
    /// It is particularly useful to run this test also for the WASM architecture to prevent compatibility issues as the one fixed with `451536ea`.
    #[test]
    #[wasm_bindgen_test]
    fn test_recursive_verifier_gl_regression() -> Result<()> {
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        // Deserialize common data
        let common_data = CommonCircuitData::<F, D, NUM_HASH_OUT_ELTS>::from_bytes(
            RECURSIVE_VERIFIER_GL_COMMON_DATA.to_vec(),
            &DefaultGateSerializer,
        )
        .unwrap();

        // Deserialize verifier data
        let verifier_data = VerifierOnlyCircuitData::<
            PoseidonGoldilocksConfig,
            GOLDILOCKS_EXTENSION_FIELD_DEGREE,
            GOLDILOCKS_NUM_HASH_OUT_ELTS,
        >::from_bytes(RECURSIVE_VERIFIER_GL_VERIFIER_DATA.to_vec())
        .unwrap();

        // Deserialize the proof
        let proof = ProofWithPublicInputs::<F, C, D, NUM_HASH_OUT_ELTS>::from_bytes(
            RECURSIVE_VERIFIER_GL_PROOF.to_vec(),
            &common_data,
        )
        .unwrap();

        // Verify the proof
        verify::<F, C, D, NUM_HASH_OUT_ELTS>(proof, &verifier_data, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_bb() -> Result<()> {
        init_logger();
        const D: usize = BABYBEAR_EXTENSION_FIELD_DEGREE;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let config = CircuitConfig::standard_recursion_zk_config_bb();

        let (proof, vd, common_data) = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config, 4_000)?;
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_one_lookup() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let config = CircuitConfig::standard_recursion_zk_config_gl();

        let (proof, vd, common_data) =
            dummy_lookup_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config, 10)?;
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_two_luts() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let config = CircuitConfig::standard_recursion_config_gl();

        let (proof, vd, common_data) = dummy_two_luts_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config)?;
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_too_many_rows() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;
        let config = CircuitConfig::standard_recursion_config_gl();

        let (proof, vd, common_data) =
            dummy_too_many_rows_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config)?;
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_recursive_verifier_gl() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let config = CircuitConfig::standard_recursion_config_gl();

        // Start with a degree 2^14 proof
        let (proof, vd, common_data) = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config, 16_000)?;
        assert_eq!(common_data.degree_bits(), 14);
        assert_eq!(
            vd.circuit_digest.elements,
            [
                40392719057770864,
                9247014007799316719,
                17436525775388713746,
                10078131498506678571
            ]
            .map(F::from_canonical_u64)
        );
        // Shrink it to 2^13.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            Some(13),
            false,
            false,
        )?;
        assert_eq!(common_data.degree_bits(), 13);
        assert_eq!(
            vd.circuit_digest.elements,
            [
                13853083556319302030,
                7032558940778999676,
                15988113482817466578,
                3553937068079282312
            ]
            .map(F::from_canonical_u64)
        );
        // Shrink it to 2^12.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);
        assert_eq!(
            vd.circuit_digest.elements,
            [
                5000568515610536070,
                17514281206585518617,
                17334557576105184524,
                4950795566141018980
            ]
            .map(F::from_canonical_u64)
        );

        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_recursive_verifier_bb() -> Result<()> {
        init_logger();
        const D: usize = BABYBEAR_EXTENSION_FIELD_DEGREE;
        type C = Poseidon2BabyBearConfig;
        const NUM_HASH_OUT_ELTS: usize = BABYBEAR_NUM_HASH_OUT_ELTS;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let config = CircuitConfig::standard_recursion_config_bb_wide();
        info!(" ****************  Generating Dummy Proof ****************");
        // Start with a degree 2^14 proof
        let (proof, vd, common_data) = dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&config, 16_000)?;
        assert_eq!(common_data.degree_bits(), 14);

        info!(" ****************  Generating 1st Recursive Proof ****************");
        // Shrink it to 2^13.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            Some(13),
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 13);

        info!(" ****************  Generating 2nd Recursive Proof ****************");
        // Shrink it to 2^12.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        info!(" ****************  Generating 3rd Recursive Proof ****************");
        // Shrink it to 2^12.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }
    /// Creates a chain of recursive proofs where the last proof is made as small as reasonably
    /// possible, using a high rate, high PoW bits, etc.
    #[test]
    #[ignore]
    fn test_size_optimized_recursion() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        type C = PoseidonGoldilocksConfig;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type KC = KeccakGoldilocksConfig;
        type F = <C as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let standard_config = CircuitConfig::standard_recursion_config_gl();

        // An initial dummy proof.
        let (proof, vd, common_data) =
            dummy_proof::<F, C, D, NUM_HASH_OUT_ELTS>(&standard_config, 4_000)?;
        assert_eq!(common_data.degree_bits(), 12);

        // A standard recursive proof.
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &standard_config,
            None,
            false,
            false,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A high-rate recursive proof, designed to be verifiable with fewer routed wires.
        let high_rate_config = CircuitConfig {
            fri_config: FriConfig {
                rate_bits: 7,
                proof_of_work_bits: 16,
                num_query_rounds: 12,
                ..standard_config.fri_config.clone()
            },
            ..standard_config
        };
        let (proof, vd, common_data) = recursive_proof::<F, C, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &high_rate_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12);

        // A final proof, optimized for size.
        let final_config = CircuitConfig {
            num_routed_wires: 37,
            fri_config: FriConfig {
                rate_bits: 8,
                cap_height: 0,
                proof_of_work_bits: 20,
                reduction_strategy: FriReductionStrategy::MinSize(None),
                num_query_rounds: 10,
            },
            ..high_rate_config
        };
        let (proof, vd, common_data) = recursive_proof::<F, KC, C, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &final_config,
            None,
            true,
            true,
        )?;
        assert_eq!(common_data.degree_bits(), 12, "final proof too large");

        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    #[test]
    fn test_recursive_verifier_multi_hash() -> Result<()> {
        init_logger();
        const D: usize = GOLDILOCKS_EXTENSION_FIELD_DEGREE;
        const NUM_HASH_OUT_ELTS: usize = GOLDILOCKS_NUM_HASH_OUT_ELTS;
        type PC = PoseidonGoldilocksConfig;
        type KC = KeccakGoldilocksConfig;
        type F = <PC as GenericConfig<D, NUM_HASH_OUT_ELTS>>::F;

        let config = CircuitConfig::standard_recursion_config_gl();
        let (proof, vd, common_data) = dummy_proof::<F, PC, D, NUM_HASH_OUT_ELTS>(&config, 4_000)?;

        let (proof, vd, common_data) = recursive_proof::<F, PC, PC, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            false,
            false,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        let (proof, vd, common_data) = recursive_proof::<F, KC, PC, D, NUM_HASH_OUT_ELTS>(
            proof,
            vd,
            common_data,
            &config,
            None,
            false,
            false,
        )?;
        test_serialization(&proof, &vd, &common_data)?;

        Ok(())
    }

    type Proof<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize> = (
        ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
        VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
        CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    );

    /// Creates a dummy proof which should have roughly `num_dummy_gates` gates.
    fn dummy_proof<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<Proof<F, C, D, NUM_HASH_OUT_ELTS>>
where {
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
        for _ in 0..num_dummy_gates {
            builder.add_gate(NoopGate, vec![]);
        }
        match F::ORDER_U64 {
            Goldilocks::ORDER_U64 => {
                builder.add_gate_to_gate_set(GateRef::new(PoseidonGate::new()))
            }
            BabyBear::ORDER_U64 => {
                builder.add_gate_to_gate_set(GateRef::new(Poseidon2BabyBearGate::new(config)))
            }
            _ => panic!(),
        };
        let zeroes = [builder.zero(); NUM_HASH_OUT_ELTS];
        builder.register_public_inputs(&zeroes);
        let data = builder.build::<C>();
        let inputs = PartialWitness::new();
        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy lookup proof which does one lookup to one LUT.
    fn dummy_lookup_proof<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        config: &CircuitConfig,
        num_dummy_gates: u64,
    ) -> Result<Proof<F, C, D, NUM_HASH_OUT_ELTS>>
where {
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;

        let tip5_index = builder.add_lookup_table_from_pairs(table);

        let output_a = builder.add_lookup_from_index(initial_a, tip5_index);
        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);

        for _ in 0..num_dummy_gates + 1 {
            builder.add_gate(NoopGate, vec![]);
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);

        let data = builder.build::<C>();
        let mut inputs = PartialWitness::new();
        inputs.set_target(initial_a, F::from_canonical_usize(look_val_a));
        inputs.set_target(initial_b, F::from_canonical_usize(look_val_b));

        let proof = data.prove(inputs)?;
        data.verify(proof.clone())?;

        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(out_a),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_b),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy lookup proof which does one lookup to two different LUTs.
    fn dummy_two_luts_proof<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        config: &CircuitConfig,
    ) -> Result<Proof<F, C, D, NUM_HASH_OUT_ELTS>>
where {
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();

        let first_out = tip5_table[look_val_a];
        let second_out = tip5_table[look_val_b];

        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let other_table = OTHER_TABLE.to_vec();

        let tip5_index = builder.add_lookup_table_from_pairs(table);
        let output_a = builder.add_lookup_from_index(initial_a, tip5_index);

        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
        let sum = builder.add(output_a, output_b);

        let s = first_out + second_out;
        let final_out = other_table[s as usize];

        let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());

        let other_index = builder.add_lookup_table_from_pairs(table2);
        let output_final = builder.add_lookup_from_index(sum, other_index);

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);

        builder.register_public_input(sum);
        builder.register_public_input(output_a);
        builder.register_public_input(output_b);
        builder.register_public_input(output_final);

        let mut pw = PartialWitness::new();
        pw.set_target(initial_a, F::one());
        pw.set_target(initial_b, F::two());

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        data.verify(proof.clone())?;

        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(first_out),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        assert!(
            proof.public_inputs[4] == F::from_canonical_u16(second_out),
            "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(s),
            "Sum between the first two LUT outputs is incorrect."
        );
        assert!(
            proof.public_inputs[5] == F::from_canonical_u16(final_out),
            "Output of the second LUT at index {s} is incorrect."
        );

        Ok((proof, data.verifier_only, data.common))
    }

    /// Creates a dummy proof which has more than 256 lookups to one LUT.
    fn dummy_too_many_rows_proof<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        config: &CircuitConfig,
    ) -> Result<Proof<F, C, D, NUM_HASH_OUT_ELTS>>
where {
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());

        let initial_a = builder.add_virtual_target();
        let initial_b = builder.add_virtual_target();

        let look_val_a = 1;
        let look_val_b = 2;

        let tip5_table = TIP5_TABLE.to_vec();
        let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

        let out_a = table[look_val_a].1;
        let out_b = table[look_val_b].1;

        let tip5_index = builder.add_lookup_table_from_pairs(table);
        let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
        let mut output = builder.add_lookup_from_index(initial_a, tip5_index);
        for _ in 0..514 {
            output = builder.add_lookup_from_index(initial_a, tip5_index);
        }

        builder.register_public_input(initial_a);
        builder.register_public_input(initial_b);
        builder.register_public_input(output_b);
        builder.register_public_input(output);

        let mut pw = PartialWitness::new();

        pw.set_target(initial_a, F::from_canonical_usize(look_val_a));
        pw.set_target(initial_b, F::from_canonical_usize(look_val_b));

        let data = builder.build::<C>();
        let proof = data.prove(pw)?;
        assert!(
            proof.public_inputs[2] == F::from_canonical_u16(out_b),
            "First lookup, at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[1]
        );
        assert!(
            proof.public_inputs[3] == F::from_canonical_u16(out_a),
            "Lookups at index {} in the Tip5 table gives an incorrect output.",
            proof.public_inputs[0]
        );
        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    fn recursive_proof<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        InnerC: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        inner_proof: ProofWithPublicInputs<F, InnerC, D, NUM_HASH_OUT_ELTS>,
        inner_vd: VerifierOnlyCircuitData<InnerC, D, NUM_HASH_OUT_ELTS>,
        inner_cd: CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
        config: &CircuitConfig,
        min_degree_bits: Option<usize>,
        print_gate_counts: bool,
        print_timing: bool,
    ) -> Result<Proof<F, C, D, NUM_HASH_OUT_ELTS>>
    where
        InnerC::Hasher: AlgebraicHasher<F, NUM_HASH_OUT_ELTS>,
    {
        let mut builder = CircuitBuilder::<F, D, NUM_HASH_OUT_ELTS>::new(config.clone());
        let mut pw = PartialWitness::new();
        let pt = builder.add_virtual_proof_with_pis(&inner_cd);
        pw.set_proof_with_pis_target(&pt, &inner_proof);

        let inner_data = builder.add_virtual_verifier_data(inner_cd.config.fri_config.cap_height);
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        );
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest);

        builder.verify_proof::<InnerC>(&pt, &inner_data, &inner_cd);

        if print_gate_counts {
            builder.print_gate_counts(0);
        }

        if let Some(min_degree_bits) = min_degree_bits {
            // We don't want to pad all the way up to 2^min_degree_bits, as the builder will add a
            // few special gates afterward. So just pad to 2^(min_degree_bits - 1) + 1. Then the
            // builder will pad to the next power of two, 2^min_degree_bits.
            let min_gates = (1 << (min_degree_bits - 1)) + 1;
            for _ in builder.num_gates()..min_gates {
                builder.add_gate(NoopGate, vec![]);
            }
        }

        let data = builder.build::<C>();

        let mut timing = ProvingProcessInfo::new("prove", Level::Debug);
        let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
        if print_timing {
            timing.print();
        }

        data.verify(proof.clone())?;

        Ok((proof, data.verifier_only, data.common))
    }

    /// Test serialization and print some size info.
    fn test_serialization<
        F: RichField + HasExtension<D>,
        C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
        const D: usize,
        const NUM_HASH_OUT_ELTS: usize,
    >(
        proof: &ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
        vd: &VerifierOnlyCircuitData<C, D, NUM_HASH_OUT_ELTS>,
        common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
    ) -> Result<()>
where {
        let proof_bytes = proof.to_bytes();
        info!("Proof length: {} bytes", proof_bytes.len());
        let proof_from_bytes = ProofWithPublicInputs::from_bytes(proof_bytes, common_data)?;
        assert_eq!(proof, &proof_from_bytes);

        #[cfg(feature = "std")]
        let now = std::time::Instant::now();

        let compressed_proof = proof.clone().compress(&vd.circuit_digest, common_data)?;
        let decompressed_compressed_proof = compressed_proof
            .clone()
            .decompress(&vd.circuit_digest, common_data)?;

        #[cfg(feature = "std")]
        info!("{:.4}s to compress proof", now.elapsed().as_secs_f64());

        assert_eq!(proof, &decompressed_compressed_proof);

        let compressed_proof_bytes = compressed_proof.to_bytes();
        info!(
            "Compressed proof length: {} bytes",
            compressed_proof_bytes.len()
        );
        let compressed_proof_from_bytes =
            CompressedProofWithPublicInputs::from_bytes(compressed_proof_bytes, common_data)?;
        assert_eq!(compressed_proof, compressed_proof_from_bytes);

        Ok(())
    }

    fn init_logger() {
        let _ = env_logger::builder().format_timestamp(None).try_init();
    }
}
