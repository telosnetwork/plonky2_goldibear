use anyhow::ensure;
use p3_field::TwoAdicField;

use plonky2_field::types::HasExtension;

use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::config::GenericConfig;
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};

pub(crate) fn validate_proof_with_pis_shape<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    proof_with_pis: &ProofWithPublicInputs<F, C, D, NUM_HASH_OUT_ELTS>,
    common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
) -> anyhow::Result<()>
where
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    F::Extension: TwoAdicField,
{
    let ProofWithPublicInputs {
        proof,
        public_inputs,
    } = proof_with_pis;
    validate_proof_shape(proof, common_data)?;
    ensure!(
        public_inputs.len() == common_data.num_public_inputs,
        "Number of public inputs doesn't match circuit data."
    );
    Ok(())
}

fn validate_proof_shape<F, C, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    proof: &Proof<F, C, D, NUM_HASH_OUT_ELTS>,
    common_data: &CommonCircuitData<F, D, NUM_HASH_OUT_ELTS>,
) -> anyhow::Result<()>
where
    F: RichField + HasExtension<D>,
    C: GenericConfig<D, NUM_HASH_OUT_ELTS, F = F, FE = F::Extension>,
    F::Extension: TwoAdicField,
{
    let config = &common_data.config;
    let Proof {
        wires_cap,
        plonk_zs_partial_products_cap,
        quotient_polys_cap,
        openings,
        // The shape of the opening proof will be checked in the FRI verifier (see
        // validate_fri_proof_shape), so we ignore it here.
        opening_proof: _,
    } = proof;
    let OpeningSet {
        constants,
        plonk_sigmas,
        wires,
        plonk_zs,
        plonk_zs_next,
        partial_products,
        quotient_polys,
        lookup_zs,
        lookup_zs_next,
    } = openings;
    let cap_height = common_data.fri_params.config.cap_height;
    ensure!(wires_cap.height() == cap_height);
    ensure!(plonk_zs_partial_products_cap.height() == cap_height);
    ensure!(quotient_polys_cap.height() == cap_height);
    ensure!(constants.len() == common_data.num_constants);
    ensure!(plonk_sigmas.len() == config.num_routed_wires);
    ensure!(wires.len() == config.num_wires);
    ensure!(plonk_zs.len() == config.num_challenges);
    ensure!(plonk_zs_next.len() == config.num_challenges);
    ensure!(partial_products.len() == config.num_challenges * common_data.num_partial_products);
    ensure!(quotient_polys.len() == common_data.num_quotient_polys());
    ensure!(lookup_zs.len() == common_data.num_all_lookup_polys());
    ensure!(lookup_zs_next.len() == common_data.num_all_lookup_polys());
    Ok(())
}
