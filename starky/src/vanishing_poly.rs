use p3_field::{ExtensionField, PackedField};

use plonky2::field::types::HasExtension;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::cross_table_lookup::{
    CtlCheckVars, CtlCheckVarsTarget, eval_cross_table_lookup_checks,
    eval_cross_table_lookup_checks_circuit,
};
use crate::lookup::{
    eval_ext_lookups_circuit, eval_packed_lookups_generic, Lookup, LookupCheckVars,
    LookupCheckVarsTarget,
};
use crate::stark::Stark;

/// Evaluates all constraint, permutation and cross-table lookup polynomials
/// of the current STARK at the local and next values.
pub(crate) fn eval_vanishing_poly<F, FE, P, S, const D: usize, const NUM_HASH_OUT_ELTS: usize, const D2: usize>(
    stark: &S,
    vars: &S::EvaluationFrame<FE, P, D2>,
    lookups: &[Lookup<F>],
    lookup_vars: Option<LookupCheckVars<F, FE, P, D2>>,
    ctl_vars: Option<&[CtlCheckVars<F, FE, P, D2, NUM_HASH_OUT_ELTS>]>,
    consumer: &mut ConstraintConsumer<P>,
) where
    F: RichField + HasExtension<D> + HasExtension<D2, Extension = FE>,
    FE: ExtensionField<F>,

    P: PackedField<Scalar = FE>,
    S: Stark<F, D, NUM_HASH_OUT_ELTS>,
{
    // Evaluate all of the STARK's table constraints.
    stark.eval_packed_generic(vars, consumer);
    if let Some(lookup_vars) = lookup_vars {
        // Evaluate the STARK constraints related to the permutation arguments.
        eval_packed_lookups_generic::<F, FE, P, S, D, NUM_HASH_OUT_ELTS, D2>(
            stark,
            lookups,
            vars,
            lookup_vars,
            consumer,
        );
    }
    if let Some(ctl_vars) = ctl_vars {
        // Evaluate the STARK constraints related to the CTLs.
        eval_cross_table_lookup_checks::<F, FE, P, S, D, NUM_HASH_OUT_ELTS, D2>(
            vars,
            ctl_vars,
            consumer,
            stark.constraint_degree(),
        );
    }
}

/// Circuit version of `eval_vanishing_poly`.
/// Evaluates all constraint, permutation and cross-table lookup polynomials
/// of the current STARK at the local and next values.
pub(crate) fn eval_vanishing_poly_circuit<F, S, const D: usize, const NUM_HASH_OUT_ELTS: usize>(
    builder: &mut CircuitBuilder<F, D, NUM_HASH_OUT_ELTS>,
    stark: &S,
    vars: &S::EvaluationFrameTarget,
    lookup_vars: Option<LookupCheckVarsTarget<D>>,
    ctl_vars: Option<&[CtlCheckVarsTarget<F, D, NUM_HASH_OUT_ELTS>]>,
    consumer: &mut RecursiveConstraintConsumer<F, D, NUM_HASH_OUT_ELTS>,
) where
    F: RichField + HasExtension<D>,
    S: Stark<F, D, NUM_HASH_OUT_ELTS>,
{
    // Evaluate all of the STARK's table constraints.
    stark.eval_ext_circuit(builder, vars, consumer);
    if let Some(lookup_vars) = lookup_vars {
        // Evaluate all of the STARK's constraints related to the permutation argument.
        eval_ext_lookups_circuit::<F, S, D, NUM_HASH_OUT_ELTS>(builder, stark, vars, lookup_vars, consumer);
    }
    if let Some(ctl_vars) = ctl_vars {
        // Evaluate all of the STARK's constraints related to the CTLs.
        eval_cross_table_lookup_checks_circuit::<S, F, D, NUM_HASH_OUT_ELTS>(
            builder,
            vars,
            ctl_vars,
            consumer,
            stark.constraint_degree(),
        );
    }
}
