use alloc::vec::Vec;

use p3_field::{batch_multiplicative_inverse, Field, TwoAdicField};
use plonky2_util::log2_ceil;

use crate::fft::ifft;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};

/// Computes the unique degree < n interpolant of an arbitrary list of n (point, value) pairs.
///
/// Note that the implementation assumes that `F` is two-adic, in particular that
/// `2^{F::TWO_ADICITY} >= points.len()`. This leads to a simple FFT-based implementation.
pub fn interpolant<F: TwoAdicField>(points: &[(F, F)]) -> PolynomialCoeffs<F> {
    let n = points.len();
    let n_log = log2_ceil(n);

    //WAS: let subgroup = two_adic_subgroup::<F>(n_log);
    let generator = F::two_adic_generator(n_log);
    let subgroup: Vec<F> = generator.powers().take(1 << n_log).collect();

    let barycentric_weights = barycentric_weights(points);
    let subgroup_evals = subgroup
        .into_iter()
        .map(|x| interpolate(points, x, &barycentric_weights))
        .collect();

    let mut coeffs = ifft(PolynomialValues::new(subgroup_evals));
    coeffs.trim();
    coeffs
}

/// Interpolate the polynomial defined by an arbitrary set of (point, value) pairs at the given
/// point `x`.
pub fn interpolate<F: Field>(points: &[(F, F)], x: F, barycentric_weights: &[F]) -> F {
    // If x is in the list of points, the Lagrange formula would divide by zero.
    for &(x_i, y_i) in points {
        if x_i == x {
            return y_i;
        }
    }

    let l_x: F = points.iter().map(|&(x_i, _y_i)| x - x_i).product();

    let sum = (0..points.len())
        .map(|i| {
            let x_i = points[i].0;
            let y_i = points[i].1;
            let w_i = barycentric_weights[i];
            w_i / (x - x_i) * y_i
        })
        .sum();

    l_x * sum
}

pub fn barycentric_weights<F: Field>(points: &[(F, F)]) -> Vec<F> {
    let n = points.len();

    batch_multiplicative_inverse::<F>(
        &(0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| points[i].0 - points[j].0)
                    .product::<F>()
            })
            .collect::<Vec<_>>(),
    )
}

/// Interpolate the linear polynomial passing through `points` on `x`.
pub fn interpolate2<F: Field>(points: [(F, F); 2], x: F) -> F {
    // a0 -> a1
    // b0 -> b1
    // x  -> a1 + (x-a0)*(b1-a1)/(b0-a0)
    let (a0, a1) = points[0];
    let (b0, b1) = points[1];
    assert_ne!(a0, b0);
    a1 + (x - a0) * (b1 - a1) / (b0 - a0)
}

#[cfg(test)]
mod tests {
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    use super::*;
    use crate::polynomial::PolynomialCoeffs;
    use crate::types::{two_adic_subgroup, Sample};
    use crate::GOLDILOCKS_EXTENSION_FIELD_DEGREE;

    #[test]
    fn interpolant_random() {
        type F = Goldilocks;

        for deg in 0..10 {
            let domain = F::rand_vec(deg);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    #[test]
    fn interpolant_random_roots_of_unity() {
        type F = Goldilocks;

        for deg_log in 0..4 {
            let deg = 1 << deg_log;
            let domain = two_adic_subgroup::<F>(deg_log);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    #[test]
    fn interpolant_random_overspecified() {
        type F = Goldilocks;

        for deg in 0..10 {
            let points = deg + 5;
            let domain = F::rand_vec(points);
            let coeffs = F::rand_vec(deg);
            let coeffs = PolynomialCoeffs { coeffs };

            let points = eval_naive(&coeffs, &domain);
            assert_eq!(interpolant(&points), coeffs);
        }
    }

    fn eval_naive<F: TwoAdicField>(coeffs: &PolynomialCoeffs<F>, domain: &[F]) -> Vec<(F, F)> {
        domain.iter().map(|&x| (x, coeffs.eval(x))).collect()
    }

    #[test]
    fn test_interpolate2() {
        type F = BinomialExtensionField<Goldilocks, GOLDILOCKS_EXTENSION_FIELD_DEGREE>;
        let points = [(F::rand(), F::rand()), (F::rand(), F::rand())];
        let x = F::rand();

        let ev0 = interpolant(&points).eval(x);
        let ev1 = interpolate(&points, x, &barycentric_weights(&points));
        let ev2 = interpolate2(points, x);

        assert_eq!(ev0, ev1);
        assert_eq!(ev0, ev2);
    }
}
