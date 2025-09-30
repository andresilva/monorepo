//! Helpers for constructing signing schemes in tests.

use crate::threshold_simplex::signing::BlsThresholdScheme;
use commonware_cryptography::bls12381::{
    dkg::ops::evaluate_all,
    primitives::{group::Share, poly, variant::Variant},
};

/// Build `BlsThresholdScheme` instances for a set of shares.
pub fn schemes_for_shares<V: Variant>(
    polynomial: &poly::Public<V>,
    shares: &[Share],
    threshold: usize,
    participant_count: usize,
) -> Vec<BlsThresholdScheme<V>> {
    let evaluations = evaluate_all::<V>(polynomial, participant_count as u32);
    let identity = polynomial.constant().clone();
    shares
        .iter()
        .map(|share| {
            BlsThresholdScheme::new(
                evaluations.clone(),
                identity.clone(),
                share.clone(),
                threshold,
            )
        })
        .collect()
}

/// Build a single `BlsThresholdScheme` for the provided share.
pub fn scheme_for_share<V: Variant>(
    polynomial: &poly::Public<V>,
    share: &Share,
    threshold: usize,
    participant_count: usize,
) -> BlsThresholdScheme<V> {
    schemes_for_shares(
        polynomial,
        std::slice::from_ref(share),
        threshold,
        participant_count,
    )
    .into_iter()
    .next()
    .expect("scheme_for_share should yield one scheme")
}
