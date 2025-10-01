//! Helpers for constructing signing schemes in tests.

use crate::{
    threshold_simplex::{
        signing::{self, BlsThresholdScheme, SigningScheme, VoteContext},
        types::Proposal,
    },
    types::Round,
};
use commonware_cryptography::{
    bls12381::{
        dkg::ops::evaluate_all,
        primitives::{group::Share, poly, variant::Variant},
    },
    sha256::Digest as Sha256Digest,
    Digest,
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

pub fn sign_notarize<V, D>(
    scheme: &BlsThresholdScheme<V>,
    namespace: &[u8],
    proposal: Proposal<D>,
) -> signing::Notarize<BlsThresholdScheme<V>, D>
where
    V: Variant,
    D: Digest,
{
    let signer = scheme.signer_id();
    let vote = scheme
        .sign_vote(
            VoteContext::Notarize {
                namespace,
                proposal: &proposal,
            },
            signer,
        )
        .expect("scheme should sign notarize vote");

    signing::Notarize { proposal, vote }
}

pub fn sign_nullify<V>(
    scheme: &BlsThresholdScheme<V>,
    namespace: &[u8],
    round: Round,
) -> signing::Nullify<BlsThresholdScheme<V>>
where
    V: Variant,
{
    let signer = scheme.signer_id();
    let vote = scheme
        .sign_vote::<Sha256Digest>(VoteContext::Nullify { namespace, round }, signer)
        .expect("scheme should sign nullify vote");

    signing::Nullify { round, vote }
}

pub fn sign_finalize<V, D>(
    scheme: &BlsThresholdScheme<V>,
    namespace: &[u8],
    proposal: Proposal<D>,
) -> signing::Finalize<BlsThresholdScheme<V>, D>
where
    V: Variant,
    D: Digest,
{
    let signer = scheme.signer_id();
    let vote = scheme
        .sign_vote(
            VoteContext::Finalize {
                namespace,
                proposal: &proposal,
            },
            signer,
        )
        .expect("scheme should sign finalize vote");

    signing::Finalize { proposal, vote }
}
