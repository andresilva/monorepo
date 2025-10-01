//! Byzantine participant that behaves correctly except increments the epoch by 1
//! in all outgoing votes (notarize/finalize/nullify). This helps ensure peers
//! reject messages from an unexpected epoch.

use super::signing::{sign_finalize, sign_notarize, sign_nullify};
use crate::{
    threshold_simplex::{signing::BlsThresholdScheme, types::Voter},
    types::{Epoch, View},
    ThresholdSupervisor,
};
use commonware_codec::{DecodeExt, Encode};
use commonware_cryptography::{
    bls12381::primitives::{group, variant::Variant},
    Hasher,
};
use commonware_p2p::{Receiver, Recipients, Sender};
use commonware_runtime::{Handle, Spawner};
use commonware_utils::quorum;
use std::marker::PhantomData;
use tracing::debug;

type Scheme<V> = BlsThresholdScheme<V>;

pub struct Config<V, S>
where
    V: Variant,
    S: ThresholdSupervisor<
        Seed = V::Signature,
        Index = View,
        Share = group::Share,
        Identity = V::Public,
        Polynomial = Vec<V::Public>,
    >,
{
    pub supervisor: S,
    pub namespace: Vec<u8>,
    pub(crate) marker: PhantomData<V>,
}

impl<V, S> Config<V, S>
where
    V: Variant,
    S: ThresholdSupervisor<
        Seed = V::Signature,
        Index = View,
        Share = group::Share,
        Identity = V::Public,
        Polynomial = Vec<V::Public>,
    >,
{
    pub fn new(supervisor: S, namespace: Vec<u8>) -> Self {
        Self {
            supervisor,
            namespace,
            marker: PhantomData,
        }
    }
}

pub struct Reconfigurer<
    E: Spawner,
    V: Variant,
    H: Hasher,
    S: ThresholdSupervisor<
        Seed = V::Signature,
        Index = View,
        Share = group::Share,
        Identity = V::Public,
        Polynomial = Vec<V::Public>,
    >,
> {
    context: E,
    supervisor: S,
    namespace: Vec<u8>,
    _hasher: PhantomData<H>,
    _variant: PhantomData<V>,
}

impl<
        E: Spawner,
        V: Variant,
        H: Hasher,
        S: ThresholdSupervisor<
            Seed = V::Signature,
            Index = View,
            Share = group::Share,
            Identity = V::Public,
            Polynomial = Vec<V::Public>,
        >,
    > Reconfigurer<E, V, H, S>
{
    pub fn new(context: E, cfg: Config<V, S>) -> Self {
        Self {
            context,
            supervisor: cfg.supervisor,
            namespace: cfg.namespace,
            _hasher: PhantomData,
            _variant: PhantomData,
        }
    }

    pub fn start(mut self, pending_network: (impl Sender, impl Receiver)) -> Handle<()> {
        self.context.spawn_ref()(self.run(pending_network))
    }

    async fn run(self, pending_network: (impl Sender, impl Receiver)) {
        let (mut sender, mut receiver) = pending_network;
        while let Ok((s, msg)) = receiver.recv().await {
            // Parse message
            let msg = match Voter::<Scheme<V>, H::Digest>::decode(msg) {
                Ok(msg) => msg,
                Err(err) => {
                    debug!(?err, sender = ?s, "failed to decode message");
                    continue;
                }
            };

            // Process message
            match msg {
                Voter::Notarize(notarize) => {
                    // Build identical proposal but with epoch incremented by 1
                    let scheme = match self.build_scheme(notarize.view()) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let mut proposal = notarize.proposal.clone();
                    let old_round = proposal.round;
                    let new_epoch: Epoch = old_round.epoch().saturating_add(1);
                    proposal.round = (new_epoch, old_round.view()).into();

                    // Sign and broadcast
                    let n = sign_notarize(&scheme, &self.namespace, proposal);
                    let msg = Voter::Notarize(n).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();
                }
                Voter::Finalize(finalize) => {
                    // Build identical proposal but with epoch incremented by 1
                    let scheme = match self.build_scheme(finalize.view()) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let mut proposal = finalize.proposal.clone();
                    let old_round = proposal.round;
                    let new_epoch: Epoch = old_round.epoch().saturating_add(1);
                    proposal.round = (new_epoch, old_round.view()).into();

                    // Sign and broadcast
                    let f = sign_finalize(&scheme, &self.namespace, proposal);
                    let msg = Voter::Finalize(f).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();
                }
                Voter::Nullify(nullify) => {
                    // Re-sign nullify for the next epoch
                    let scheme = match self.build_scheme(nullify.view()) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let old_round = nullify.round;
                    let new_epoch: Epoch = old_round.epoch().saturating_add(1);
                    let new_round = (new_epoch, old_round.view()).into();

                    let n = sign_nullify(&scheme, &self.namespace, new_round);
                    let msg = Voter::<Scheme<V>, H::Digest>::Nullify(n).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();
                }
                _ => continue,
            }
        }
    }

    fn build_scheme(&self, view: View) -> Option<Scheme<V>> {
        let share = self.supervisor.share(view)?.clone();
        let polynomial = self.supervisor.polynomial(view)?.clone();
        let participants = self.supervisor.participants(view)?;
        let threshold = quorum(participants.len() as u32) as usize;
        let identity = self.supervisor.identity().clone();

        Some(BlsThresholdScheme::new(
            polynomial, identity, share, threshold,
        ))
    }
}
