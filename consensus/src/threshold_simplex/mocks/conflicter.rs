//! Byzantine participant that sends conflicting notarize/finalize messages.

use super::signing::{sign_finalize, sign_notarize};
use crate::{
    threshold_simplex::{
        signing::BlsThresholdScheme,
        types::{Proposal, Voter},
    },
    types::View,
    ThresholdSupervisor,
};
use commonware_codec::{DecodeExt, Encode};
use commonware_cryptography::{
    bls12381::primitives::{group, variant::Variant},
    Digest, Hasher,
};
use commonware_p2p::{Receiver, Recipients, Sender};
use commonware_runtime::{Clock, Handle, Spawner};
use commonware_utils::quorum;
use rand::{CryptoRng, Rng};
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

pub struct Conflicter<
    E: Clock + Rng + CryptoRng + Spawner,
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
        E: Clock + Rng + CryptoRng + Spawner,
        V: Variant,
        H: Hasher,
        S: ThresholdSupervisor<
            Seed = V::Signature,
            Index = View,
            Share = group::Share,
            Identity = V::Public,
            Polynomial = Vec<V::Public>,
        >,
    > Conflicter<E, V, H, S>
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

    async fn run(mut self, pending_network: (impl Sender, impl Receiver)) {
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
                    // Notarize random digest
                    let view = notarize.view();
                    let scheme = match self.build_scheme(view) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let payload = H::Digest::random(&mut self.context);
                    let proposal =
                        Proposal::new(notarize.proposal.round, notarize.proposal.parent, payload);
                    let n = sign_notarize(&scheme, &self.namespace, proposal);
                    let msg = Voter::Notarize(n).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();

                    // Notarize received digest
                    let n = sign_notarize(&scheme, &self.namespace, notarize.proposal);
                    let msg = Voter::Notarize(n).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();
                }
                Voter::Finalize(finalize) => {
                    // Finalize random digest
                    let view = finalize.view();
                    let scheme = match self.build_scheme(view) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let payload = H::Digest::random(&mut self.context);
                    let proposal =
                        Proposal::new(finalize.proposal.round, finalize.proposal.parent, payload);
                    let f = sign_finalize(&scheme, &self.namespace, proposal);
                    let msg = Voter::Finalize(f).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();

                    // Finalize provided digest
                    let f = sign_finalize(&scheme, &self.namespace, finalize.proposal);
                    let msg = Voter::Finalize(f).encode().into();
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
