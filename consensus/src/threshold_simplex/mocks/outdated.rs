//! Byzantine participant that sends outdated notarize and finalize messages.

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
    Hasher,
};
use commonware_p2p::{Receiver, Recipients, Sender};
use commonware_runtime::{Clock, Handle, Spawner};
use commonware_utils::quorum;
use rand::{CryptoRng, Rng};
use std::{collections::HashMap, marker::PhantomData};
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
    pub view_delta: u64,
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
    pub fn new(supervisor: S, namespace: Vec<u8>, view_delta: u64) -> Self {
        Self {
            supervisor,
            namespace,
            view_delta,
            marker: PhantomData,
        }
    }
}

pub struct Outdated<
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

    history: HashMap<u64, Proposal<H::Digest>>,
    view_delta: u64,

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
    > Outdated<E, V, H, S>
{
    pub fn new(context: E, cfg: Config<V, S>) -> Self {
        Self {
            context,
            supervisor: cfg.supervisor,

            namespace: cfg.namespace,

            history: HashMap::new(),
            view_delta: cfg.view_delta,

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
            let view = msg.view();

            // Process message
            match msg {
                Voter::Notarize(notarize) => {
                    // Store proposal
                    self.history.insert(view, notarize.proposal.clone());

                    // Notarize old digest
                    let view = view.saturating_sub(self.view_delta);
                    let scheme = match self.build_scheme(view) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let Some(proposal) = self.history.get(&view) else {
                        continue;
                    };
                    debug!(?view, "notarizing old proposal");
                    let n = sign_notarize(&scheme, &self.namespace, proposal.clone());
                    let msg = Voter::Notarize(n).encode().into();
                    sender.send(Recipients::All, msg, true).await.unwrap();
                }
                Voter::Finalize(finalize) => {
                    // Store proposal
                    self.history.insert(view, finalize.proposal.clone());

                    // Finalize old digest
                    let view = view.saturating_sub(self.view_delta);
                    let scheme = match self.build_scheme(view) {
                        Some(scheme) => scheme,
                        None => continue,
                    };
                    let Some(proposal) = self.history.get(&view) else {
                        continue;
                    };
                    debug!(?view, "finalizing old proposal");
                    let f = sign_finalize(&scheme, &self.namespace, proposal.clone());
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
