use crate::threshold_simplex::{
    signing::SigningScheme,
    types::{LegacyVoter, Voter},
};
use commonware_cryptography::{bls12381::primitives::variant::Variant, Digest};
use futures::{channel::mpsc, stream, SinkExt};
use std::marker::PhantomData;

pub enum Message<G: SigningScheme, D: Digest> {
    Verified(Voter<G, D>),
}

#[derive(Clone)]
pub struct Mailbox<V: Variant, G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<Message<G, D>>,
    _marker: PhantomData<V>,
}

impl<V, G, D> Mailbox<V, G, D>
where
    V: Variant,
    G: SigningScheme<
        SignerId = u32,
        Signature = (V::Signature, V::Signature),
        Certificate = (V::Signature, V::Signature),
    >,
    D: Digest,
{
    pub fn new(sender: mpsc::Sender<Message<G, D>>) -> Self {
        Self {
            sender,
            _marker: PhantomData,
        }
    }

    pub async fn verified(&mut self, voters: Vec<LegacyVoter<V, D>>) {
        let voters = voters
            .into_iter()
            .map(|legacy| Voter::from(legacy))
            .collect::<Vec<Voter<G, D>>>();
        self.verified_signing(voters).await;
    }

    pub async fn verified_signing(&mut self, voters: Vec<Voter<G, D>>) {
        self.sender
            .send_all(&mut stream::iter(
                voters
                    .into_iter()
                    .map(|voter| Ok(Message::Verified(voter))),
            ))
            .await
            .expect("Failed to send batch of voters");
    }
}
