use crate::threshold_simplex::types::{LegacyVoter, Voter};
use crate::threshold_simplex::signing::SigningScheme;
use commonware_cryptography::{bls12381::primitives::variant::Variant, Digest};
use futures::{channel::mpsc, stream, SinkExt};

// If either of these requests fails, it will not send a reply.
pub enum MessageLegacy<V: Variant, D: Digest> {
    Verified(LegacyVoter<V, D>),
}

#[derive(Clone)]
pub struct MailboxLegacy<V: Variant, D: Digest> {
    sender: mpsc::Sender<MessageLegacy<V, D>>,
}

impl<V: Variant, D: Digest> MailboxLegacy<V, D> {
    pub fn new(sender: mpsc::Sender<MessageLegacy<V, D>>) -> Self {
        Self { sender }
    }

    pub async fn verified(&mut self, voters: Vec<LegacyVoter<V, D>>) {
        self.sender
            .send_all(&mut stream::iter(
                voters
                    .into_iter()
                    .map(|voter| Ok(MessageLegacy::Verified(voter))),
            ))
            .await
            .expect("Failed to send batch of voters");
    }
}

#[allow(dead_code)]
pub enum MessageSigning<G: SigningScheme, D: Digest> {
    Verified(Voter<G, D>),
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct MailboxSigning<G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<MessageSigning<G, D>>,
}

#[allow(dead_code)]
impl<G: SigningScheme, D: Digest> MailboxSigning<G, D> {
    pub fn new(sender: mpsc::Sender<MessageSigning<G, D>>) -> Self {
        Self { sender }
    }

    pub async fn verified(&mut self, voters: Vec<Voter<G, D>>) {
        self.sender
            .send_all(&mut stream::iter(
                voters
                    .into_iter()
                    .map(|voter| Ok(MessageSigning::Verified(voter))),
            ))
            .await
            .expect("Failed to send batch of voters");
    }
}
