use crate::{
    threshold_simplex::signing::SigningScheme,
    threshold_simplex::types::{LegacyVoter, Voter},
    types::View,
};
use commonware_cryptography::{bls12381::primitives::variant::Variant, Digest, PublicKey};
use futures::{
    channel::{mpsc, oneshot},
    SinkExt,
};

pub enum MessageLegacy<P: PublicKey, V: Variant, D: Digest> {
    Update {
        current: View,
        leader: P,
        finalized: View,

        active: oneshot::Sender<bool>,
    },
    Constructed(LegacyVoter<V, D>),
}

#[derive(Clone)]
pub struct MailboxLegacy<P: PublicKey, V: Variant, D: Digest> {
    sender: mpsc::Sender<MessageLegacy<P, V, D>>,
}

impl<P: PublicKey, V: Variant, D: Digest> MailboxLegacy<P, V, D> {
    pub fn new(sender: mpsc::Sender<MessageLegacy<P, V, D>>) -> Self {
        Self { sender }
    }

    pub async fn update(&mut self, current: View, leader: P, finalized: View) -> bool {
        let (active, active_receiver) = oneshot::channel();
        self.sender
            .send(MessageLegacy::Update {
                current,
                leader,
                finalized,
                active,
            })
            .await
            .expect("Failed to send update");
        active_receiver.await.unwrap()
    }

    pub async fn constructed(&mut self, message: LegacyVoter<V, D>) {
        self.sender
            .send(MessageLegacy::Constructed(message))
            .await
            .expect("Failed to send message");
    }
}

#[allow(dead_code)]
pub enum MessageSigning<P: PublicKey, G: SigningScheme, D: Digest> {
    Update {
        current: View,
        leader: P,
        finalized: View,

        active: oneshot::Sender<bool>,
    },
    Constructed(Voter<G, D>),
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct MailboxSigning<P: PublicKey, G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<MessageSigning<P, G, D>>,
}

#[allow(dead_code)]
impl<P: PublicKey, G: SigningScheme, D: Digest> MailboxSigning<P, G, D> {
    pub fn new(sender: mpsc::Sender<MessageSigning<P, G, D>>) -> Self {
        Self { sender }
    }

    pub async fn update(&mut self, current: View, leader: P, finalized: View) -> bool {
        let (active, active_receiver) = oneshot::channel();
        self.sender
            .send(MessageSigning::Update {
                current,
                leader,
                finalized,
                active,
            })
            .await
            .expect("Failed to send update");
        active_receiver.await.unwrap()
    }

    pub async fn constructed(&mut self, message: Voter<G, D>) {
        self.sender
            .send(MessageSigning::Constructed(message))
            .await
            .expect("Failed to send message");
    }
}
pub type Message<P, V, D> = MessageLegacy<P, V, D>;
pub type Mailbox<P, V, D> = MailboxLegacy<P, V, D>;
