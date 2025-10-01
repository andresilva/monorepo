use crate::{
    threshold_simplex::{
        signing::SigningScheme,
        types::Voter,
    },
    types::View,
};
use commonware_cryptography::{Digest, PublicKey};
use futures::{
    channel::{mpsc, oneshot},
    SinkExt,
};

pub enum Message<P: PublicKey, G: SigningScheme, D: Digest> {
    Update {
        current: View,
        leader: P,
        finalized: View,

        active: oneshot::Sender<bool>,
    },
    Constructed(Voter<G, D>),
}

#[derive(Clone)]
pub struct Mailbox<P: PublicKey, G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<Message<P, G, D>>,
}

impl<P: PublicKey, G: SigningScheme, D: Digest> Mailbox<P, G, D> {
    pub fn new(sender: mpsc::Sender<Message<P, G, D>>) -> Self {
        Self { sender }
    }

    pub async fn update(&mut self, current: View, leader: P, finalized: View) -> bool {
        let (active, active_receiver) = oneshot::channel();
        self.sender
            .send(Message::Update {
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
            .send(Message::Constructed(message))
            .await
            .expect("Failed to send message");
    }
}
