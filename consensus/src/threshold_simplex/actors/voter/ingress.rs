use crate::threshold_simplex::{
    signing::SigningScheme,
    types::Voter,
};
use commonware_cryptography::Digest;
use futures::{channel::mpsc, stream, SinkExt};

pub enum Message<G: SigningScheme, D: Digest> {
    Verified(Voter<G, D>),
}

#[derive(Clone)]
pub struct Mailbox<G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<Message<G, D>>,
}

impl<G: SigningScheme, D: Digest> Mailbox<G, D> {
    pub fn new(sender: mpsc::Sender<Message<G, D>>) -> Self {
        Self { sender }
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
