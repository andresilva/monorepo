use crate::{
    threshold_simplex::signing::{self, SigningScheme},
    types::View,
};
use commonware_cryptography::Digest;
use futures::{channel::mpsc, SinkExt};

pub enum Message<G: SigningScheme, D: Digest> {
    Fetch {
        notarizations: Vec<View>,
        nullifications: Vec<View>,
    },
    Notarized {
        notarization: signing::Notarization<G, D>,
    },
    Nullified {
        nullification: signing::Nullification<G>,
    },
    Finalized {
        // Used to indicate when to prune old notarizations/nullifications.
        view: View,
    },
}

#[derive(Clone)]
pub struct Mailbox<G: SigningScheme, D: Digest> {
    sender: mpsc::Sender<Message<G, D>>,
}

impl<G: SigningScheme, D: Digest> Mailbox<G, D> {
    pub fn new(sender: mpsc::Sender<Message<G, D>>) -> Self {
        Self { sender }
    }

    pub async fn fetch(&mut self, notarizations: Vec<View>, nullifications: Vec<View>) {
        self.sender
            .send(Message::Fetch {
                notarizations,
                nullifications,
            })
            .await
            .expect("Failed to send notarizations");
    }

    pub async fn notarized_signing(&mut self, notarization: signing::Notarization<G, D>) {
        self.sender
            .send(Message::Notarized { notarization })
            .await
            .expect("Failed to send notarization");
    }

    pub async fn nullified_signing(&mut self, nullification: signing::Nullification<G>) {
        self.sender
            .send(Message::Nullified { nullification })
            .await
            .expect("Failed to send nullification");
    }

    pub async fn finalized(&mut self, view: View) {
        self.sender
            .send(Message::Finalized { view })
            .await
            .expect("Failed to send finalized view");
    }
}
