use crate::{
    threshold_simplex::{
        signing::{self, SigningScheme},
        types::{Notarization, Nullification},
    },
    types::View,
};
use commonware_cryptography::{bls12381::primitives::variant::Variant, Digest};
use futures::{channel::mpsc, SinkExt};

pub enum MessageLegacy<V: Variant, D: Digest> {
    Fetch {
        notarizations: Vec<View>,
        nullifications: Vec<View>,
    },
    Notarized {
        notarization: Notarization<V, D>,
    },
    Nullified {
        nullification: Nullification<V>,
    },
    Finalized {
        // Used to indicate when to prune old notarizations/nullifications.
        view: View,
    },
}

#[derive(Clone)]
pub struct MailboxLegacy<V: Variant, D: Digest> {
    sender: mpsc::Sender<MessageLegacy<V, D>>,
}

impl<V: Variant, D: Digest> MailboxLegacy<V, D> {
    pub fn new(sender: mpsc::Sender<MessageLegacy<V, D>>) -> Self {
        Self { sender }
    }

    pub async fn fetch(&mut self, notarizations: Vec<View>, nullifications: Vec<View>) {
        self.sender
            .send(MessageLegacy::Fetch {
                notarizations,
                nullifications,
            })
            .await
            .expect("Failed to send notarizations");
    }

    pub async fn notarized(&mut self, notarization: Notarization<V, D>) {
        self.sender
            .send(MessageLegacy::Notarized { notarization })
            .await
            .expect("Failed to send notarization");
    }

    pub async fn nullified(&mut self, nullification: Nullification<V>) {
        self.sender
            .send(MessageLegacy::Nullified { nullification })
            .await
            .expect("Failed to send nullification");
    }

    pub async fn finalized(&mut self, view: View) {
        self.sender
            .send(MessageLegacy::Finalized { view })
            .await
            .expect("Failed to send finalized view");
    }
}

#[allow(dead_code)]
pub enum MessageSigning<G: SigningScheme, D: Digest> {
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
        view: View,
    },
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

    pub async fn fetch(&mut self, notarizations: Vec<View>, nullifications: Vec<View>) {
        self.sender
            .send(MessageSigning::Fetch {
                notarizations,
                nullifications,
            })
            .await
            .expect("Failed to send notarizations");
    }

    pub async fn notarized(&mut self, notarization: signing::Notarization<G, D>) {
        self.sender
            .send(MessageSigning::Notarized { notarization })
            .await
            .expect("Failed to send notarization");
    }

    pub async fn nullified(&mut self, nullification: signing::Nullification<G>) {
        self.sender
            .send(MessageSigning::Nullified { nullification })
            .await
            .expect("Failed to send nullification");
    }

    pub async fn finalized(&mut self, view: View) {
        self.sender
            .send(MessageSigning::Finalized { view })
            .await
            .expect("Failed to send finalized view");
    }
}

pub type Message<V, D> = MessageLegacy<V, D>;
pub type Mailbox<V, D> = MailboxLegacy<V, D>;
