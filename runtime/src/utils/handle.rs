use crate::{utils::extract_panic_message, Error};
use futures::{
    channel::oneshot,
    stream::{AbortHandle, Abortable},
    FutureExt as _,
};
use prometheus_client::metrics::gauge::Gauge;
use std::{
    future::Future,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    pin::Pin,
    sync::{Arc, Mutex, Once},
    task::{Context, Poll},
};
use tracing::error;

/// Handle to a spawned task.
pub struct Handle<T>
where
    T: Send + 'static,
{
    aborter: Option<AbortHandle>,
    receiver: oneshot::Receiver<Result<T, Error>>,

    guard: Arc<HandleGuard>,
}

impl<T> Handle<T>
where
    T: Send + 'static,
{
    pub(crate) fn init_future<F>(
        f: F,
        running: Gauge,
        catch_panic: bool,
        children: Arc<Mutex<Vec<AbortHandleGuard>>>,
    ) -> (impl Future<Output = ()>, Self)
    where
        F: Future<Output = T> + Send + 'static,
    {
        // Initialize channels to handle result/abort
        let (sender, receiver) = oneshot::channel();
        let (aborter, abort_registration) = AbortHandle::new_pair();
        let guard = HandleGuard::new(running);

        // Wrap the future to handle panics
        let wrapped = {
            let guard = guard.clone();
            async move {
                // Run future
                let result = AssertUnwindSafe(f).catch_unwind().await;

                // Mark the task as finished
                guard.finish();

                // Handle result
                let result = match result {
                    Ok(result) => Ok(result),
                    Err(err) => {
                        if !catch_panic {
                            resume_unwind(err);
                        }
                        let err = extract_panic_message(&*err);
                        error!(?err, "task panicked");
                        Err(Error::Exited)
                    }
                };
                let _ = sender.send(result);
            }
        };

        // Make the future abortable
        let abortable = Abortable::new(wrapped, abort_registration);
        (
            abortable.map(move |_| {
                // Abort all children
                for handle in children.lock().unwrap().drain(..) {
                    handle.abort();
                }
            }),
            Self {
                aborter: Some(aborter),
                receiver,
                guard,
            },
        )
    }

    pub(crate) fn init_blocking<F>(f: F, running: Gauge, catch_panic: bool) -> (impl FnOnce(), Self)
    where
        F: FnOnce() -> T + Send + 'static,
    {
        // Initialize channel to handle result
        let (sender, receiver) = oneshot::channel();
        let guard = HandleGuard::new(running);

        // Wrap the closure with panic handling
        let f = {
            let guard = guard.clone();
            move || {
                // Run blocking task
                let result = catch_unwind(AssertUnwindSafe(f));

                // Mark the task as finished
                guard.finish();

                // Handle result
                let result = match result {
                    Ok(value) => Ok(value),
                    Err(err) => {
                        if !catch_panic {
                            resume_unwind(err);
                        }
                        let err = extract_panic_message(&*err);
                        error!(?err, "task panicked");
                        Err(Error::Exited)
                    }
                };
                let _ = sender.send(result);
            }
        };

        // Return the task and handle
        (
            f,
            Self {
                aborter: None,
                receiver,
                guard,
            },
        )
    }

    /// Abort the task (if not blocking).
    pub fn abort(&self) {
        // Get aborter and abort
        let Some(aborter) = &self.aborter else {
            return;
        };
        aborter.abort();

        // Mark the task as finished
        self.guard.finish();
    }

    /// Returns an [AbortHandleGuard] that can be used to abort the task.
    pub(crate) fn abort_handle(&self) -> Option<AbortHandleGuard> {
        self.aborter
            .clone()
            .map(|inner| AbortHandleGuard::new(inner, self.guard.clone()))
    }
}

impl<T> Future for Handle<T>
where
    T: Send + 'static,
{
    type Output = Result<T, Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(cx) {
            Poll::Ready(Ok(Ok(value))) => {
                self.guard.finish();
                Poll::Ready(Ok(value))
            }
            Poll::Ready(Ok(Err(err))) => {
                self.guard.finish();
                Poll::Ready(Err(err))
            }
            Poll::Ready(Err(_)) => {
                self.guard.finish();
                Poll::Ready(Err(Error::Closed))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Tracks the lifecycle of a spawned task so completion logic runs exactly once.
struct HandleGuard {
    running: Gauge,
    once: Once,
}

impl HandleGuard {
    fn new(running: Gauge) -> Arc<Self> {
        let guard = Arc::new(Self {
            running,
            once: Once::new(),
        });

        guard.running.inc();
        guard
    }

    fn finish(&self) {
        self.once.call_once(|| {
            self.running.dec();
        });
    }
}

impl Drop for HandleGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Couples an `AbortHandle` with the task guard so both stay in sync.
#[derive(Clone)]
pub(crate) struct AbortHandleGuard {
    inner: AbortHandle,
    guard: Arc<HandleGuard>,
}

impl AbortHandleGuard {
    fn new(inner: AbortHandle, guard: Arc<HandleGuard>) -> Self {
        Self { inner, guard }
    }

    /// Abort the associated task and finalize its guard immediately.
    fn abort(&self) {
        self.inner.abort();
        self.guard.finish();
    }
}

#[cfg(test)]
mod tests {
    use crate::{deterministic, Metrics, Runner, Spawner};
    use futures::future;

    #[test]
    fn test_abort_guard_finishes_immediately() {
        // Constants for the test
        const LABEL: &str = "abort_guard_test";
        const RUNTIME_LABEL: &str = "runtime_tasks_running{";
        const LABEL_FORMAT: &str = "name=\"abort_guard_test\"";
        const TASK_FORMAT: &str = "task=\"Future\"";

        // Run the test
        let runner = deterministic::Runner::default();
        runner.start(|context| async move {
            // Use a label so we can target the metric line precisely
            let context = context.with_label(LABEL);

            // Spawn a task that never completes
            let handle = context.clone().spawn(|_| async move {
                future::pending::<()>().await;
            });

            // Running gauge should be 1 for this label
            let before = context.encode();
            let before_ok = before.lines().any(|line| {
                line.starts_with(RUNTIME_LABEL)
                    && line.contains(LABEL_FORMAT)
                    && line.contains(TASK_FORMAT)
                    && line.trim_end().ends_with(" 1")
            });
            assert!(before_ok, "metrics before abort: {}", before);

            // Abort via AbortHandleGuard, which should finish the task guard
            // immediately, thus decrementing the gauge
            let guard = handle.abort_handle().expect("abort handle not present");
            guard.abort();

            // Gauge should now be 0 without requiring an additional poll
            let after = context.encode();
            let after_ok = after.lines().any(|line| {
                line.starts_with(RUNTIME_LABEL)
                    && line.contains(LABEL_FORMAT)
                    && line.contains(TASK_FORMAT)
                    && line.trim_end().ends_with(" 0")
            });
            assert!(after_ok, "metrics after abort: {}", after);
        });
    }
}
