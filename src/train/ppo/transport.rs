//! Networked actor-learner transport (Phase 4 of the distributed-training
//! epic #265 — see `docs/DISTRIBUTED_TRAINING_DESIGN.md`).
//!
//! [`actor_learner`](super::actor_learner) is generic over the actor→learner
//! experience sender
//! ([`ActorChannels::experience_tx`](super::actor_learner::ActorChannels))
//! and the learner→actor broadcast/control senders
//! ([`ActorHandle`]'s `BR`/`CTRL` type
//! parameters). Both default to the in-process `crossbeam_channel` types
//! every pre-#281 caller already used — **that path is unchanged, still the
//! default, and every Phase 2/3 test keeps exercising it verbatim.**
//!
//! This module supplies the second implementation of that seam: a TCP
//! transport that lets a remote actor process stream `Experience` batches to
//! a learner over the network, and receive `PolicyBroadcast`/`ControlMessage`
//! frames back, using [`actor_thread`](super::actor_learner::actor_thread)
//! and [`learner_loop`](super::actor_learner::learner_loop) completely
//! unmodified — only the channel endpoints they are handed differ.
//!
//! # Wire format
//!
//! Every frame is one [`WireMessage`] value, `bincode`-encoded
//! ([`bincode::config::standard`] via `bincode`'s `serde` integration — see
//! `write_message` / `read_message`) and length-prefixed with a 4-byte
//! big-endian `u32`. `Experience`/`PolicyBroadcast`/`ControlMessage` derive
//! `Serialize`/`Deserialize` directly (`PolicyBroadcast`'s `Arc<Vec<u8>>`
//! payload via serde's `rc` feature — see `src/multi_agent/messages.rs`), so
//! no separate wire DTO is needed.
//!
//! # Topology
//!
//! One TCP connection per remote actor, carrying **both** directions:
//! - actor → learner: [`WireMessage::Experience`] frames only.
//! - learner → actor: [`WireMessage::Broadcast`] / [`WireMessage::Control`]
//!   frames only.
//!
//! Each side runs one background thread that only ever reads its incoming
//! direction and forwards decoded values onto an ordinary local
//! `crossbeam_channel`:
//! - **Actor side** ([`connect_actor_transport`]): a "demux" thread reads
//!   `Broadcast`/`Control` frames off the socket and forwards them onto local
//!   `broadcast_tx`/`control_tx` senders — so
//!   [`actor_thread`](super::actor_learner::actor_thread)'s `broadcast_rx` /
//!   `control_rx` stay the exact concrete crossbeam receivers it always used.
//!   Only `experience_tx` is TCP-backed ([`TcpExperienceSender`]).
//! - **Learner side** ([`accept_actors_over_tcp`]): one reader thread per
//!   accepted connection decodes `Experience` frames and forwards them onto a
//!   shared crossbeam sender feeding the single `experience_rx`
//!   [`learner_loop`](super::actor_learner::learner_loop) already consumes.
//!   That reader thread *is* the resulting
//!   [`ActorHandle`]'s join handle: it
//!   returns best-effort [`ActorStats`]
//!   (steps/episodes forwarded) once the remote actor disconnects, so an actor
//!   dying mid-training surfaces the same way a local actor panic would — the
//!   learner's fill loop simply stops receiving from that column and the others
//!   continue.
//!
//! # Loopback verification
//!
//! `tests/test_actor_learner_tcp.rs` runs a learner (accepting on
//! `127.0.0.1:0`, i.e. an OS-assigned ephemeral port) against two remote
//! actor threads dialing back in over `TcpStream::connect`, training
//! CartPole PPO to completion with clean shutdown — the loopback-verification
//! acceptance criterion for issue #281. Real multi-host runs (distinct
//! `alc-*` hosts) are documented as a runbook in
//! `docs/DISTRIBUTED_TRAINING_DESIGN.md` rather than automated, since this
//! environment cannot reach those hosts.

use std::{
    io::{self, Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::{Deserialize, Serialize};

use super::actor_learner::{ActorChannels, ActorHandle, ActorStats};
use crate::multi_agent::{ControlMessage, Experience, PolicyBroadcast};

// ---------------------------------------------------------------------------
// The transport seam: traits `actor_thread` / `learner_loop` are generic
// over. Blanket-implemented for the in-process crossbeam types below, so the
// existing default path satisfies them with zero behavior change; the TCP
// types further down implement them for the networked path.
// ---------------------------------------------------------------------------

/// Actor-side sender half of the actors→learner experience stream.
///
/// [`ActorChannels::experience_tx`](super::actor_learner::ActorChannels)'s
/// bound. Implemented for [`crossbeam_channel::Sender<Experience>`] (the
/// default, unchanged in-process path) and [`TcpExperienceSender`] (networked).
pub trait ExperienceSender: Send {
    /// Send one experience. Returns `false` when the receiving side is gone
    /// (disconnected channel, or — for a network transport — a dead
    /// connection), mirroring `crossbeam_channel::Sender::send(..).is_ok()`.
    fn send(&self, experience: Experience) -> bool;
}

/// Learner-side sender half of one actor's broadcast channel.
///
/// [`ActorHandle`]'s `BR` bound.
/// Implemented for [`crossbeam_channel::Sender<PolicyBroadcast>`] (default)
/// and [`TcpPeerSender`] (networked).
pub trait BroadcastSender: Send {
    /// Send one policy broadcast to this actor. Returns `false` when the
    /// actor is gone.
    fn send(&self, broadcast: PolicyBroadcast) -> bool;
}

/// Learner-side sender half of one actor's control channel.
///
/// [`ActorHandle`]'s `CTRL` bound.
/// Implemented for [`crossbeam_channel::Sender<ControlMessage>`] (default)
/// and [`TcpPeerSender`] (networked).
pub trait ControlSender: Send {
    /// Send one control message to this actor. Returns `false` when the
    /// actor is gone.
    fn send(&self, message: ControlMessage) -> bool;
}

impl ExperienceSender for Sender<Experience> {
    fn send(&self, experience: Experience) -> bool {
        Sender::send(self, experience).is_ok()
    }
}

impl BroadcastSender for Sender<PolicyBroadcast> {
    fn send(&self, broadcast: PolicyBroadcast) -> bool {
        Sender::send(self, broadcast).is_ok()
    }
}

impl ControlSender for Sender<ControlMessage> {
    fn send(&self, message: ControlMessage) -> bool {
        Sender::send(self, message).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Wire encoding
// ---------------------------------------------------------------------------

/// One frame of the actor↔learner wire protocol.
///
/// Directional by convention (see the module docs): actor→learner sockets
/// only ever carry [`WireMessage::Experience`]; learner→actor sockets only
/// ever carry [`WireMessage::Broadcast`] / [`WireMessage::Control`]. Bundled
/// into one enum (rather than three separate wire types) so both directions
/// share one `write_message`/`read_message` pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WireMessage {
    /// One actor→learner experience tuple.
    Experience(Experience),
    /// One learner→actor policy broadcast.
    Broadcast(PolicyBroadcast),
    /// One learner→actor control message.
    Control(ControlMessage),
}

/// Encode `msg` with `bincode`'s `serde` integration and write it to
/// `stream` as a 4-byte big-endian length prefix followed by the payload.
///
/// # Errors
/// Returns an error when encoding fails (never expected for these plain
/// types) or the underlying socket write fails.
fn write_message(stream: &mut impl Write, msg: &WireMessage) -> io::Result<()> {
    let bytes = bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()
}

/// Read one length-prefixed [`WireMessage`] frame from `stream`.
///
/// Returns `Ok(None)` on a clean EOF at a frame boundary (the peer closed
/// the connection between frames — the expected shutdown path), and
/// `Err(..)` for any other I/O failure or decode error (including a
/// truncated frame, which is a genuine error rather than a clean close).
///
/// # Errors
/// Returns an error when the socket read fails (other than a clean EOF
/// before any bytes of the next frame) or the payload fails to decode.
fn read_message(stream: &mut impl Read) -> io::Result<Option<WireMessage>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    let (msg, _consumed) = bincode::serde::decode_from_slice(&buf, bincode::config::standard())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(msg))
}

// ---------------------------------------------------------------------------
// TCP sender types
// ---------------------------------------------------------------------------

/// Shut down `stream` (both directions) when `handle` is the last
/// `Arc` reference to it, ignoring any error (the socket may already be
/// gone).
///
/// This is the fix for a real deadlock class in this transport:
/// `TcpStream::try_clone` duplicates the underlying OS socket, and merely
/// *dropping* (closing) one clone does **not** send a TCP `FIN` while
/// another local clone of the same socket stays open — the read/write
/// halves used by the demux/reader background threads are exactly such
/// clones. An explicit `shutdown()` call, by contrast, acts on the shared
/// socket itself (not the per-fd refcount): it immediately unblocks any
/// local thread blocked reading *any* clone of this socket, **and** sends a
/// real `FIN` so the remote peer's blocked read sees a clean EOF too. Used
/// by both [`TcpExperienceSender`] and [`TcpPeerSender`]'s `Drop` impls so
/// each side's demux/reader thread reliably observes EOF once the sender
/// that shares its socket is done, instead of only when every dup'd fd
/// happens to be closed (which can never happen if the reader thread that
/// would close its own clone is exactly the thread blocked waiting for
/// EOF to exit).
fn shutdown_when_last_ref(handle: &Arc<Mutex<TcpStream>>) {
    if Arc::strong_count(handle) == 1
        && let Ok(stream) = handle.lock()
    {
        let _ = stream.shutdown(Shutdown::Both);
    }
}

/// Actor-side [`ExperienceSender`]: writes each [`Experience`] as one
/// [`WireMessage::Experience`] frame to the learner's socket.
///
/// `actor_thread` only ever calls `send` from its own single thread, so the
/// `Mutex` here is never contended — it exists to satisfy `Sync`-free
/// interior mutability for the `&self`-taking trait method, not for real
/// concurrent access.
#[derive(Clone)]
pub struct TcpExperienceSender(Arc<Mutex<TcpStream>>);

impl ExperienceSender for TcpExperienceSender {
    fn send(&self, experience: Experience) -> bool {
        let Ok(mut guard) = self.0.lock() else {
            return false;
        };
        write_message(&mut *guard, &WireMessage::Experience(experience)).is_ok()
    }
}

impl Drop for TcpExperienceSender {
    fn drop(&mut self) {
        shutdown_when_last_ref(&self.0);
    }
}

/// Learner-side sender for both [`PolicyBroadcast`] and [`ControlMessage`]
/// frames to one remote actor connection. One type serves both
/// [`BroadcastSender`] and [`ControlSender`] because both directions share
/// the same socket (learner → actor) and the same framing.
#[derive(Clone)]
pub struct TcpPeerSender(Arc<Mutex<TcpStream>>);

impl BroadcastSender for TcpPeerSender {
    fn send(&self, broadcast: PolicyBroadcast) -> bool {
        let Ok(mut guard) = self.0.lock() else {
            return false;
        };
        write_message(&mut *guard, &WireMessage::Broadcast(broadcast)).is_ok()
    }
}

impl ControlSender for TcpPeerSender {
    fn send(&self, message: ControlMessage) -> bool {
        let Ok(mut guard) = self.0.lock() else {
            return false;
        };
        write_message(&mut *guard, &WireMessage::Control(message)).is_ok()
    }
}

impl Drop for TcpPeerSender {
    fn drop(&mut self) {
        shutdown_when_last_ref(&self.0);
    }
}

/// An [`ActorHandle`] to a remote actor connected over TCP — the return
/// type of [`accept_actors_over_tcp`], named to keep that signature (and
/// callers') types readable.
pub type TcpActorHandle = ActorHandle<TcpPeerSender, TcpPeerSender>;

// ---------------------------------------------------------------------------
// Actor side: dial the learner
// ---------------------------------------------------------------------------

/// Dial `addr` (the learner's listening address) and build the
/// [`ActorChannels`] a remote actor process hands to
/// [`actor_thread`](super::actor_learner::actor_thread) — the exact same
/// function the in-process path uses, unmodified.
///
/// Spawns one background "demux" thread that reads incoming
/// [`WireMessage::Broadcast`] / [`WireMessage::Control`] frames off the
/// socket and forwards them onto ordinary local `crossbeam_channel` senders,
/// so `actor_thread`'s `broadcast_rx` / `control_rx` stay the exact concrete
/// crossbeam types it always used — only `experience_tx` is TCP-backed. The
/// demux thread exits (and is safe to `join`) once the local
/// `broadcast_rx`/`control_rx` receivers are dropped (i.e. once
/// `actor_thread` returns and its `ActorChannels` goes out of scope) or the
/// learner closes the connection, whichever comes first.
///
/// # Errors
/// Returns an error when the connection cannot be established.
pub fn connect_actor_transport(
    addr: impl ToSocketAddrs,
) -> Result<(ActorChannels<TcpExperienceSender>, thread::JoinHandle<()>)> {
    let stream =
        TcpStream::connect(addr).map_err(|e| anyhow!("failed to connect to learner: {e}"))?;
    stream.set_nodelay(true).ok();
    let mut read_stream = stream
        .try_clone()
        .map_err(|e| anyhow!("failed to clone tcp stream for demux: {e}"))?;

    let (broadcast_tx, broadcast_rx) = unbounded();
    let (control_tx, control_rx) = unbounded();

    let demux = thread::Builder::new()
        .name("thrust-actor-tcp-demux".to_string())
        .spawn(move || {
            loop {
                match read_message(&mut read_stream) {
                    Ok(Some(WireMessage::Broadcast(broadcast))) => {
                        if broadcast_tx.send(broadcast).is_err() {
                            break;
                        }
                    }
                    Ok(Some(WireMessage::Control(message))) => {
                        if control_tx.send(message).is_err() {
                            break;
                        }
                    }
                    Ok(Some(WireMessage::Experience(_))) => {
                        // Wrong direction for a learner→actor socket; a
                        // conformant learner never sends this. Ignore rather
                        // than tear down the connection over a protocol
                        // violation that costs nothing to skip.
                        tracing::warn!(
                            "actor tcp demux: ignoring unexpected Experience frame from learner"
                        );
                    }
                    Ok(None) => break, // learner closed the connection cleanly
                    Err(e) => {
                        tracing::warn!("actor tcp demux: read error, stopping: {e}");
                        break;
                    }
                }
            }
        })
        .expect("failed to spawn tcp demux thread");

    let experience_tx = TcpExperienceSender(Arc::new(Mutex::new(stream)));
    Ok((ActorChannels { experience_tx, broadcast_rx, control_rx }, demux))
}

// ---------------------------------------------------------------------------
// Learner side: accept remote actors
// ---------------------------------------------------------------------------

/// Accept `num_actors` remote actor connections on `listener` and build the
/// shared experience receiver plus per-actor [`ActorHandle`]s
/// [`learner_loop`](super::actor_learner::learner_loop) consumes —
/// unmodified from the in-process path (only the `BR`/`CTRL` type
/// parameters differ, inferred as [`TcpPeerSender`]).
///
/// Blocks accepting connections one at a time until `num_actors` remote
/// actors have dialed in. Each accepted connection gets one background
/// reader thread that decodes incoming [`WireMessage::Experience`] frames
/// and forwards them onto a clone of the shared experience sender feeding
/// the returned `experience_rx`; that thread *is* the resulting
/// [`ActorHandle`]'s join handle and returns best-effort [`ActorStats`]
/// (steps/episodes forwarded) once the connection closes — so
/// [`ActorHandle::join`](super::actor_learner::ActorHandle::join) works the
/// same way for a remote actor as for a local one, and an actor dying
/// mid-training surfaces as that column's reader thread exiting rather than
/// a learner panic.
///
/// # Errors
/// Returns an error when accepting a connection fails.
pub fn accept_actors_over_tcp(
    listener: &TcpListener,
    num_actors: usize,
) -> Result<(Receiver<Experience>, Vec<TcpActorHandle>)> {
    let (experience_tx, experience_rx) = unbounded();
    let mut handles = Vec::with_capacity(num_actors);

    for actor_id in 0..num_actors {
        let (stream, _peer_addr) = listener
            .accept()
            .map_err(|e| anyhow!("failed to accept actor connection: {e}"))?;
        stream.set_nodelay(true).ok();
        let mut read_stream = stream
            .try_clone()
            .map_err(|e| anyhow!("failed to clone tcp stream for actor {actor_id}: {e}"))?;
        let peer_sender = TcpPeerSender(Arc::new(Mutex::new(stream)));
        let exp_tx = experience_tx.clone();

        let join = thread::Builder::new()
            .name(format!("thrust-learner-tcp-reader-{actor_id}"))
            .spawn(move || -> Result<ActorStats> {
                let mut stats = ActorStats { actor_id, ..Default::default() };
                loop {
                    match read_message(&mut read_stream) {
                        Ok(Some(WireMessage::Experience(experience))) => {
                            stats.steps_sent += 1;
                            if experience.is_done() {
                                stats.episodes_completed += 1;
                            }
                            if exp_tx.send(experience).is_err() {
                                break; // learner dropped experience_rx; nothing left to do
                            }
                        }
                        Ok(Some(_)) => {
                            // Wrong direction for an actor→learner socket;
                            // ignore rather than abort the run.
                            tracing::warn!(
                                actor_id,
                                "learner tcp reader: ignoring unexpected non-Experience frame"
                            );
                        }
                        Ok(None) => break, // actor closed the connection cleanly
                        Err(e) => {
                            tracing::warn!(
                                actor_id,
                                "learner tcp reader: read error, stopping: {e}"
                            );
                            break;
                        }
                    }
                }
                Ok(stats)
            })
            .expect("failed to spawn tcp learner reader thread");

        handles.push(ActorHandle::from_parts(actor_id, join, peer_sender.clone(), peer_sender));
    }

    Ok((experience_rx, handles))
}

#[cfg(test)]
mod tests {
    use std::{net::TcpListener, time::Duration};

    use super::*;

    /// Round-trip every [`WireMessage`] variant through `write_message` /
    /// `read_message` over a real loopback socket pair, proving the wire
    /// encoding (not just in-memory serde) preserves every field.
    #[test]
    fn wire_message_roundtrips_over_loopback_socket() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let experience = Experience::new(
            2,
            vec![0.1, -0.2, 0.3, 0.4],
            vec![1],
            1.5,
            vec![0.5, 0.6, 0.7, 0.8],
            true,
            false,
            0.25,
            -0.42,
        );
        let broadcast = PolicyBroadcast::Bytes { version: 9, bytes: Arc::new(vec![1, 2, 3, 4, 5]) };
        let control = ControlMessage::Shutdown;

        let mut writer = client;
        write_message(&mut writer, &WireMessage::Experience(experience.clone())).unwrap();
        write_message(&mut writer, &WireMessage::Broadcast(broadcast.clone())).unwrap();
        write_message(&mut writer, &WireMessage::Control(control.clone())).unwrap();
        drop(writer); // signal clean EOF after the three frames

        match read_message(&mut server).unwrap().unwrap() {
            WireMessage::Experience(got) => {
                assert_eq!(got.agent_id, experience.agent_id);
                assert_eq!(got.observation, experience.observation);
                assert_eq!(got.action, experience.action);
                assert_eq!(got.reward, experience.reward);
                assert_eq!(got.next_observation, experience.next_observation);
                assert_eq!(got.terminated, experience.terminated);
                assert_eq!(got.truncated, experience.truncated);
                assert_eq!(got.value, experience.value);
                assert_eq!(got.log_prob, experience.log_prob);
            }
            other => panic!("expected Experience, got {other:?}"),
        }

        match read_message(&mut server).unwrap().unwrap() {
            WireMessage::Broadcast(PolicyBroadcast::Bytes { version, bytes }) => {
                assert_eq!(version, 9);
                assert_eq!(*bytes, vec![1, 2, 3, 4, 5]);
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }

        match read_message(&mut server).unwrap().unwrap() {
            WireMessage::Control(ControlMessage::Shutdown) => {}
            other => panic!("expected Control(Shutdown), got {other:?}"),
        }

        // Clean EOF at the next frame boundary decodes as `None`, not an error.
        assert!(read_message(&mut server).unwrap().is_none());
    }

    /// A truncated frame (length prefix present, payload cut short) is a
    /// genuine I/O error, not a clean `None` — distinguishing a mid-frame
    /// disconnect (bug / crash) from a between-frames shutdown.
    #[test]
    fn read_message_errors_on_truncated_frame() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        // Claim a 100-byte payload, then only send 3 bytes and close.
        client.write_all(&100u32.to_be_bytes()).unwrap();
        client.write_all(&[1, 2, 3]).unwrap();
        drop(client);

        let err = read_message(&mut server).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    /// Actor disconnect mid-training: the learner survives and the shared
    /// experience channel keeps working for the surviving actor.
    ///
    /// Accepts 2 remote actor connections, sends a few experiences on each,
    /// then abruptly drops actor 0's connection (simulating a crash) while
    /// actor 1's connection stays open. Asserts: (1) actor 0's
    /// [`ActorHandle::join`] returns cleanly (no hang, no panic) with the
    /// steps it forwarded before disconnecting; (2) the shared
    /// `experience_rx` is unaffected by actor 0's disconnect — a short
    /// `recv_timeout` on it (with no traffic pending) reports a plain
    /// timeout, not `Disconnected`, since actor 1's sender clone keeps the
    /// channel alive. This is the transport-level guarantee `learner_loop`'s
    /// fill loop relies on: one actor going away does not sever the shared
    /// channel the other columns still feed.
    #[test]
    fn learner_survives_one_actor_disconnecting_mid_training() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let mut client0 = TcpStream::connect(addr).unwrap();
        let mut client1 = TcpStream::connect(addr).unwrap();

        let (experience_rx, mut handles) = accept_actors_over_tcp(&listener, 2).unwrap();

        // Both actors send a couple of experiences.
        let make_exp = |agent_id: usize| {
            Experience::new(
                agent_id,
                vec![0.0; 4],
                vec![0],
                1.0,
                vec![0.0; 4],
                false,
                false,
                0.0,
                0.0,
            )
        };
        write_message(&mut client0, &WireMessage::Experience(make_exp(0))).unwrap();
        write_message(&mut client0, &WireMessage::Experience(make_exp(0))).unwrap();
        write_message(&mut client1, &WireMessage::Experience(make_exp(1))).unwrap();

        // Drain those 3 so the reader threads have processed them before we
        // pull the plug (avoids a race between the disconnect and the last
        // read on actor 0's connection).
        for _ in 0..3 {
            experience_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("expected buffered experience");
        }

        // Simulate actor 0 crashing: close its connection abruptly.
        drop(client0);

        // Actor 0's handle joins cleanly (reader thread sees EOF, exits,
        // returns its forwarded-step count) without hanging.
        let actor0 = handles.remove(0);
        let stats0 = actor0
            .join()
            .expect("learner-side reader thread for a disconnected actor must not panic");
        assert_eq!(stats0.actor_id, 0);
        assert_eq!(stats0.steps_sent, 2, "actor 0's reader should report the 2 steps it forwarded");

        // The shared channel survives: actor 1's sender clone is still
        // alive, so a short timeout reports Timeout (no more traffic
        // pending), never Disconnected.
        match experience_rx.recv_timeout(Duration::from_millis(200)) {
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            other => panic!(
                "expected the shared experience channel to survive actor 0's disconnect \
                 (Timeout, no more traffic), got {other:?}"
            ),
        }

        // Actor 1 can still send and be received — the learner truly kept
        // training on the surviving actor.
        write_message(&mut client1, &WireMessage::Experience(make_exp(1))).unwrap();
        let got = experience_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("actor 1 still delivers");
        assert_eq!(got.agent_id, 1);

        // Clean up actor 1's handle too.
        drop(client1);
        let actor1 = handles.remove(0);
        let stats1 = actor1.join().expect("actor 1 reader thread must not panic");
        assert_eq!(stats1.actor_id, 1);
        assert_eq!(stats1.steps_sent, 2);
    }
}
