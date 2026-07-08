//! Frame protocol codec for the `env-atari` subprocess binding.
//!
//! `AtariEnv` (the Rust parent) talks to `envs/atari/ale_worker.py` (the
//! `ale-py` child) over the child's stdin/stdout using a tiny, dependency-free
//! binary protocol. This module is the single source of truth for that wire
//! format so both the Rust codec and the Python worker stay in lock-step.
//!
//! # Framing
//!
//! Every message is a **4-byte little-endian unsigned length prefix** followed
//! by exactly that many payload bytes. The first payload byte is a *tag* that
//! identifies the message; the remaining bytes are tag-specific. Pure
//! `struct`-packing — no msgpack, no serde — keeps the Python side stdlib-only
//! and makes the Rust codec trivially unit-testable without spawning a process.
//!
//! # Messages
//!
//! Parent → child ([`Command`]):
//!
//! | Tag  | Name            | Payload                                   |
//! |------|-----------------|-------------------------------------------|
//! | 0x01 | `Reset`         | 8-byte LE `u64` seed                      |
//! | 0x02 | `Step`          | 4-byte LE `i32` action index              |
//! | 0x03 | `CloneState`    | (empty)                                   |
//! | 0x04 | `RestoreState`  | opaque state bytes from a prior `State`   |
//! | 0x05 | `Close`         | (empty — child flushes and exits)         |
//!
//! Child → parent ([`Response`]):
//!
//! | Tag  | Name    | Payload                                                        |
//! |------|---------|---------------------------------------------------------------|
//! | 0x81 | `Obs`   | 1-byte terminated, 1-byte truncated, 4-byte LE `f32` reward, then `H × W × C` LE `f32` pixels |
//! | 0x82 | `State` | opaque ALE `cloneSystemState` blob                            |
//! | 0x83 | `Error` | UTF-8 error string                                            |

use std::io::{self, Read, Write};

/// Tag byte for [`Command::Reset`].
pub const TAG_RESET: u8 = 0x01;
/// Tag byte for [`Command::Step`].
pub const TAG_STEP: u8 = 0x02;
/// Tag byte for [`Command::CloneState`].
pub const TAG_CLONE_STATE: u8 = 0x03;
/// Tag byte for [`Command::RestoreState`].
pub const TAG_RESTORE_STATE: u8 = 0x04;
/// Tag byte for [`Command::Close`].
pub const TAG_CLOSE: u8 = 0x05;

/// Tag byte for [`Response::Obs`].
pub const TAG_OBS: u8 = 0x81;
/// Tag byte for [`Response::State`].
pub const TAG_STATE: u8 = 0x82;
/// Tag byte for [`Response::Error`].
pub const TAG_ERROR: u8 = 0x83;

/// Observation frame height (raw ALE screen), in pixels.
pub const OBS_HEIGHT: usize = 210;
/// Observation frame width (raw ALE screen), in pixels.
pub const OBS_WIDTH: usize = 160;
/// Observation frame channel count (RGB).
pub const OBS_CHANNELS: usize = 3;
/// Total number of `f32` elements in one raw observation
/// (`OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS`).
pub const OBS_LEN: usize = OBS_HEIGHT * OBS_WIDTH * OBS_CHANNELS;

/// A message sent from the Rust parent to the Python child.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Reset the emulator with the given RNG seed.
    Reset(u64),
    /// Step the emulator with an action index into the minimal action set.
    Step(i32),
    /// Request an opaque snapshot of the full emulator state.
    CloneState,
    /// Restore the emulator to a previously-snapshotted opaque state.
    RestoreState(Vec<u8>),
    /// Ask the child to flush and exit cleanly.
    Close,
}

/// A message sent from the Python child back to the Rust parent.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    /// An observation frame plus the transition flags and reward.
    Obs {
        /// Whether the episode terminated on this transition.
        terminated: bool,
        /// Whether the episode was truncated on this transition.
        truncated: bool,
        /// Reward accumulated over the transition.
        reward: f32,
        /// Raw pixel data, row-major `H × W × C`, values in `0.0..=255.0`.
        pixels: Vec<f32>,
    },
    /// An opaque emulator-state blob (from `ale.cloneSystemState()`).
    State(Vec<u8>),
    /// A human-readable error reported by the worker.
    Error(String),
}

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Encode a [`Command`] into a complete length-prefixed frame.
#[must_use]
pub fn encode_command(cmd: &Command) -> Vec<u8> {
    frame(&command_payload(cmd))
}

fn command_payload(cmd: &Command) -> Vec<u8> {
    match cmd {
        Command::Reset(seed) => {
            let mut p = Vec::with_capacity(9);
            p.push(TAG_RESET);
            p.extend_from_slice(&seed.to_le_bytes());
            p
        }
        Command::Step(action) => {
            let mut p = Vec::with_capacity(5);
            p.push(TAG_STEP);
            p.extend_from_slice(&action.to_le_bytes());
            p
        }
        Command::CloneState => vec![TAG_CLONE_STATE],
        Command::RestoreState(bytes) => {
            let mut p = Vec::with_capacity(1 + bytes.len());
            p.push(TAG_RESTORE_STATE);
            p.extend_from_slice(bytes);
            p
        }
        Command::Close => vec![TAG_CLOSE],
    }
}

/// Decode a [`Command`] from a frame payload (the bytes *after* the length
/// prefix).
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidData`] if the payload is empty, carries an
/// unknown tag, or has the wrong length for its tag.
pub fn decode_command(payload: &[u8]) -> io::Result<Command> {
    let (&tag, rest) =
        payload.split_first().ok_or_else(|| invalid_data("empty command payload"))?;
    match tag {
        TAG_RESET => {
            let bytes: [u8; 8] =
                rest.try_into().map_err(|_| invalid_data("RESET payload must be 8 bytes"))?;
            Ok(Command::Reset(u64::from_le_bytes(bytes)))
        }
        TAG_STEP => {
            let bytes: [u8; 4] =
                rest.try_into().map_err(|_| invalid_data("STEP payload must be 4 bytes"))?;
            Ok(Command::Step(i32::from_le_bytes(bytes)))
        }
        TAG_CLONE_STATE => Ok(Command::CloneState),
        TAG_RESTORE_STATE => Ok(Command::RestoreState(rest.to_vec())),
        TAG_CLOSE => Ok(Command::Close),
        other => Err(invalid_data(format!("unknown command tag {other:#04x}"))),
    }
}

/// Encode a [`Response`] into a complete length-prefixed frame.
#[must_use]
pub fn encode_response(resp: &Response) -> Vec<u8> {
    frame(&response_payload(resp))
}

fn response_payload(resp: &Response) -> Vec<u8> {
    match resp {
        Response::Obs { terminated, truncated, reward, pixels } => {
            let mut p = Vec::with_capacity(7 + pixels.len() * 4);
            p.push(TAG_OBS);
            p.push(u8::from(*terminated));
            p.push(u8::from(*truncated));
            p.extend_from_slice(&reward.to_le_bytes());
            for px in pixels {
                p.extend_from_slice(&px.to_le_bytes());
            }
            p
        }
        Response::State(bytes) => {
            let mut p = Vec::with_capacity(1 + bytes.len());
            p.push(TAG_STATE);
            p.extend_from_slice(bytes);
            p
        }
        Response::Error(msg) => {
            let mut p = Vec::with_capacity(1 + msg.len());
            p.push(TAG_ERROR);
            p.extend_from_slice(msg.as_bytes());
            p
        }
    }
}

/// Decode a [`Response`] from a frame payload (the bytes *after* the length
/// prefix).
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidData`] if the payload is empty, carries an
/// unknown tag, or is malformed for its tag (e.g. an `Obs` whose pixel byte
/// count is not a multiple of 4).
pub fn decode_response(payload: &[u8]) -> io::Result<Response> {
    let (&tag, rest) =
        payload.split_first().ok_or_else(|| invalid_data("empty response payload"))?;
    match tag {
        TAG_OBS => {
            if rest.len() < 6 {
                return Err(invalid_data("OBS payload shorter than its 6-byte header"));
            }
            let terminated = rest[0] != 0;
            let truncated = rest[1] != 0;
            let reward = f32::from_le_bytes(rest[2..6].try_into().unwrap());
            let pixel_bytes = &rest[6..];
            if !pixel_bytes.len().is_multiple_of(4) {
                return Err(invalid_data("OBS pixel byte count is not a multiple of 4"));
            }
            let pixels = pixel_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(Response::Obs { terminated, truncated, reward, pixels })
        }
        TAG_STATE => Ok(Response::State(rest.to_vec())),
        TAG_ERROR => Ok(Response::Error(String::from_utf8_lossy(rest).into_owned())),
        other => Err(invalid_data(format!("unknown response tag {other:#04x}"))),
    }
}

fn read_frame<R: Read>(reader: &mut R) -> io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;
    Ok(payload)
}

/// Write a [`Command`] frame to `writer`. Does not flush.
///
/// # Errors
///
/// Propagates any I/O error from the underlying writer.
pub fn write_command<W: Write>(writer: &mut W, cmd: &Command) -> io::Result<()> {
    writer.write_all(&encode_command(cmd))
}

/// Read one [`Command`] frame from `reader` (used by the worker-side mock in
/// tests and by the Python worker's Rust-side test doubles).
///
/// # Errors
///
/// Propagates I/O errors (including [`io::ErrorKind::UnexpectedEof`] when the
/// peer closes the pipe) and decode errors.
pub fn read_command<R: Read>(reader: &mut R) -> io::Result<Command> {
    let payload = read_frame(reader)?;
    decode_command(&payload)
}

/// Write a [`Response`] frame to `writer`. Does not flush.
///
/// # Errors
///
/// Propagates any I/O error from the underlying writer.
pub fn write_response<W: Write>(writer: &mut W, resp: &Response) -> io::Result<()> {
    writer.write_all(&encode_response(resp))
}

/// Read one [`Response`] frame from `reader`.
///
/// # Errors
///
/// Propagates I/O errors (including [`io::ErrorKind::UnexpectedEof`] when the
/// child closes the pipe) and decode errors.
pub fn read_response<R: Read>(reader: &mut R) -> io::Result<Response> {
    let payload = read_frame(reader)?;
    decode_response(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Strip the 4-byte length prefix, asserting it matches the payload length.
    fn payload_of(frame: &[u8]) -> &[u8] {
        let len = u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize;
        assert_eq!(len, frame.len() - 4, "length prefix must match payload length");
        &frame[4..]
    }

    #[test]
    fn codec_reset_round_trip() {
        let frame = encode_command(&Command::Reset(42));
        let decoded = decode_command(payload_of(&frame)).unwrap();
        assert_eq!(decoded, Command::Reset(42));
    }

    #[test]
    fn codec_step_round_trip() {
        let frame = encode_command(&Command::Step(3));
        let decoded = decode_command(payload_of(&frame)).unwrap();
        assert_eq!(decoded, Command::Step(3));

        // Negative action indices survive the i32 round-trip too.
        let frame = encode_command(&Command::Step(-7));
        assert_eq!(decode_command(payload_of(&frame)).unwrap(), Command::Step(-7));
    }

    #[test]
    fn codec_obs_response_round_trip() {
        let pixels: Vec<f32> = (0..OBS_LEN).map(|i| (i % 256) as f32).collect();
        let resp = Response::Obs { terminated: false, truncated: true, reward: 1.0, pixels };
        let frame = encode_response(&resp);
        let decoded = decode_response(payload_of(&frame)).unwrap();
        match decoded {
            Response::Obs { terminated, truncated, reward, pixels } => {
                assert!(!terminated);
                assert!(truncated);
                assert_eq!(reward, 1.0);
                assert_eq!(pixels.len(), OBS_LEN);
                assert_eq!(pixels[1], 1.0);
                assert_eq!(pixels[255], 255.0);
            }
            other => panic!("expected Obs, got {other:?}"),
        }
    }

    #[test]
    fn codec_state_round_trip() {
        let blob = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x7F];
        let frame = encode_response(&Response::State(blob.clone()));
        let decoded = decode_response(payload_of(&frame)).unwrap();
        assert_eq!(decoded, Response::State(blob.clone()));

        // A RestoreState command carries the same opaque bytes back the other way.
        let frame = encode_command(&Command::RestoreState(blob.clone()));
        assert_eq!(decode_command(payload_of(&frame)).unwrap(), Command::RestoreState(blob));
    }

    #[test]
    fn codec_error_response() {
        let resp = Response::Error("missing ROM".to_string());
        let frame = encode_response(&resp);
        let decoded = decode_response(payload_of(&frame)).unwrap();
        assert_eq!(decoded, Response::Error("missing ROM".to_string()));
    }

    #[test]
    fn read_write_stream_round_trip() {
        // Exercise the streaming read/write helpers over an in-memory buffer.
        let mut buf: Vec<u8> = Vec::new();
        write_command(&mut buf, &Command::Reset(7)).unwrap();
        write_command(&mut buf, &Command::CloneState).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        assert_eq!(read_command(&mut cursor).unwrap(), Command::Reset(7));
        assert_eq!(read_command(&mut cursor).unwrap(), Command::CloneState);
    }

    #[test]
    fn decode_rejects_unknown_and_short_frames() {
        assert!(decode_command(&[]).is_err());
        assert!(decode_command(&[0xEE]).is_err());
        assert!(decode_command(&[TAG_RESET, 1, 2]).is_err()); // wrong seed length
        assert!(decode_response(&[TAG_OBS, 0, 0, 0]).is_err()); // truncated header
    }
}
