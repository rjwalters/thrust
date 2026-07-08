#!/usr/bin/env python3
"""ALE subprocess worker for thrust's ``env-atari`` feature.

thrust's ``AtariEnv`` (Rust) spawns this script and drives it over a tiny
length-prefixed binary protocol on stdin/stdout. This keeps the emulator —
Farama's ``ale-py``, which is GPL-2.0-or-later — as a *separate program* so the
thrust crate stays MIT/Apache and pulls no native build dependency (Option D of
docs/ALE_BINDING_STRATEGY.md).

This script is a standalone program, not a package: no ``setup.py``, no
``__init__.py``. Its only third-party dependency is ``ale-py`` (which brings
``numpy``); everything else is the Python standard library.

Wire format (must stay in lock-step with ``src/env/games/atari/protocol.rs``):

  * every message = 4-byte little-endian length prefix + that many payload bytes
  * payload[0] is a tag byte; the rest is tag-specific

  Rust -> Python (commands):
    0x01 RESET          8-byte LE u64 seed
    0x02 STEP           4-byte LE i32 action index (into the minimal action set)
    0x03 CLONE_STATE    (empty)
    0x04 RESTORE_STATE  opaque state bytes from a prior STATE response
    0x05 CLOSE          (empty)

  Python -> Rust (responses):
    0x81 OBS    1-byte terminated, 1-byte truncated, 4-byte LE f32 reward,
                4-byte LE u32 lives,
                then H*W*C little-endian f32 pixels (raw 210x160x3 RGB)
    0x82 STATE  opaque ALE cloneSystemState blob
    0x83 ERROR  UTF-8 error string
"""

import os
import struct
import sys

# Command tags (Rust -> Python).
TAG_RESET = 0x01
TAG_STEP = 0x02
TAG_CLONE_STATE = 0x03
TAG_RESTORE_STATE = 0x04
TAG_CLOSE = 0x05

# Response tags (Python -> Rust).
TAG_OBS = 0x81
TAG_STATE = 0x82
TAG_ERROR = 0x83


def _read_exact(n):
    """Read exactly ``n`` bytes from stdin, or return ``None`` on clean EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_frame():
    """Read one length-prefixed frame payload, or ``None`` at EOF."""
    header = _read_exact(4)
    if header is None:
        return None
    (length,) = struct.unpack("<I", header)
    if length == 0:
        return b""
    payload = _read_exact(length)
    if payload is None:
        return None
    return payload


def write_frame(payload):
    """Write one length-prefixed frame and flush."""
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def send_error(message):
    write_frame(bytes([TAG_ERROR]) + message.encode("utf-8"))


def send_state(blob):
    write_frame(bytes([TAG_STATE]) + bytes(blob))


def send_obs(terminated, truncated, reward, lives, pixel_bytes):
    header = bytes([TAG_OBS, 1 if terminated else 0, 1 if truncated else 0])
    write_frame(
        header
        + struct.pack("<f", float(reward))
        + struct.pack("<I", int(lives) & 0xFFFFFFFF)
        + pixel_bytes
    )


def _resolve_rom(rom_id):
    """Resolve a ROM path from ALE_ROM_PATH, or ``None`` to defer to ale-py.

    ``ALE_ROM_PATH`` may point at a specific ``.bin`` file or a directory
    containing ``<rom_id>.bin``. When unset, return ``None`` so the caller can
    fall back to ale-py's own ROM resolution.
    """
    path = os.environ.get("ALE_ROM_PATH")
    if not path:
        return None
    if os.path.isdir(path):
        candidate = os.path.join(path, rom_id + ".bin")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            "ROM '%s' not found in ALE_ROM_PATH directory %s" % (rom_id, path)
        )
    if os.path.isfile(path):
        return path
    raise FileNotFoundError("ALE_ROM_PATH '%s' is neither a file nor a directory" % path)


def _resolve_rom_path(rom_id):
    """Resolve the ROM file path for ``rom_id`` (env override or ale-py registry).

    Raises ``FileNotFoundError`` with an actionable message if nothing resolves.
    """
    rom_path = _resolve_rom(rom_id)
    if rom_path is not None:
        return rom_path
    # Defer to ale-py's bundled ROM resolution (AutoROM / packaged ROMs).
    try:
        from ale_py import roms as ale_roms
    except Exception as exc:
        raise RuntimeError(
            "ROM '%s' could not be resolved and ale-py's ROM registry is "
            "unavailable (%s). Set ALE_ROM_PATH or run "
            "'AutoROM --accept-license'." % (rom_id, exc)
        )
    # ale-py exposes ROMs as attributes named in TitleCase (e.g. Pong).
    attr = rom_id[:1].upper() + rom_id[1:]
    rom_path = getattr(ale_roms, attr, None)
    if rom_path is None and hasattr(ale_roms, "get_rom_path"):
        rom_path = ale_roms.get_rom_path(rom_id)
    if rom_path is None:
        raise FileNotFoundError(
            "ROM '%s' not found in ale-py's ROM registry. Set ALE_ROM_PATH "
            "or run 'AutoROM --accept-license'." % rom_id
        )
    return rom_path


def _load_rom(ale, rom_path, seed):
    """(Re)load ``rom_path`` into ``ale`` with ``seed`` as the RNG seed.

    ALE only applies ``random_seed`` at ``loadROM`` time, so the seed must be
    set *before* the load. Callers reload the ROM whenever the seed changes so
    that ``AtariEnv::new(game, seed)`` actually controls the emulator RNG.
    """
    ale.setInt("random_seed", int(seed) & 0x7FFFFFFF)
    ale.loadROM(rom_path)


def _load_ale(rom_id, seed):
    """Import ale-py and return ``(ALEInterface, minimal_actions, rom_path)``.

    Raises with an actionable message on any failure so the Rust side can turn
    it into a typed error.
    """
    try:
        from ale_py import ALEInterface
    except Exception as exc:  # pragma: no cover - exercised only with ale-py absent
        raise RuntimeError(
            "ale-py could not be imported (%s). Install it with: pip install ale-py" % exc
        )

    ale = ALEInterface()
    rom_path = _resolve_rom_path(rom_id)
    _load_rom(ale, rom_path, seed)
    minimal_actions = list(ale.getMinimalActionSet())
    return ale, minimal_actions, rom_path


def _screen_bytes(ale):
    """Return the current RGB screen as row-major little-endian f32 bytes."""
    screen = ale.getScreenRGB()  # numpy uint8 array, shape (H, W, 3)
    return screen.astype("<f4").tobytes()


def _clone_state_bytes(ale):
    """Serialise the full ALE system state to opaque bytes.

    Current ale-py (>= 0.10) serialises via ``ALEState.serialize()``. Older
    builds exposed a top-level ``encodeState``; keep it as a legacy fallback.
    """
    state = ale.cloneSystemState()
    if hasattr(state, "serialize"):
        return bytes(state.serialize())
    if hasattr(ale, "encodeState"):
        return bytes(ale.encodeState(state))
    raise RuntimeError("this ale-py build exposes no state serialisation API")


def _restore_state_bytes(ale, blob):
    """Restore the ALE system state from opaque bytes produced by clone.

    Current ale-py reconstructs an ``ALEState`` from the serialised blob and
    hands it to ``restoreSystemState``. Older builds exposed a top-level
    ``decodeState``; keep it as a legacy fallback.
    """
    blob = bytes(blob)
    try:
        from ale_py import ALEState
    except Exception:  # pragma: no cover - only on very old ale-py
        ALEState = None
    if ALEState is not None:
        ale.restoreSystemState(ALEState(blob))
        return
    if hasattr(ale, "decodeState"):
        ale.restoreSystemState(ale.decodeState(blob))
        return
    raise RuntimeError("this ale-py build exposes no state deserialisation API")


def main():
    if len(sys.argv) < 2:
        send_error("usage: ale_worker.py <rom_id>")
        return 2
    rom_id = sys.argv[1]

    try:
        ale, minimal_actions, rom_path = _load_ale(rom_id, 0)
    except Exception as exc:
        send_error(str(exc))
        return 1

    # Track the seed the ROM is currently loaded under so RESET only pays the
    # reload cost when the seed actually changes (ALE applies random_seed at
    # loadROM time only).
    loaded_seed = 0

    while True:
        payload = read_frame()
        if payload is None:  # peer closed the pipe
            return 0
        if len(payload) == 0:
            send_error("received an empty (untagged) frame")
            return 1
        tag = payload[0]
        body = payload[1:]

        try:
            if tag == TAG_RESET:
                (seed,) = struct.unpack("<Q", body)
                # ALE only honours random_seed at loadROM time, so reload the
                # ROM whenever the requested seed differs from the loaded one.
                if seed != loaded_seed:
                    _load_rom(ale, rom_path, seed)
                    loaded_seed = seed
                ale.reset_game()
                send_obs(ale.game_over(), False, 0.0, ale.lives(), _screen_bytes(ale))
            elif tag == TAG_STEP:
                (action_index,) = struct.unpack("<i", body)
                if 0 <= action_index < len(minimal_actions):
                    action = minimal_actions[action_index]
                else:
                    action = minimal_actions[0]
                reward = ale.act(action)
                terminated = ale.game_over()
                send_obs(terminated, False, reward, ale.lives(), _screen_bytes(ale))
            elif tag == TAG_CLONE_STATE:
                send_state(_clone_state_bytes(ale))
            elif tag == TAG_RESTORE_STATE:
                _restore_state_bytes(ale, body)
                send_obs(ale.game_over(), False, 0.0, ale.lives(), _screen_bytes(ale))
            elif tag == TAG_CLOSE:
                return 0
            else:
                send_error("unknown command tag 0x%02x" % tag)
                return 1
        except Exception as exc:  # pragma: no cover - runtime emulator errors
            send_error("worker error handling tag 0x%02x: %s" % (tag, exc))
            return 1


if __name__ == "__main__":
    sys.exit(main())
