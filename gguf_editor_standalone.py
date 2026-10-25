#!/usr/bin/env python3
"""Standalone, dependency-free GGUF metadata editor (single file, no gguf-py import).

Edits key/value metadata in place:
  - fixed-size scalar values are patched directly into the file (no other
    byte in the file moves).
  - adding/removing a key, or changing a string/array to a different byte
    size, rebuilds only the KV + tensor-info header; the tensor-data blob
    that follows is then shifted with chunked pread/pwrite *only if the
    header's size actually changed*. Tensor bytes are never parsed,
    decoded, or duplicated into a second file -- this script never even
    imports numpy.

Usage:
  gguf_editor_standalone.py model.gguf                      # curses TUI
  gguf_editor_standalone.py model.gguf --list
  gguf_editor_standalone.py model.gguf --set general.name STRING "New Name"
  gguf_editor_standalone.py model.gguf --remove some.key
"""
from __future__ import annotations

import argparse
import curses
import logging
import os
import struct
import sys
from enum import IntEnum
from typing import Any, BinaryIO

logger = logging.getLogger("gguf-editor")

GGUF_MAGIC = 0x46554747  # b'GGUF' read as little-endian u32
DEFAULT_ALIGNMENT = 32
CHUNK_SIZE = 64 * 1024 * 1024


class ValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


SCALAR_STRUCT_FMT = {
    ValueType.UINT8: 'B', ValueType.INT8: 'b',
    ValueType.UINT16: 'H', ValueType.INT16: 'h',
    ValueType.UINT32: 'I', ValueType.INT32: 'i',
    ValueType.UINT64: 'Q', ValueType.INT64: 'q',
    ValueType.FLOAT32: 'f', ValueType.FLOAT64: 'd',
    ValueType.BOOL: 'B',
}
SCALAR_SIZE = {vt: struct.calcsize('<' + fmt) for vt, fmt in SCALAR_STRUCT_FMT.items()}


class GGUFFormatError(Exception):
    pass


# --------------------------------------------------------------------------
# Low-level byte (de)serialization -- independent of the rest of the repo.
# --------------------------------------------------------------------------

def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n:
        raise GGUFFormatError('Unexpected end of file while parsing GGUF header')
    return buf


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack('<Q', _read_exact(f, 8))
    return _read_exact(f, length).decode('utf-8')


def _pack_string(s: str) -> bytes:
    raw = s.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _read_scalar_bytes(f: BinaryIO, vtype: ValueType) -> bytes:
    return _read_exact(f, SCALAR_SIZE[vtype])


def _decode_scalar(vtype: ValueType, raw: bytes) -> Any:
    if vtype == ValueType.BOOL:
        return raw[0] != 0
    return struct.unpack('<' + SCALAR_STRUCT_FMT[vtype], raw)[0]


def _pack_scalar(vtype: ValueType, value: Any) -> bytes:
    if vtype == ValueType.BOOL:
        return struct.pack('<B', 1 if value else 0)
    return struct.pack('<' + SCALAR_STRUCT_FMT[vtype], value)


def _read_value_bytes(f: BinaryIO, vtype: ValueType) -> tuple[bytes, ValueType | None]:
    """Read one value's raw on-disk bytes (whatever its type). Returns (raw_bytes, array_item_type)."""
    if vtype == ValueType.STRING:
        (length,) = struct.unpack('<Q', _read_exact(f, 8))
        return struct.pack('<Q', length) + _read_exact(f, length), None
    if vtype in SCALAR_STRUCT_FMT:
        return _read_scalar_bytes(f, vtype), None
    if vtype == ValueType.ARRAY:
        (item_type_raw,) = struct.unpack('<I', _read_exact(f, 4))
        item_type = ValueType(item_type_raw)
        (count,) = struct.unpack('<Q', _read_exact(f, 8))
        out = struct.pack('<IQ', item_type_raw, count)
        for _ in range(count):
            item_bytes, _ = _read_value_bytes(f, item_type)
            out += item_bytes
        return out, item_type
    raise GGUFFormatError(f'Unknown value type {vtype}')


def _decode_value(vtype: ValueType, item_type: ValueType | None, raw: bytes) -> Any:
    if vtype == ValueType.STRING:
        (length,) = struct.unpack_from('<Q', raw, 0)
        return raw[8:8 + length].decode('utf-8')
    if vtype in SCALAR_STRUCT_FMT:
        return _decode_scalar(vtype, raw)
    if vtype == ValueType.ARRAY:
        assert item_type is not None
        _item_type_raw, count = struct.unpack_from('<IQ', raw, 0)
        offs = 12
        items = []
        for _ in range(count):
            if item_type == ValueType.STRING:
                (slen,) = struct.unpack_from('<Q', raw, offs)
                items.append(raw[offs + 8:offs + 8 + slen].decode('utf-8'))
                offs += 8 + slen
            else:
                size = SCALAR_SIZE[item_type]
                items.append(_decode_scalar(item_type, raw[offs:offs + size]))
                offs += size
        return items
    raise GGUFFormatError(f'Unknown value type {vtype}')


def _pack_value(vtype: ValueType, item_type: ValueType | None, value: Any) -> bytes:
    if vtype == ValueType.STRING:
        return _pack_string(value)
    if vtype in SCALAR_STRUCT_FMT:
        return _pack_scalar(vtype, value)
    if vtype == ValueType.ARRAY:
        assert item_type is not None
        items = list(value)
        out = struct.pack('<IQ', int(item_type), len(items))
        for item in items:
            out += _pack_string(item) if item_type == ValueType.STRING else _pack_scalar(item_type, item)
        return out
    raise GGUFFormatError(f'Unknown value type {vtype}')


# --------------------------------------------------------------------------
# In-memory model of a GGUF file's metadata (KV list + tensor-info list).
# --------------------------------------------------------------------------

class KVField:
    __slots__ = ('key', 'vtype', 'item_type', 'value', 'value_size')

    def __init__(self, key: str, vtype: ValueType, item_type: ValueType | None, value: Any, value_size: int):
        self.key = key
        self.vtype = vtype
        self.item_type = item_type
        self.value = value
        self.value_size = value_size  # on-disk size of the value bytes as currently parsed

    def type_label(self) -> str:
        if self.vtype == ValueType.ARRAY:
            return f'ARRAY<{self.item_type.name}>'
        return self.vtype.name

    def display_value(self) -> str:
        if self.vtype == ValueType.ARRAY:
            preview = self.value[:8]
            tail = ', ...' if len(self.value) > 8 else ''
            return f'[{", ".join(str(v) for v in preview)}{tail}] ({len(self.value)} items)'
        return str(self.value)


class TensorInfo:
    __slots__ = ('name', 'dims', 'ggml_type', 'rel_offset')

    def __init__(self, name: str, dims: list[int], ggml_type: int, rel_offset: int):
        self.name = name
        self.dims = dims
        self.ggml_type = ggml_type
        self.rel_offset = rel_offset

    def to_bytes(self) -> bytes:
        out = _pack_string(self.name)
        out += struct.pack('<I', len(self.dims))
        for d in self.dims:
            out += struct.pack('<Q', d)
        out += struct.pack('<I', self.ggml_type)
        out += struct.pack('<Q', self.rel_offset)
        return out


class GGUFModel:
    """Parses a GGUF file's header/metadata and supports in-place patching."""

    def __init__(self, path: str):
        self.path = path
        self.version: int
        self.alignment = DEFAULT_ALIGNMENT
        self.fields: dict[str, KVField] = {}
        self.order: list[str] = []
        self.tensors: list[TensorInfo] = []
        self.data_offset = 0
        self._removed: set[str] = set()
        self._edited: set[str] = set()
        self._dirty = False
        self._parse()

    def _parse(self) -> None:
        with open(self.path, 'rb') as f:
            (magic,) = struct.unpack('<I', _read_exact(f, 4))
            if magic != GGUF_MAGIC:
                raise GGUFFormatError('Not a GGUF file (bad magic)')
            (version,) = struct.unpack('<I', _read_exact(f, 4))
            if version not in (2, 3):
                raise GGUFFormatError(f'Unsupported GGUF version {version}')
            self.version = version
            tensor_count, kv_count = struct.unpack('<QQ', _read_exact(f, 16))

            for _ in range(kv_count):
                key = _read_string(f)
                (raw_vtype,) = struct.unpack('<I', _read_exact(f, 4))
                vtype = ValueType(raw_vtype)
                raw, item_type = _read_value_bytes(f, vtype)
                value = _decode_value(vtype, item_type, raw)
                self.fields[key] = KVField(key, vtype, item_type, value, len(raw))
                self.order.append(key)

            align_field = self.fields.get('general.alignment')
            if align_field is not None:
                self.alignment = int(align_field.value)
                if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
                    raise GGUFFormatError('general.alignment must be a non-zero power of two')

            for _ in range(tensor_count):
                name = _read_string(f)
                (n_dims,) = struct.unpack('<I', _read_exact(f, 4))
                dims = list(struct.unpack('<' + 'Q' * n_dims, _read_exact(f, 8 * n_dims)))
                (ggml_type,) = struct.unpack('<I', _read_exact(f, 4))
                (rel_offset,) = struct.unpack('<Q', _read_exact(f, 8))
                self.tensors.append(TensorInfo(name, dims, ggml_type, rel_offset))

            unaligned = f.tell()
            padding = (-unaligned) % self.alignment
            self.data_offset = unaligned + padding

    # -- read-only accessors -------------------------------------------------

    def keys(self) -> list[str]:
        return [k for k in self.order if k not in self._removed]

    def get(self, key: str) -> KVField | None:
        if key in self._removed:
            return None
        return self.fields.get(key)

    @property
    def needs_rebuild(self) -> bool:
        return self._dirty

    # -- edits ---------------------------------------------------------------

    def set_scalar(self, key: str, value: Any) -> None:
        """Patch a fixed-size scalar value directly in the file. No rebuild needed."""
        field = self.get(key)
        if field is None:
            raise KeyError(key)
        if field.vtype in (ValueType.STRING, ValueType.ARRAY):
            raise ValueError('set_scalar only supports fixed-size scalar fields')
        new_bytes = _pack_scalar(field.vtype, value)
        assert len(new_bytes) == field.value_size

        offset = self._locate_value_offset(key)
        with open(self.path, 'r+b') as f:
            f.seek(offset)
            f.write(new_bytes)
        field.value = value

    def stage_edit(self, key: str, vtype: ValueType, value: Any, item_type: ValueType | None = None) -> None:
        """Add or replace a key. Applied on commit() since its byte size may differ."""
        size = len(_pack_value(vtype, item_type, value))
        self.fields[key] = KVField(key, vtype, item_type, value, size)
        if key not in self.order:
            self.order.append(key)
        self._removed.discard(key)
        self._edited.add(key)
        self._dirty = True

    def stage_remove(self, key: str) -> None:
        if key not in self.fields:
            raise KeyError(key)
        self._removed.add(key)
        self._dirty = True

    def _locate_value_offset(self, target_key: str) -> int:
        """Re-walk the KV section to find the exact file offset of target_key's value bytes."""
        with open(self.path, 'rb') as f:
            f.seek(8)  # past magic + version
            (_tensor_count, kv_count) = struct.unpack('<QQ', _read_exact(f, 16))
            for _ in range(kv_count):
                key = _read_string(f)
                (raw_vtype,) = struct.unpack('<I', _read_exact(f, 4))
                vtype = ValueType(raw_vtype)
                value_offset = f.tell()
                _raw, _item_type = _read_value_bytes(f, vtype)
                if key == target_key:
                    return value_offset
        raise KeyError(target_key)

    # -- rebuild / commit -----------------------------------------------------

    def _build_header(self) -> bytes:
        kv_bytes = b''
        kv_count = 0
        for key in self.order:
            if key in self._removed:
                continue
            field = self.fields[key]
            kv_bytes += _pack_string(key)
            kv_bytes += struct.pack('<I', int(field.vtype))
            kv_bytes += _pack_value(field.vtype, field.item_type, field.value)
            kv_count += 1

        ti_bytes = b''.join(t.to_bytes() for t in self.tensors)

        header = struct.pack('<II', GGUF_MAGIC, self.version)
        header += struct.pack('<QQ', len(self.tensors), kv_count)
        header += kv_bytes
        header += ti_bytes
        return header

    def commit(self) -> None:
        """Write all staged add/remove/resize edits. Shifts tensor data only if the header size changed."""
        if not self._dirty:
            return

        header = self._build_header()
        new_data_offset = len(header) + ((-len(header)) % self.alignment)
        header += b'\x00' * (new_data_offset - len(header))

        old_data_offset = self.data_offset
        old_filesize = os.path.getsize(self.path)
        tensor_data_len = old_filesize - old_data_offset
        delta = new_data_offset - old_data_offset

        fd = os.open(self.path, os.O_RDWR)
        try:
            if delta == 0:
                os.pwrite(fd, header, 0)
            elif delta > 0:
                os.ftruncate(fd, new_data_offset + tensor_data_len)
                pos = tensor_data_len
                while pos > 0:
                    chunk = min(CHUNK_SIZE, pos)
                    pos -= chunk
                    buf = os.pread(fd, chunk, old_data_offset + pos)
                    os.pwrite(fd, buf, new_data_offset + pos)
                os.pwrite(fd, header, 0)
            else:
                pos = 0
                while pos < tensor_data_len:
                    chunk = min(CHUNK_SIZE, tensor_data_len - pos)
                    buf = os.pread(fd, chunk, old_data_offset + pos)
                    os.pwrite(fd, buf, new_data_offset + pos)
                    pos += chunk
                os.pwrite(fd, header, 0)
                os.ftruncate(fd, new_data_offset + tensor_data_len)
        finally:
            os.close(fd)

        self.data_offset = new_data_offset
        self._removed.clear()
        self._edited.clear()
        self._dirty = False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

TYPE_BY_NAME = {t.name: t for t in ValueType}
EDITABLE_SCALAR_TYPES = [t for t in ValueType if t not in (ValueType.STRING, ValueType.ARRAY)]


def _parse_cli_scalar(vtype: ValueType, raw: str) -> Any:
    if vtype == ValueType.STRING:
        return raw
    if vtype == ValueType.BOOL:
        return raw.strip().lower() in ('1', 'true', 'yes', 'y')
    if vtype in (ValueType.FLOAT32, ValueType.FLOAT64):
        return float(raw)
    return int(raw, 0)


def run_cli(model: GGUFModel, args: argparse.Namespace) -> None:
    if args.list:
        for key in model.keys():
            field = model.get(key)
            print(f'{key}\t{field.type_label()}\t{field.display_value()}')
        return

    for key in args.remove or []:
        model.stage_remove(key)

    for key, type_name, raw_value in args.set or []:
        vtype = TYPE_BY_NAME.get(type_name.upper())
        if vtype is None or vtype == ValueType.ARRAY:
            logger.error(f'Unsupported --set type {type_name!r} (arrays are not supported by --set)')
            sys.exit(1)
        value = _parse_cli_scalar(vtype, raw_value)
        existing = model.get(key)
        if existing is not None and existing.vtype == vtype and vtype != ValueType.STRING:
            model.set_scalar(key, value)
        else:
            model.stage_edit(key, vtype, value)

    if model.needs_rebuild:
        model.commit()
    logger.info('Changes applied in place.')


# --------------------------------------------------------------------------
# curses TUI
# --------------------------------------------------------------------------

def _prompt(stdscr, y: int, label: str) -> str | None:
    curses.echo()
    curses.curs_set(1)
    stdscr.move(y, 0)
    stdscr.clrtoeol()
    stdscr.addstr(y, 0, label)
    stdscr.refresh()
    try:
        raw = stdscr.getstr(y, len(label)).decode('utf-8')
    except (KeyboardInterrupt, curses.error):
        raw = None
    curses.noecho()
    curses.curs_set(0)
    return raw


def _tui_main(stdscr, model: GGUFModel) -> None:
    curses.curs_set(0)
    top = 0
    sel = 0
    status = 'j/k move  e edit  a add  d delete  w write & quit  q quit without saving'

    while True:
        keys = model.keys()
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f'GGUF Editor -- {model.path}'[:w - 1], curses.A_BOLD)
        visible = h - 3
        top = min(top, sel)
        if sel >= top + visible:
            top = sel - visible + 1

        for row, key in enumerate(keys[top:top + visible]):
            idx = top + row
            field = model.get(key)
            line = f'{key:<50.50} {field.type_label():<14.14} {field.display_value()}'[:w - 1]
            attr = curses.A_REVERSE if idx == sel else curses.A_NORMAL
            stdscr.addstr(2 + row, 0, line, attr)

        stdscr.addstr(h - 1, 0, status[:w - 1], curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch == ord('q'):
            if model.needs_rebuild:
                r = _prompt(stdscr, h - 1, 'Discard unsaved changes? [y/N] ')
                if r and r.strip().lower() == 'y':
                    return
                continue
            return
        elif ch == ord('w'):
            model.commit()
            return
        elif ch in (curses.KEY_UP, ord('k')):
            sel = max(0, sel - 1)
        elif ch in (curses.KEY_DOWN, ord('j')):
            sel = min(len(keys) - 1, sel + 1)
        elif ch == ord('d') and keys:
            model.stage_remove(keys[sel])
            sel = min(sel, len(model.keys()) - 1)
        elif ch == ord('e') and keys:
            field = model.get(keys[sel])
            if field.vtype == ValueType.ARRAY:
                _prompt(stdscr, h - 1, 'Array editing not supported here; use --set on the CLI. Press enter.')
                continue
            raw = _prompt(stdscr, h - 1, f'New value for {keys[sel]} ({field.vtype.name}): ')
            if raw is None:
                continue
            try:
                value = _parse_cli_scalar(field.vtype, raw)
                if field.vtype == ValueType.STRING:
                    model.stage_edit(keys[sel], field.vtype, value)
                else:
                    model.set_scalar(keys[sel], value)
            except ValueError as e:
                _prompt(stdscr, h - 1, f'Invalid value: {e}. Press enter.')
        elif ch == ord('a'):
            key = _prompt(stdscr, h - 1, 'New key name: ')
            if not key:
                continue
            type_name = _prompt(stdscr, h - 1, 'Type (' + '/'.join(t.name for t in EDITABLE_SCALAR_TYPES) + '): ')
            vtype = TYPE_BY_NAME.get((type_name or '').strip().upper())
            if vtype is None or vtype == ValueType.ARRAY:
                _prompt(stdscr, h - 1, 'Invalid type. Press enter.')
                continue
            raw = _prompt(stdscr, h - 1, 'Value: ')
            if raw is None:
                continue
            try:
                model.stage_edit(key, vtype, _parse_cli_scalar(vtype, raw))
            except ValueError as e:
                _prompt(stdscr, h - 1, f'Invalid value: {e}. Press enter.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Standalone in-place GGUF metadata editor (TUI by default)')
    parser.add_argument('model', type=str, help='GGUF file to edit (edited in place)')
    parser.add_argument('--list', action='store_true', help='List all metadata keys and exit')
    parser.add_argument(
        '--set', nargs=3, action='append', metavar=('KEY', 'TYPE', 'VALUE'),
        help='Set/add a scalar or string key, e.g. --set general.name STRING "My Model"',
    )
    parser.add_argument('--remove', action='append', metavar='KEY', help='Remove a key')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    model = GGUFModel(args.model)
    if args.list or args.set or args.remove:
        run_cli(model, args)
    else:
        curses.wrapper(_tui_main, model)


if __name__ == '__main__':
    main()
