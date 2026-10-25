#!/usr/bin/env python3
"""Standalone, dependency-free GGUF metadata editor.

Optimized to handle huge vocabularies and large files with progress indicators.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from enum import IntEnum
from typing import Any, BinaryIO

logger = None # Simple logging via print/sys.stderr

GGUF_MAGIC = 0x46554747
DEFAULT_ALIGNMENT = 32
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB buffer for shifting weights

class ValueType(IntEnum):
    UINT8 = 0; INT8 = 1; UINT16 = 2; INT16 = 3; UINT32 = 4; INT32 = 5
    FLOAT32 = 6; BOOL = 7; STRING = 8; ARRAY = 9; UINT64 = 10; INT64 = 11; FLOAT64 = 12

SCALAR_STRUCT_FMT = {
    ValueType.UINT8: 'B', ValueType.INT8: 'b', ValueType.UINT16: 'H', ValueType.INT16: 'h',
    ValueType.UINT32: 'I', ValueType.INT32: 'i', ValueType.UINT64: 'Q', ValueType.INT64: 'q',
    ValueType.FLOAT32: 'f', ValueType.FLOAT64: 'd', ValueType.BOOL: 'B',
}
SCALAR_SIZE = {vt: struct.calcsize('<' + fmt) for vt, fmt in SCALAR_STRUCT_FMT.items()}

class GGUFFormatError(Exception): pass

def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) != n: raise GGUFFormatError('Unexpected end of file')
    return buf

def _read_string_as_bytes(f: BinaryIO) -> bytes:
    (length,) = struct.unpack('<Q', _read_exact(f, 8))
    return struct.pack('<Q', length) + _read_exact(f, length)

def _pack_string(s: str) -> bytes:
    raw = s.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw

def _read_value_bytes(f: BinaryIO, vtype: ValueType) -> tuple[bytes, ValueType | None]:
    if vtype == ValueType.STRING: return _read_string_as_bytes(f), None
    if vtype in SCALAR_STRUCT_FMT: return _read_exact(f, SCALAR_SIZE[vtype]), None
    if vtype == ValueType.ARRAY:
        (item_type_raw,) = struct.unpack('<I', _read_exact(f, 4))
        item_type = ValueType(item_type_raw)
        (count,) = struct.unpack('<Q', _read_exact(f, 8))
        parts = [struct.pack('<IQ', item_type_raw, count)]
        for _ in range(count):
            if item_type == ValueType.STRING: parts.append(_read_string_as_bytes(f))
            else: parts.append(_read_exact(f, SCALAR_SIZE[item_type]))
        return b''.join(parts), item_type
    raise GGUFFormatError(f'Unknown value type {vtype}')

def _draw_progress(current: int, total: int, prefix: str = ""):
    """Standard dependency-free progress bar."""
    bar_len = 30
    if total <= 0: return
    filled_len = int(bar_len * current // total)
    percents = round(100.0 * current / total, 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stderr.write(f'\r{prefix} [{bar}] {percents}% ({current / (1024**3):.2f} / {total / (1024**3):.2f} GB)')
    sys.stderr.flush()

# --------------------------------------------------------------------------

class KVField:
    __slots__ = ('key', 'vtype', 'item_type', 'raw_value')
    def __init__(self, key: str, vtype: ValueType, item_type: ValueType | None, raw_value: bytes):
        self.key, self.vtype, self.item_type, self.raw_value = key, vtype, item_type, raw_value
    def type_label(self) -> str:
        return f'ARRAY<{self.item_type.name}>' if self.vtype == ValueType.ARRAY else self.vtype.name
    def display_value(self) -> str:
        if self.vtype == ValueType.ARRAY: return f"({len(self.raw_value)} bytes raw array)"
        if self.vtype == ValueType.STRING:
            (length,) = struct.unpack_from('<Q', self.raw_value, 0)
            return self.raw_value[8:8 + length].decode('utf-8', errors='replace')
        return "scalar value"

class TensorInfo:
    __slots__ = ('raw_bytes')
    def __init__(self, raw_bytes: bytes): self.raw_bytes = raw_bytes

class GGUFModel:
    def __init__(self, path: str):
        self.path = path
        self.fields: dict[str, KVField] = {}
        self.order: list[str] = []
        self.tensors: list[TensorInfo] = []
        self._dirty = False
        self._parse()

    def _parse(self) -> None:
        with open(self.path, 'rb') as f:
            (magic,) = struct.unpack('<I', _read_exact(f, 4))
            if magic != GGUF_MAGIC: raise GGUFFormatError('Not a GGUF file')
            (self.version,) = struct.unpack('<I', _read_exact(f, 4))
            tensor_count, kv_count = struct.unpack('<QQ', _read_exact(f, 16))

            for i in range(kv_count):
                if i % 20 == 0: sys.stderr.write(f"\rParsing metadata: {i}/{kv_count}...")
                key_raw = _read_string_as_bytes(f)
                key = key_raw[8:].decode('utf-8')
                (raw_vtype,) = struct.unpack('<I', _read_exact(f, 4))
                vtype = ValueType(raw_vtype)
                val_bytes, item_type = _read_value_bytes(f, vtype)
                self.fields[key] = KVField(key, vtype, item_type, val_bytes)
                self.order.append(key)
            
            sys.stderr.write(f"\rParsing metadata: {kv_count}/{kv_count} Done.\n")

            for i in range(tensor_count):
                if i % 100 == 0: sys.stderr.write(f"\rParsing tensors: {i}/{tensor_count}...")
                start = f.tell()
                _ = _read_string_as_bytes(f)
                (n_dims,) = struct.unpack('<I', _read_exact(f, 4))
                f.seek(8 * n_dims + 4 + 8, os.SEEK_CUR)
                end = f.tell()
                f.seek(start)
                self.tensors.append(TensorInfo(_read_exact(f, end - start)))

            sys.stderr.write(f"\rParsing tensors: {tensor_count}/{tensor_count} Done.\n")
            pos = f.tell()
            self.alignment = struct.unpack('<I', self.fields['general.alignment'].raw_value)[0] if 'general.alignment' in self.fields else 32
            self.data_offset = pos + ((-pos) % self.alignment)

    def stage_edit(self, key: str, vtype: ValueType, value: Any):
        raw = _pack_string(value) if vtype == ValueType.STRING else struct.pack('<' + SCALAR_STRUCT_FMT[vtype], value)
        self.fields[key] = KVField(key, vtype, None, raw)
        if key not in self.order: self.order.append(key)
        self._dirty = True

    def commit(self) -> None:
        if not self._dirty: return
        
        # Re-build header
        parts = [struct.pack('<IIQQ', GGUF_MAGIC, self.version, len(self.tensors), len([k for k in self.order if k in self.fields]))]
        for key in self.order:
            if key not in self.fields: continue
            f = self.fields[key]
            parts.extend([_pack_string(key), struct.pack('<I', int(f.vtype)), f.raw_value])
        for t in self.tensors: parts.append(t.raw_bytes)
            
        header = b''.join(parts)
        new_off = len(header) + ((-len(header)) % self.alignment)
        header += b'\x00' * (new_off - len(header))

        old_off = self.data_offset
        delta = new_off - old_off
        fd = os.open(self.path, os.O_RDWR)
        try:
            tensor_len = os.path.getsize(self.path) - old_off
            if delta != 0:
                moved = 0
                if delta > 0: # Header grew, move weights backwards
                    os.ftruncate(fd, new_off + tensor_len)
                    for pos in range(tensor_len - CHUNK_SIZE, -CHUNK_SIZE, -CHUNK_SIZE):
                        read_at = max(0, pos)
                        sz = CHUNK_SIZE if pos >= 0 else (CHUNK_SIZE + pos)
                        os.pwrite(fd, os.pread(fd, sz, old_off + read_at), new_off + read_at)
                        moved += sz
                        _draw_progress(moved, tensor_len, "Shifting weights (backwards)")
                else: # Header shrank, move weights forwards
                    for pos in range(0, tensor_len, CHUNK_SIZE):
                        sz = min(CHUNK_SIZE, tensor_len - pos)
                        os.pwrite(fd, os.pread(fd, sz, old_off + pos), new_off + pos)
                        moved += sz
                        _draw_progress(moved, tensor_len, "Shifting weights (forwards)")
                    os.ftruncate(fd, new_off + tensor_len)
                sys.stderr.write("\n")
            os.pwrite(fd, header, 0)
        finally:
            os.close(fd)
        print(f"File updated successfully. New header size: {new_off} bytes.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--chat-template', '-t')
    parser.add_argument('--set', nargs=3, action='append', metavar=('KEY', 'TYPE', 'VALUE'))
    args = parser.parse_args()

    model = GGUFModel(args.model)
    if args.list:
        for k in model.order:
            f = model.fields[k]
            print(f"{k}\t{f.type_label()}\t{f.display_value()}")
    elif args.chat_template:
        content = args.chat_template
        if os.path.isfile(content):
            with open(content, 'r', encoding='utf-8') as f: content = f.read().strip()
        model.stage_edit("tokenizer.chat_template", ValueType.STRING, content)
        model.commit()
    elif args.set:
        for k, t, v in args.set:
            vtype = ValueType[t.upper()]
            val = float(v) if vtype in (ValueType.FLOAT32, ValueType.FLOAT64) else (int(v, 0) if vtype != ValueType.STRING else v)
            model.stage_edit(k, vtype, val)
        model.commit()

if __name__ == '__main__':
    main()
