import argparse
import sys

import numpy as np
from gguf.constants import GGUFValueType
from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter


def infer_type(val_str):
    if val_str.lower() == "true":
        return True, GGUFValueType.BOOL
    if val_str.lower() == "false":
        return False, GGUFValueType.BOOL
    try:
        if "." in val_str:
            return float(val_str), GGUFValueType.FLOAT32
        return int(val_str), GGUFValueType.INT32
    except ValueError:
        return val_str, GGUFValueType.STRING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--set", nargs="*", help="key=value")
    parser.add_argument("--template", help="file path for tokenizer.chat_template")
    parser.add_argument("--dump-template", help="dump tokenizer.chat_template to this file path")
    args = parser.parse_args()

    reader = GGUFReader(args.input)

    if args.dump_template:
        field = reader.get_field("tokenizer.chat_template")
        if not field:
            print("Error: tokenizer.chat_template not found in model.")
            sys.exit(1)
        with open(args.dump_template, "w", encoding="utf-8") as f:
            f.write(field.contents())
        print(f"Chat template written to {args.dump_template}")

    if args.dump:
        print("--- Metadata ---")
        for key, field in reader.fields.items():
            val = field.contents()
            if isinstance(val, (list, np.ndarray)) and len(val) > 5:
                print(f"{key}: {val[:5]} ... (total {len(val)})")
            else:
                print(f"{key}: {val}")

        print("\n--- Tensors ---")
        for tensor in reader.tensors:
            # Using tensor.shape for logical shape, tensor.data.shape for byte shape
            print(f"{tensor.name}: {tensor.shape} (bytes: {tensor.data.shape})")

    if not args.set and not args.template:
        sys.exit(0)

    if not args.output:
        print("Error: Modification requires an output file path.")
        sys.exit(1)

    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field else "unknown"

    writer = GGUFWriter(args.output, arch=arch)

    updates = {}
    if args.set:
        for item in args.set:
            k, v = item.split("=", 1)
            updates[k] = infer_type(v)

    if args.template:
        with open(args.template, "r", encoding="utf-8") as f:
            updates["tokenizer.chat_template"] = (f.read(), GGUFValueType.STRING)

    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue

        if key in updates:
            val, vtype = updates[key]
            writer.add_key_value(key, val, vtype)
        else:
            val = field.contents()
            vtype = field.types[0] if field.types else GGUFValueType.STRING
            sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
            writer.add_key_value(key, val, vtype, sub_type)

    for k, v in updates.items():
        if k not in reader.fields:
            val, vtype = v
            writer.add_key_value(k, val, vtype)

    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
