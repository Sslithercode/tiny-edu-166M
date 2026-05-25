"""
Convert parchment_instruct.onnx (fp32, ~957 MB) to fp16 (~478 MB).

FP16 is a lossless dtype cast — no accuracy loss like INT8 quantisation.
With keep_io_types=True the graph boundary stays fp32 (input_ids is int64
anyway, logits come back as fp32), so onnxruntime's CPU provider handles
the internal fp16 ↔ fp32 casts automatically.

Install deps first (one-time):
    pip install onnx onnxconverter-common

Then run:
    python quantize_fp16.py
"""

from pathlib import Path
import onnx
from onnx import TensorProto
from onnxconverter_common import convert_float_to_float16

BUNDLE = Path("onnx_bundle")
SRC    = BUNDLE / "parchment_instruct.onnx"
DST    = BUNDLE / "parchment_instruct_fp16.onnx"

print(f"Loading  {SRC}  ({SRC.stat().st_size / 1e6:.0f} MB) ...")
model = onnx.load(str(SRC))

print("Converting to fp16 (shape inference runs — takes ~30 s on this model) ...")
fp16_model = convert_float_to_float16(
    model,
    keep_io_types=True,       # logits output stays fp32; safe on CPU
    disable_shape_infer=False, # must be False so all fp32 constants get cast correctly
)

# ── Post-process: fix Cast nodes where 'to' attribute still says FLOAT (1)
# but the graph has re-typed their output as FLOAT16 (10).
# This happens with boolean-mask Cast nodes (causal mask in SDPA).
# We build a lookup of value_info types so we can check actual output types.
vi_type = {}
for vi in list(fp16_model.graph.value_info) + list(fp16_model.graph.output):
    vi_type[vi.name] = vi.type.tensor_type.elem_type

fixed = 0
for node in fp16_model.graph.node:
    if node.op_type != "Cast":
        continue
    for attr in node.attribute:
        if attr.name == "to" and attr.i == TensorProto.FLOAT:
            # Check if the actual output was re-typed to fp16
            out_name = node.output[0]
            if vi_type.get(out_name) == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT16
                fixed += 1

if fixed:
    print(f"  fixed {fixed} Cast node(s) with stale 'to=FLOAT' attribute")

onnx.save(fp16_model, str(DST))
size = DST.stat().st_size / 1e6
print(f"Saved    {DST}  ({size:.0f} MB)")
print()
print("To switch inference to fp16, update onnx_bundle/bundle_meta.json:")
print('  "model_file": "parchment_instruct_fp16.onnx"')
