import sys, onnx
if len(sys.argv) != 3:
    print("uso: python tools/rm_init_from_inputs.py <in.onnx> <out.onnx>")
    sys.exit(1)
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init_names = {t.name for t in g.initializer}
g.input[:] = [t for t in g.input if t.name not in init_names]
onnx.save(m, dst)
print(f"Saved -> {dst}")