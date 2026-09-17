import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

SRC = "models/model.onnx"
DST = "models/model_std.onnx"


def main() -> None:
    model = onnx.load(SRC)
    linear_nodes = [n for n in model.graph.node if n.op_type == "LinearRegressor"]
    if len(linear_nodes) != 1:
        raise RuntimeError(
            f"expected exactly one LinearRegressor node, found {len(linear_nodes)} -- "
            "this script only handles the simple single-node linear-model case."
        )
    node = linear_nodes[0]
    attrs = {a.name: a for a in node.attribute}
    coefficients = np.array(attrs["coefficients"].floats, dtype=np.float32)
    intercepts = np.array(attrs["intercepts"].floats, dtype=np.float32)
    n_targets = intercepts.shape[0]
    n_features = coefficients.shape[0] // n_targets
    W = coefficients.reshape(n_targets, n_features)

    gemm = helper.make_node(
        "Gemm",
        inputs=[node.input[0], "W", "B"],
        outputs=[node.output[0]],
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
        name="LinearRegressor_as_Gemm",
    )
    graph = helper.make_graph(
        [gemm],
        f"{model.graph.name}_std",
        model.graph.input,
        model.graph.output,
        initializer=[
            numpy_helper.from_array(W, name="W"),
            numpy_helper.from_array(intercepts, name="B"),
        ],
    )
    new_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(new_model)

    # Verify the rewrite is numerically faithful before trusting it downstream.
    x = np.random.randn(20, n_features).astype(np.float32)
    original = ort.InferenceSession(SRC).run(None, {node.input[0]: x})[0]
    onnx.save(new_model, DST)
    rewritten = ort.InferenceSession(DST).run(None, {node.input[0]: x})[0]
    max_diff = np.abs(original - rewritten).max()
    print(f"Rewrote {SRC} -> {DST} (LinearRegressor -> Gemm)")
    print(f"Max abs diff vs original on {x.shape[0]} random rows: {max_diff:.6f}")
    if max_diff > 1e-2:
        raise RuntimeError("rewrite diverged from the original model -- do not use it")
    print(
        "Rewrite verified. Safe to use models/model_std.onnx for OpenVINO conversion."
    )


if __name__ == "__main__":
    main()
