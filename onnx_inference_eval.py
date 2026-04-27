from pathlib import Path
import argparse
import time

import cv2
import numpy as np


DEFAULT_ONNX_PATH = Path("best.onnx")
DEFAULT_IMAGE_PATH = Path("test.jpg")
DEFAULT_PT_PATH = Path("trash_yolo_project/waste-bin-detector-v8s-9608/weights/best.pt")
IMG_SIZE = 640


def format_size(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def letterbox(image: np.ndarray, new_shape: tuple[int, int] = (IMG_SIZE, IMG_SIZE)) -> np.ndarray:
    """Resize image to fit new_shape while preserving aspect ratio, then pad."""
    original_h, original_w = image.shape[:2]
    target_h, target_w = new_shape

    scale = min(target_w / original_w, target_h / original_h)
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = letterbox(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, dtype=np.float32)


def run_onnx(session, input_name: str, input_tensor: np.ndarray) -> list[np.ndarray]:
    return session.run(None, {input_name: input_tensor})


def compare_with_pytorch(
    pt_path: Path,
    onnx_outputs: list[np.ndarray],
    input_tensor: np.ndarray,
) -> None:
    import torch
    from ultralytics import YOLO

    if not pt_path.exists():
        print(f"\nSkipping PyTorch comparison. Missing weights: {pt_path}")
        return

    model = YOLO(str(pt_path))
    model.model.eval()

    with torch.no_grad():
        torch_input = torch.from_numpy(input_tensor)
        torch_outputs = model.model(torch_input)

    if isinstance(torch_outputs, (list, tuple)):
        torch_output = torch_outputs[0]
    else:
        torch_output = torch_outputs

    torch_output = torch_output.detach().cpu().numpy()
    onnx_output = onnx_outputs[0]

    print("\nPyTorch vs ONNX sanity check")
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")

    if torch_output.shape != onnx_output.shape:
        print("Shapes differ, so numeric allclose comparison was skipped.")
        return

    is_close = np.allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-3)
    max_abs_diff = np.max(np.abs(torch_output - onnx_output))
    print(f"Outputs close: {is_close}")
    print(f"Max absolute difference: {max_abs_diff:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 ONNX inference and latency evaluation.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH, help="Path to best.onnx")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Path to test image")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed inference runs")
    parser.add_argument(
        "--compare-pytorch",
        action="store_true",
        help="Also compare raw ONNX output against the PyTorch model output",
    )
    parser.add_argument("--pt", type=Path, default=DEFAULT_PT_PATH, help="Path to PyTorch weights")
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"Could not find ONNX model: {args.onnx}")
    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer")

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "onnxruntime is required for this script. Install it with: "
            "python -m pip install -r requirements_onnx.txt"
        ) from error

    input_tensor = preprocess_image(args.image)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    input_info = session.get_inputs()[0]
    input_name = input_info.name

    print(f"ONNX model: {args.onnx} ({format_size(args.onnx)})")
    if args.pt.exists():
        print(f"PyTorch model: {args.pt} ({format_size(args.pt)})")
    print(f"Image: {args.image}")
    print(f"Input name: {input_name}")
    print(f"Model input shape: {input_info.shape}")
    print(f"Runtime input shape: {input_tensor.shape}")

    run_onnx(session, input_name, input_tensor)

    start = time.perf_counter()
    outputs = run_onnx(session, input_name, input_tensor)
    single_latency_ms = (time.perf_counter() - start) * 1000

    latencies_ms = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = run_onnx(session, input_name, input_tensor)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    print("\nInference results")
    print(f"Single-run latency: {single_latency_ms:.2f} ms")
    print(f"Average latency over {args.runs} runs: {np.mean(latencies_ms):.2f} ms")
    print(f"Min latency: {np.min(latencies_ms):.2f} ms")
    print(f"Max latency: {np.max(latencies_ms):.2f} ms")

    print("\nOutput shapes")
    for index, output in enumerate(outputs):
        print(f"output[{index}]: {output.shape}")

    if args.compare_pytorch:
        compare_with_pytorch(args.pt, outputs, input_tensor)


if __name__ == "__main__":
    main()
