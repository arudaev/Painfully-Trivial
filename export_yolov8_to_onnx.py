from pathlib import Path
import shutil

from ultralytics import YOLO


WEIGHTS_PATH = Path("trash_yolo_project/waste-bin-detector-v8s-9608/weights/best.pt")
ONNX_PATH = Path("best.onnx")
IMG_SIZE = 640


def format_size(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Could not find PyTorch weights: {WEIGHTS_PATH}")

    print(f"Loading YOLOv8 model: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    print("Exporting to ONNX...")
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            dynamic=True,
            simplify=True,
            opset=None,
            nms=False,
            device="cpu",
        )
    )

    if exported_path.resolve() != ONNX_PATH.resolve():
        shutil.copy2(exported_path, ONNX_PATH)

    print("\nExport complete")
    print(f"PyTorch model: {WEIGHTS_PATH} ({format_size(WEIGHTS_PATH)})")
    print(f"ONNX model:    {ONNX_PATH} ({format_size(ONNX_PATH)})")


if __name__ == "__main__":
    main()
