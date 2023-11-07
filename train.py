from ultralytics.models import YOLO


def main():
    model = YOLO('yolov8s.yaml').load('weights/yolov8s.pt')
    model.train(data='data/ACDC-fog.yaml', epochs=50, imgsz=640)
    print()


if __name__ == '__main__':
    main()

