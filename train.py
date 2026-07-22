from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\metro\Desktop\PythonProject\runs\detect\hava_savunma_unsuru_v6-7\weights\best.pt")

    results = model.train(
        data=r"C:\Users\metro\Desktop\PythonProject\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name='hava_savunma_unsuru_v6-8',
        device=0,
        cache=False
    )

if __name__ == '__main__':
    main()