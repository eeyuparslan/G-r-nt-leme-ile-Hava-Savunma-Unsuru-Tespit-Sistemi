from ultralytics import YOLO

def main():
    # 1. v6-6 ağırlıklarını yükle
    model = YOLO(r"C:\Users\metro\Desktop\PythonProject\runs\detect\hava_savunma_unsuru_v6-7\weights\best.pt")

    # 2. Yeni eğitim başlat
    results = model.train(
        data=r"C:\Users\metro\Desktop\PythonProject\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name='hava_savunma_unsuru_v6-8', # İsimlendirmeyi v6-7 olarak güncelledim (v6-6 sonrası olduğu için)
        device=0,
        cache=False
    )

if __name__ == '__main__':
    main()