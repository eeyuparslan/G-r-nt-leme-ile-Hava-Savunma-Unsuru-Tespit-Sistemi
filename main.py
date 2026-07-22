import cv2
import os
from ultralytics import YOLO
import csv
from datetime import datetime


def main():
    # 1. EN BAŞARILI MODEL YOLU (v6-7)
    model_path = r"runs\detect\hava_savunma_unsuru_v6-7\weights\best.pt"
    model = YOLO(model_path)
    sinif_isimleri = model.names

    veri_klasoru = r"C:\Users\metro\Desktop\Hava_Savunma_Tespit\test_verileri"
    csv_dosyasi = "toplu_tespit_raporu.csv"

    # CSV Başlıklarını yaz
    with open(csv_dosyasi, mode='w', newline='', encoding='utf-8') as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow(['Dosya_Adi', 'Tarih_Saat', 'Kare_No', 'Sinif', 'Hedef_ID', 'Guven(%)', 'X', 'Y', 'Durum'])

    dosyalar = [f for f in os.listdir(veri_klasoru) if
                f.lower().endswith(('.jpg', '.png', '.mp4', '.avi', '.mov', '.mkv'))]

    print(f"Toplam {len(dosyalar)} dosya işleniyor...\n")

    for dosya_adi in dosyalar:
        dosya_yolu = os.path.join(veri_klasoru, dosya_adi)
        is_video = dosya_adi.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

        print(f"İŞLENİYOR: {dosya_adi}")

        if not is_video:
            # FOTOĞRAF İŞLEME
            results = model.predict(source=dosya_yolu, conf=0.5, device=0, verbose=False)
            annotated_frame = draw_status(results[0], results[0].plot())

            self_save_results(results[0], dosya_adi, 0, csv_dosyasi, sinif_isimleri)
            cv2.imshow("Otonom Tarama", annotated_frame)
            cv2.waitKey(0)
        else:
            # VİDEO İŞLEME
            cap = cv2.VideoCapture(dosya_yolu)
            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                results = model.track(frame, persist=True, conf=0.5, tracker="bytetrack.yaml", device=0, verbose=False)
                annotated_frame = draw_status(results[0], results[0].plot())

                self_save_results(results[0], dosya_adi, int(cap.get(cv2.CAP_PROP_POS_FRAMES)), csv_dosyasi,
                                  sinif_isimleri)

                cv2.imshow("Otonom Tarama", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()

    cv2.destroyAllWindows()
    print("\nTÜM GÖREVLER TAMAMLANDI.")


def draw_status(result, frame):
    """Görsel üzerine Durum bilgisini işler."""
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        mesaj = "Hava Savunma Unsuru Tespit Edildi"
        renk = (0, 255, 0)  # Yeşil
    else:
        mesaj = "Hava Savunma Unsuru Yok"
        renk = (0, 0, 255)  # Kırmızı

    cv2.putText(frame, mesaj, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
    return frame


def self_save_results(result, dosya_adi, kare_no, csv_yolu, sinif_isimleri):
    boxes = result.boxes
    durum = "Tespit Edildi" if (boxes is not None and len(boxes) > 0) else "Yok"

    with open(csv_yolu, mode='a', newline='', encoding='utf-8') as f:
        yazici = csv.writer(f)
        if durum == "Tespit Edildi":
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = int(box.conf.item() * 100)
                xywh = box.xywh.cpu().tolist()[0]
                track_id = int(box.id.item()) if box.id is not None else 0
                yazici.writerow([dosya_adi, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kare_no,
                                 sinif_isimleri[cls_id], track_id, conf, int(xywh[0]), int(xywh[1]), durum])
        else:
            # Hedefin olmadığı anları da raporluyoruz
            yazici.writerow(
                [dosya_adi, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kare_no, "N/A", "N/A", 0, 0, 0, durum])


if __name__ == '__main__':
    main()