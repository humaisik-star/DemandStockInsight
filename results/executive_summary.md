# Yönetici Özeti (otomatik üretildi)

1. Genel Durum
- Portföy: 100 SKU. En kritik KPI'lar: envanter azaltma %28.4, ort. servis seviyesi %94.7, ort. devir hızı 96.2, yeniden sipariş uyarısı sayısı 95.  
- Durum: Servis seviyesi makul (%94.7) ancak 95 uyarı ve A sınıfın gelir payı yüksek (A: 59 SKU = %79.8) nedeniyle likiditeyi önce en yüksek değerli SKU'lara yönlendirmeliyiz.

2. Öncelikli Kararlar
- Hemen sipariş/verişonlandır: Top 5 uyarılı A-SKU için derhal tedarik başlatılacak (P0011, P0008, P0009 @S004, P0009 @S005, P0015). Gerekçe: hepsi A sınıf; stok açıkları büyük ve gelir riski yüksek.  
- Öncelik sırası (değer bazlı): öncelik S005:P0009 (stok açığı en yüksek), sonra S003:P0011 ve S003:P0008. Gerekçe: stok açığı büyüklüğü satış kaybı riskiyle doğrudan ilişkilidir.  
- Kısa vadeli forecast ayarı: S001:P0019 ve S002:P0006 için tahmini artır (7 gün talep sıçramasına göre) ve acil tedarik tetikle. Gerekçe: yüksek şiddetli talep sıçramaları stokout riskini yükseltiyor.  
- Kaynak tahsisi: Reorder uyarılarının önceliğini A sınıf (59 SKU / %79.8 gelir) bazlı sınırlı bütçe ile karşıla. Gerekçe: sınırlı stok bütçesinin en yüksek gelir katkısına yönlendirilmesi gerekli.  
- Soruşturma aksiyonu: 95 uyarının nedenini 48 saat içinde analiz et; özellikle sistemsel hatalar veya tekil promotif aksiyonlar varsa düzelt. Gerekçe: uyarı sayısı operasyonel yük yaratıyor ve yanlış alarm maliyeti var.

3. Ürün Bazlı Yorum (uyarılı, en değerli ürünler)
- S003 P0011 (A): mevcut 143 vs reorder 775.4 — açık ≈ 632 adet. Öneri: hızlandırılmış sipariş, tedarik zamanı uzun ise acil parti.  
- S003 P0008 (A): mevcut 170 vs reorder 788.0 — açık ≈ 618 adet. Öneri: aynı tedarikçi ile konsolide sipariş.  
- S004 P0009 (A): mevcut 472 vs reorder 781.4 — açık ≈ 309 adet. Öneri: normal öncelikte sipariş; mümkünse S004 içinden S005'e kısmi transfer değerlendirilerek toplam maliyet düşürülsün.  
- S005 P0009 (A): mevcut 136 vs reorder 882.1 — açık ≈ 746 adet. Öneri: en yüksek öncelik, expedite veya acil PO.  
- S003 P0015 (A): mevcut 413 vs reorder 857.2 — açık ≈ 444 adet. Öneri: sipariş planına öncelik ver.

4. Anomali Açıklaması
- S001 P0019 — Talep sıçraması %45 (yüksek). Olası nedenler: lokal promo/ürün lansmanı, stok yönlendirmesi veya POS hata. Aksiyon: 24-48 saat içindeki satış ve promosyon kayıtlarını kontrol et, kısa dönem stok artır, forecast +%45 ile güncelle.  
- S002 P0006 — Talep sıçraması %40 (yüksek). Olası nedenler: aynı sınıf sebepler. Aksiyon: aynı adımlar; eğer talep kalıcıysa güvenlik stoğunu yeniden hesapla.

5. ABC-XYZ Değerlendirmesi
- AX (59 SKU): Ana odak. Mevcut 59 AX ürün, toplam gelirde %79.8 pay ediyor — hedef: maksimum servis, öncelikli bütçe ve sık frekanslı replenishment.  
- BX (18) ve BY (6): Orta öncelik. Sipariş frekansını optimize et; stok maliyetini düşürürken servis seviyesi korunmalı.  
- CX (13) ve CY (4): Düşük öncelik. Envanter minimale yakın tutulmalı; dönemsel gözlem sonrası SKU rasyonelleştirme gündeme alınsın.
