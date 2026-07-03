"""
Automatic executive summary — Azure OpenAI turns the analytics snapshot into a
decision-focused Turkish brief and saves it as results/executive_summary.md.

It gathers the same snapshot the chatbot's `yonetici_ozeti` tool returns (KPIs,
ABC / ABC-XYZ, top value alerts, anomalies) and asks the model for a management
report: headline decisions, product-level commentary, and anomaly explanations.

Run:
    .venv/bin/python executive_summary.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.assistant_tools import yonetici_ozeti

OUT_PATH = Path("results/executive_summary.md")

PROMPT = """Sen bir perakende talep-planlama direktörüsün. Aşağıdaki veri anlık
görüntüsüne dayanarak Türkçe, karar odaklı bir YÖNETİCİ ÖZETİ yaz.

Bölümler:
1. Genel Durum — 3-4 cümlelik başlık, en kritik KPI'lar.
2. Öncelikli Kararlar — madde madde, aksiyon + gerekçe (değer bazlı).
3. Ürün Bazlı Yorum — en değerli uyarılı ürünler için kısa yorum ve öneri.
4. Anomali Açıklaması — tespit edilen her anomali için olası neden ve aksiyon.
5. ABC-XYZ Değerlendirmesi — segmentlere göre 1-2 cümlelik strateji.

Sadece verideki sayıları kullan, uydurma. Net ve iş odaklı ol.

VERİ:
{data}
"""


def main():
    load_dotenv()
    required = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
    if [v for v in required if not os.environ.get(v)]:
        print("Missing Azure OpenAI config. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

    print("Gathering analytics snapshot ...")
    snapshot = yonetici_ozeti()

    print("Generating executive summary with Azure OpenAI ...")
    resp = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": "Deneyimli, kısa ve net yazan bir planlama direktörüsün."},
            {"role": "user", "content": PROMPT.format(data=json.dumps(snapshot, ensure_ascii=False, indent=2))},
        ],
    )
    summary = resp.choices[0].message.content

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("# Yönetici Özeti (otomatik üretildi)\n\n" + summary + "\n")
    print(f"\nSaved -> {OUT_PATH}\n")
    print(summary[:600], "...")


if __name__ == "__main__":
    main()
