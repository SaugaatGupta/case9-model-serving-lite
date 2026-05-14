"""Hammer /predict with 500 French/Spanish requests of unusual lengths.

Used in the demo to flip GET /drift to drift_detected.
"""
import argparse
import random

import requests

FR = [
    "J'adore ce produit, il est absolument fantastique !",
    "C'est vraiment horrible, je ne le recommande pas du tout.",
    "Le service client était impeccable, merci beaucoup.",
    "Quelle déception, je ne reviendrai jamais ici.",
    "Un excellent rapport qualité-prix, je suis ravi de mon achat.",
    "La livraison a été extrêmement lente et le colis abîmé.",
    "Magnifique, je recommande chaudement à tout le monde.",
    "Une catastrophe totale, à fuir absolument.",
]
ES = [
    "Me encanta este producto, es maravilloso de verdad.",
    "Es terrible, no lo recomendaría a nadie en absoluto.",
    "El servicio fue excelente, muy satisfecho con la compra.",
    "Qué decepción, una pérdida total de tiempo y dinero.",
    "Una experiencia fantástica de principio a fin, gracias.",
    "Llegó roto y el soporte fue pésimo, evitar.",
    "Compra acertada, calidad superior, muy contento.",
    "Lo peor que he comprado en años, una estafa.",
]


def gen_text() -> str:
    base = random.choice(FR + ES)
    r = random.random()
    if r < 0.4:
        return base[: random.randint(1, 10)]          # absurdly short
    if r < 0.8:
        return (base + " ") * random.randint(10, 25)  # absurdly long
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    for i in range(args.n):
        text = gen_text()
        try:
            r = requests.post(f"{args.url}/predict", json={"text": text}, timeout=15)
            if i % 50 == 0:
                print(f"[{i}] {r.status_code} {r.json().get('label')}")
        except Exception as e:
            print(f"[{i}] error: {e}")
    print("done")


if __name__ == "__main__":
    main()
