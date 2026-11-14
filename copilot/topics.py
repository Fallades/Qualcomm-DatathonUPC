"""Deterministic, multilingual topic tagging for M3Kang problems.

M3Kang has no topic field. These rules provide a first-pass label and a
confidence band; they are not a substitute for human review. A manual topic
stored in ``topic`` or supplied through an overrides file always wins.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

TOPICS = {
    "arithmetic": "Aritmètica",
    "algebra": "Àlgebra",
    "geometry": "Geometria",
    "number_theory": "Teoria de nombres",
    "combinatorics": "Combinatòria",
    "probability": "Probabilitat",
    "statistics": "Estadística",
    "measurement": "Mesures i proporcionalitat",
    "logic": "Lògica i patrons",
    "other": "Altres / per revisar",
}

# Multiple-language keyword patterns. More distinctive concepts receive
# larger weights; generic words such as "number" are deliberately avoided.
_RULES = {
    "geometry": [
        (r"\b(area|perimeter|circumference|radius|diameter|triangle|square|rectangle|circle|polygon|angle|vertex|vertices|cube|sphere|parallelogram|trapezoid|rhombus|diagonal|tangent)\b", 3),
        (r"\b(are[àa]|per[ií]metre|circumfer[eè]ncia|radi|di[aà]metre|triangle|quadrat|rectangle|cercle|pol[ií]gon|angle|v[eè]rtex|cub|esfera|paral[·.]?lelogram|rombe|diagonal)\b", 3),
        (r"\b(área|perímetro|circunferencia|radio|diámetro|triángulo|cuadrado|rectángulo|círculo|polígono|ángulo|vértice|cubo|esfera|diagonal)\b", 3),
    ],
    "algebra": [
        (r"\b(equation|solve for|unknown|variable|expression|linear equation|system of equations|polynomial|factoris[ez]|expand|simplif)\w*\b", 3),
        (r"\b(equaci[oó]|inc[oò]gnita|variable|expressi[oó]|sistema d'equacions|polinomi|factoritz|desenvolup|simplific)\w*\b", 3),
        (r"\b(ecuaci[oó]|inc[oó]gnita|variable|expresi[oó]|sistema de ecuaciones|polinomio|factoriz|simplific)\w*\b", 3),
    ],
    "number_theory": [
        (r"\b(prime number|prime factor|divisib|divisor|multiple of|remainder|modulo|odd number|even number|digit(s)?|consecutive integers|greatest common divisor|least common multiple)\b", 3),
        (r"\b(nombre primer|factor primer|divisib|divisor|m[uú]ltiple|residu|m[oò]dul|parells?|senars?|xifra|nombres consecutius|m[aà]xim com[uú] divisor|m[ií]nim com[uú] m[uú]ltiple)\b", 3),
        (r"\b(n[uú]mero primo|factor primo|divisib|divisor|m[uú]ltiplo|resto|m[oó]dulo|pares?|impares?|cifra|n[uú]meros consecutivos|m[aá]ximo com[uú]n divisor|m[ií]nimo com[uú]n m[uú]ltiplo)\b", 3),
    ],
    "combinatorics": [
        (r"\b(how many ways|in how many ways|arrange|排列|permutation|combination|choose from|different arrangements|possible numbers|ways can|count the number|seating arrangement)\b", 3),
        (r"\b(de quantes maneres|quantes maneres|ordenar|permutaci[oó]|combinaci[oó]|triar|escollir|nombres diferents|quants nombres|comptar|repartir|distribuir|possibilitats)\b", 3),
        (r"\b(de cu[aá]ntas maneras|cu[aá]ntas maneras|ordenar|permutaci[oó]|combinaci[oó]|elegir|escoger|n[uú]meros diferentes|cu[aá]ntos n[uú]meros|contar|repartir|distribuir|posibilidades)\b", 3),
    ],
    "probability": [
        (r"\b(probability|probable|random|chance|likelihood|equally likely|dice|die is rolled|coin is tossed|coin flip)\b", 3),
        (r"\b(probabilitat|probable|aleatori|atzar|possibilitat que|daus?|llan[cç]a una moneda|moneda)\b", 3),
        (r"\b(probabilidad|probable|aleatorio|azar|posibilidad de que|dados?|lanza una moneda|moneda)\b", 3),
    ],
    "statistics": [
        (r"\b(mean|average|median|mode|frequency|data set|statistics|standard deviation|histogram|bar chart)\b", 3),
        (r"\b(mitjana|mediana|moda|freq[uü][eè]ncia|conjunt de dades|estad[ií]stica|desviaci[oó] est[aà]ndard|histograma|gr[aà]fic de barres)\b", 3),
        (r"\b(media|mediana|moda|frecuencia|conjunto de datos|estad[ií]stica|desviaci[oó]n est[aá]ndar|histograma|gr[aá]fico de barras)\b", 3),
    ],
    "measurement": [
        (r"\b(speed|distance|time|duration|mass|weight|volume|capacity|litres?|meters?|metres?|kilometres?|hours?|minutes?|seconds?|temperature|ratio|比例|scale drawing|unit conversion)\b", 2),
        (r"\b(velocitat|dist[aà]ncia|temps|durada|massa|pes|volum|capacitat|litres?|metres?|quil[oò]metres?|hores?|minuts?|segons?|temperatura|proporci[oó]|escala|unitats?)\b", 2),
        (r"\b(velocidad|distancia|tiempo|duraci[oó]n|masa|peso|volumen|capacidad|litros?|metros?|kil[oó]metros?|horas?|minutos?|segundos?|temperatura|proporci[oó]n|escala|unidades?)\b", 2),
    ],
    "logic": [
        (r"\b(pattern|sequence|next term|rule|riddle|logic|truth|false|always|never|must be|which statement|number puzzle)\b", 2),
        (r"\b(patr[oó]|seq[uü][eè]ncia|terme seg[uü]ent|regla|endevinalla|l[oò]gica|cert|fals|sempre|mai|ha de ser|quina afirmaci[oó])\b", 2),
        (r"\b(patr[oó]n|secuencia|t[eé]rmino siguiente|regla|acertijo|l[oó]gica|verdadero|falso|siempre|nunca|debe ser|qu[eé] afirmaci[oó]n)\b", 2),
    ],
    "arithmetic": [
        (r"\b(sum|add|addition|subtract|difference|multiply|multiplication|divide|division|product|quotient|calculate|evaluate|fraction|decimal|percent|percentage|left|remaining|remain|altogether|in total|each|gave|eats|left over)\b", 1),
        (r"\b(suma|sumar|resta|difer[eè]ncia|multiplicar|multiplicaci[oó]|dividir|divisi[oó]|producte|quocient|calcular|fracci[oó]|decimal|percentatge|tant per cent|queden|rest(en|a)|d[oó]na|donar|li resten|en total|cadascun)\b", 1),
        (r"\b(suma|sumar|resta|diferencia|multiplicar|multiplicaci[oó]n|dividir|divisi[oó]n|producto|cociente|calcular|fracci[oó]n|decimal|porcentaje|quedan|restan|da|dar|en total|cada uno)\b", 1),
    ],
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def classify_text(text: str) -> dict:
    """Return ``topic``, ``topic_confidence`` and per-topic rule scores."""
    source = _normalise(text or "")
    scores = Counter()
    for topic, rules in _RULES.items():
        for pattern, weight in rules:
            if re.search(pattern, source, flags=re.IGNORECASE):
                scores[topic] += weight
    # Mathematical symbols are useful for short arithmetic questions such as
    # "What is 2 + 3 × 4?", where there may be no natural-language keywords.
    if re.search(r"\d\s*(?:[+×*÷/−-])\s*\d", source):
        scores["arithmetic"] += 2
    if not scores:
        return {"topic": "other", "topic_confidence": "low", "topic_scores": {}}

    ordered = scores.most_common()
    best_topic, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if best_score >= 3 and best_score - second_score >= 2:
        confidence = "high"
    elif best_score >= 2 and best_score > second_score:
        confidence = "medium"
    else:
        confidence = "low"
    return {"topic": best_topic, "topic_confidence": confidence,
            "topic_scores": dict(scores)}


def classify_problem(problem: dict, overrides: dict | None = None) -> dict:
    """Return a copy of a problem with a topic. ``id|language`` overrides win."""
    result = dict(problem)
    key = f"{problem.get('id', '')}|{problem.get('language', '')}"
    existing = str(problem.get("topic", "")).strip()
    manual = (overrides or {}).get(key) or (overrides or {}).get(problem.get("id", ""))
    if manual:
        topic = str(manual).strip()
        if topic not in TOPICS:
            raise ValueError(f"Unknown topic {topic!r} for {key}; valid: {', '.join(TOPICS)}")
        result.update(topic=topic, topic_confidence="manual", topic_source="manual")
    elif existing in TOPICS and problem.get("topic_source") == "manual":
        result.update(topic=existing, topic_confidence="manual", topic_source="manual")
    else:
        tagged = classify_text(problem.get("text", ""))
        result.update(tagged)
        result["topic_source"] = "rules"
    return result
