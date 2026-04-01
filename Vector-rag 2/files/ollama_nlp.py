"""
ollama_nlp.py — يبعت النص لـ gemma3:4b ويجيب NER + POS
"""

import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "gemma3:4b"

# كل نص أطول من كده هنقطعه لـ chunks
MAX_CHARS  = 3000


PROMPT_TEMPLATE = """You are an NLP expert. Analyze the following English text and return ONLY a valid JSON object with this exact structure:

{{
  "entities": [
    {{"text": "entity text", "label": "ENTITY_TYPE", "start": 0, "end": 5}}
  ],
  "pos_tags": [
    {{"token": "word", "pos": "POS_TAG"}}
  ]
}}

Entity types to use: PERSON, ORG, GPE, DATE, MONEY, PRODUCT, EVENT, LOC, MISC
POS tags to use: NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, JJ, RB, DT, IN, CC, PRP, CD

Rules:
- Return ONLY the JSON, no explanation, no markdown, no code fences
- Keep pos_tags to max 30 most important tokens
- If no entities found, return empty list

Text:
{text}"""


def run_nlp(text: str) -> dict:
    """شغّل NER + POS على النص كامل (يقطعه لو طويل)."""
    chunks = _chunk_text(text, MAX_CHARS)
    all_entities = []
    all_pos      = []
    char_offset  = 0

    for i, chunk in enumerate(chunks):
        print(f"     chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        result = _call_ollama(chunk)
        if result:
            # اضبط الـ offsets حسب مكان الـ chunk في النص الأصلي
            for ent in result.get("entities", []):
                ent["start"] = ent.get("start", 0) + char_offset
                ent["end"]   = ent.get("end",   0) + char_offset
                all_entities.append(ent)
            all_pos.extend(result.get("pos_tags", []))
        char_offset += len(chunk)

    # شيل الـ POS المكررة
    seen = set()
    unique_pos = []
    for t in all_pos:
        key = t.get("token", "")
        if key not in seen:
            seen.add(key)
            unique_pos.append(t)

    return {"entities": all_entities, "pos_tags": unique_pos}


def _call_ollama(text: str, retries: int = 3) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(text=text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048}
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            return _parse_json(raw)
        except requests.exceptions.Timeout:
            print(f"     ⏱️  Timeout (محاولة {attempt}/{retries})، بيعيد...")
        except requests.RequestException as e:
            print(f"     ⚠️  Ollama error: {e} (محاولة {attempt}/{retries})")
        
        if attempt == retries:
            print(f"     ❌ فشل بعد {retries} محاولات، بيكمل على الـ chunk الجاي...")
            return None


def _parse_json(raw: str) -> dict | None:
    """استخرج JSON من رد الموديل حتى لو فيه كلام زيادة أو اتقطع."""
    # شيل code fences لو موجودة
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # جرب parse مباشرة
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # حاول تلاقي أول { في النص
    start = raw.find("{")
    if start == -1:
        print(f"     ⚠️  مفيش JSON خالص:\n{raw[:200]}")
        return None

    json_str = raw[start:]

    # لو الـ JSON اتقطع، حاول تكمّله
    json_str = _fix_truncated_json(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    print(f"     ⚠️  مش قادر يـparse الـJSON:\n{raw[:200]}")
    return None


def _fix_truncated_json(s: str) -> str:
    """حاول تصلح JSON مقطوع بإغلاق الـ brackets الناقصة."""
    # شيل آخر object ناقص (اللي مفيهوش closing bracket)
    # لو آخر حاجة مش ] أو }، احذف لحد آخر , أو [ أو {
    s = s.rstrip()

    # لو خلص بـ , أو بـ object مفتوح، قطع من آخر entity كاملة
    while s and s[-1] not in ("]", "}"):
        # ارجع لآخر } كاملة
        last_close = max(s.rfind("}"), 0)
        if last_close == 0:
            break
        s = s[:last_close + 1]

    # عدّ الـ brackets الناقصة وأغلّها
    open_brackets  = s.count("[") - s.count("]")
    open_braces    = s.count("{") - s.count("}")

    s += "]" * open_brackets
    s += "}" * open_braces

    return s


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """قطّع النص على حدود الجمل."""
    if len(text) <= max_chars:
        return [text]

    chunks, current = [], ""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current) + len(sentence) > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
    if current:
        chunks.append(current.strip())
    return chunks