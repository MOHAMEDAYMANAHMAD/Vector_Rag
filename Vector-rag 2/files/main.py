#!/usr/bin/env python3
"""
NLP Pipeline: PDF/TXT → Ollama (gemma3:4b) NER+POS → Neo4j
"""

import sys
import json
from pathlib import Path
from extractor import extract_text
from ollama_nlp import run_nlp
from neo4j_loader import load_to_neo4j

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def get_checkpoint(doc_id: str):
    cp = CHECKPOINT_DIR / f"{doc_id}.json"
    if cp.exists():
        data = json.loads(cp.read_text())
        print(f"  ⚡ لقى checkpoint محفوظ، هيتخطى الـ Ollama!")
        return data
    return None


def save_checkpoint(doc_id: str, text: str, entities: list, pos_tags: list):
    cp = CHECKPOINT_DIR / f"{doc_id}.json"
    cp.write_text(json.dumps({"text": text, "entities": entities, "pos_tags": pos_tags}, ensure_ascii=False))
    print(f"  💾 اتحفظ checkpoint في {cp}")


def process_file(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"❌ الملف مش موجود: {file_path}")
        return

    print(f"\n📄 بيشتغل على: {path.name}")

    # Step 1: Extract text
    print("  [1/3] بيستخرج النص...")
    text = extract_text(path)
    if not text.strip():
        print("  ❌ النص فاضي!")
        return

    LINE_LIMIT = 200
    if LINE_LIMIT:
        lines = text.splitlines()[:LINE_LIMIT]
        text  = "\n".join(lines)
        print(f"  ✅ شغّال على أول {LINE_LIMIT} سطر ({len(text)} حرف)")
    else:
        print(f"  ✅ استخرج {len(text)} حرف")

    doc_id = path.stem

    # Step 2: NER + POS — لو في checkpoint يتخطاها
    cp = get_checkpoint(doc_id)
    if cp:
        entities = cp["entities"]
        tokens   = cp["pos_tags"]
        text     = cp["text"]
        print(f"  ✅ {len(entities)} entity و {len(tokens)} token من الـ checkpoint")
    else:
        print("  [2/3] بيشغّل NER + POS على gemma3:4b...")
        nlp_result = run_nlp(text)
        if not nlp_result:
            print("  ❌ Ollama مردتش نتيجة!")
            return
        entities = nlp_result.get("entities", [])
        tokens   = nlp_result.get("pos_tags", [])
        print(f"  ✅ لقى {len(entities)} entity و {len(tokens)} token")
        save_checkpoint(doc_id, text, entities, tokens)

    # Step 3: Load to Neo4j
    print("  [3/3] بيحمّل في Neo4j...")
    load_to_neo4j(doc_id, text, entities, tokens)
    print(f"  ✅ اتحمّل في Neo4j بنجاح!\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("الاستخدام: python main.py <file1.txt> <file2.pdf> ...")
        sys.exit(1)

    for f in sys.argv[1:]:
        process_file(f)

    print("🎉 خلص كل حاجة!")