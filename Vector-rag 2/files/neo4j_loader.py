"""
neo4j_loader.py — يولّد Cypher queries ويحطهم في Neo4j
"""

from neo4j import GraphDatabase

# ===================== إعدادات Neo4j =====================
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "123456789"
# =========================================================


def load_to_neo4j(doc_id: str, text: str, entities: list, pos_tags: list):
    """احفظ الـ Document + Entities + POS في Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.execute_write(_create_graph, doc_id, text, entities, pos_tags)
    finally:
        driver.close()


def _create_graph(tx, doc_id, text, entities, pos_tags):
    # 1) Document node
    tx.run(
        "MERGE (d:Document {id: $doc_id}) SET d.text_preview = $preview",
        doc_id=doc_id, preview=text[:300]
    )

    # 2) Entity nodes + علاقة HAS_ENTITY
    for ent in entities:
        tx.run(
            """
            MERGE (e:Entity {text: $text, label: $label})
            WITH e
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_ENTITY {start: $start, end: $end}]->(e)
            """,
            text=str(ent.get("text", "")),
            label=str(ent.get("label", "MISC")),
            doc_id=doc_id,
            start=int(ent.get("start", 0)),
            end=int(ent.get("end", 0))
        )

    # 3) POS Token nodes + علاقة HAS_TOKEN
    for tok in pos_tags:
        tx.run(
            """
            MERGE (t:Token {text: $token, pos: $pos})
            WITH t
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_TOKEN]->(t)
            """,
            token=str(tok.get("token", "")),
            pos=str(tok.get("pos", "")),
            doc_id=doc_id
        )

    # 4) علاقات بين الـ entities في نفس الـ document
    _link_cooccurring_entities(tx, doc_id, entities)


def _link_cooccurring_entities(tx, doc_id, entities):
    """اربط الـ entities اللي بتظهر مع بعض في نفس الـ doc."""
    ent_texts = list({e["text"] for e in entities})  # unique
    for i in range(len(ent_texts)):
        for j in range(i + 1, len(ent_texts)):
            tx.run("""
                MATCH (e1:Entity {text: $t1}), (e2:Entity {text: $t2})
                MERGE (e1)-[:CO_OCCURS_WITH {doc_id: $doc_id}]->(e2)
            """, t1=ent_texts[i], t2=ent_texts[j], doc_id=doc_id)


def generate_cypher_preview(doc_id: str, entities: list, pos_tags: list) -> str:
    """اطبع الـ Cypher queries اللي هتتنفّذ (للمراجعة)."""
    lines = [
        f"// === Document: {doc_id} ===",
        f'MERGE (d:Document {{id: "{doc_id}"}});',
        ""
    ]
    for ent in entities[:5]:  # أول 5 بس للمعاينة
        lines.append(
            f'MERGE (e:Entity {{text: "{ent["text"]}", label: "{ent["label"]}"}});'
        )
        lines.append(
            f'MATCH (d:Document {{id: "{doc_id}"}}), (e:Entity {{text: "{ent["text"]}"}}) '
            f'MERGE (d)-[:HAS_ENTITY]->(e);'
        )
    lines.append(f"\n// ... و {max(0, len(entities)-5)} entity تانية")
    return "\n".join(lines)