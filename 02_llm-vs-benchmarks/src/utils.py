import pandas as pd

def read_bsnlp_document(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_id = lines[0].strip()
    language = lines[1].strip()
    creation_date = lines[2].strip()
    url = lines[3].strip()
    title = lines[4].strip()

    body = "\n".join(lines[5:])
    full_text = title if not body.strip() else f"{title}\n{body}"

    meta = {
        "language": language,
        "creation_date": creation_date,
        "url": url,
        "title": title,
        "path": str(path),
    }
    return doc_id, full_text, meta


def read_bsnlp_annotations(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    doc_id = lines[0].strip()
    records = []

    for line in lines[1:]:
        parts = line.split("\t")
        mention, lemma, entity_type, cross_id = (p.strip() for p in parts)
        records.append(
            {
                "mention": mention,
                "lemma": lemma,
                "entity_type": entity_type,
                "cross_lingual_id": cross_id,
            }
        )

    ann_df = pd.DataFrame(records)
    return doc_id, ann_df
