def build_ner_prompt(doc_text, entities):
    """
    Constructs a strict prompt for the DeepSeek model to perform NER classification.
    """
    entities_list = "\n".join(sorted(set(entities)))
    return (
        "Read the following news text:\n"
        f"{doc_text}\n\n"
        "Task: Determine the entity type (PER, ORG, LOC, EVT, PRO) "
        "for the following list of entities found in the text.\n"
        "Answer strictly in the format:\n"
        "Entity -> TYPE\n\n"
        "List of entities:\n"
        f"{entities_list}\n"
    )


def parse_deepseek_response(response_text):
    """Parses "Entity -> TYPE" lines into a dictionary."""
    predictions = {}
    lines = response_text.strip().split("\n")
    for line in lines:
        if "->" in line:
            parts = line.split("->")
            if len(parts) == 2:
                ent_name = parts[0].strip()
                label = parts[1].strip().upper()
                predictions[ent_name] = label
    return predictions

def map_deepseek_responses(raw_responses, df, label_encoder, default_class="PER"):
    df_mapped = df.copy()
    df_mapped["deepseek_pred"] = default_class
    known_classes = set(label_encoder.classes_)
    
    for doc_id, text_resp in raw_responses.items():
        parsed = parse_deepseek_response(text_resp)
        mask = df_mapped["document_id"] == doc_id
        
        df_mapped.loc[mask, "deepseek_pred"] = df_mapped.loc[mask, "entity"].apply(
            lambda x: parsed.get(x, default_class) if parsed.get(x) in known_classes else default_class
        )
    return df_mapped
