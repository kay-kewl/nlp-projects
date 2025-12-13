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
