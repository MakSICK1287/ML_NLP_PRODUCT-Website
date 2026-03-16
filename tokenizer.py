import json

input_file = "admin.jsonl" #your labeled JSONL
output_file = "data_bio.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)

        text = obj["text"]
        labels = obj.get("label", [])

        tokens = text.split()  # simple whitespace tokenizer
        token_labels = []

        idx = 0  # current index in the text

        for token in tokens:
            token_start = text.find(token, idx)
            token_end = token_start + len(token)

            token_label = "O"

            for lab_start, lab_end, lab_type in labels:
                if token_start >= lab_start and token_end <= lab_end:

                    if token_start == lab_start:
                        token_label = f"B-{lab_type}"
                    else:
                        token_label = f"I-{lab_type}"

                    break

            token_labels.append(token_label)
            idx = token_end

        out_obj = {
            "tokens": tokens,
            "labels": token_labels
        }

        f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")