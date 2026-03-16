import json

MAX_LEN = 128

# words that usually belong to menus / footer
STOPWORDS = {
    "privacy","policy","shipping","returns","contact",
    "login","instagram","linkedin","pinterest","subscribe",
    "email","copyright","faq","careers","terms"
}


def is_noise(token):
    t = token.lower()
    return (
        t in STOPWORDS or
        "@" in t or
        "http" in t or
        t.isdigit()
    )


def clean_example(tokens, labels):

    new_tokens = []
    new_labels = []

    for t, l in zip(tokens, labels):

        # remove noisy tokens if they are not labeled entities
        if is_noise(t) and l == "O":
            continue

        new_tokens.append(t)
        new_labels.append(l)

    return new_tokens, new_labels


def split_chunks(tokens, labels):

    chunks = []

    for i in range(0, len(tokens), MAX_LEN):

        ct = tokens[i:i+MAX_LEN]
        cl = labels[i:i+MAX_LEN]

        # skip very short sequences
        if len(ct) < 5:
            continue

        chunks.append({
            "tokens": ct,
            "labels": cl
        })

    return chunks


def process_file(input_path, output_path):

    new_data = []

    with open(input_path, "r", encoding="utf8") as f:

        for line in f:

            obj = json.loads(line)

            tokens = obj["tokens"]
            labels = obj["labels"]

            tokens, labels = clean_example(tokens, labels)

            chunks = split_chunks(tokens, labels)

            new_data.extend(chunks)

    with open(output_path, "w", encoding="utf8") as out:

        for item in new_data:
            out.write(json.dumps(item) + "\n")

    print("examples:", len(new_data))


if __name__ == "__main__":

    process_file(
        "data_bio.jsonl",
        "data_clean.jsonl"
    )