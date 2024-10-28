from keywordextract import kw_model, vectorizer


def main():
    print("Model loaded! reading input...")

    with open("input.txt", encoding="utf-8") as f:
        doc = f.read()

    keywords = kw_model.extract_keywords(
        doc,
        vectorizer=vectorizer,
        top_n=10,
        keyphrase_ngram_range=(1, 3),
    )

    print(keywords)


if __name__ == "__main__":
    main()
