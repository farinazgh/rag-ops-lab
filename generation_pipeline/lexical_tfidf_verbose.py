from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

TEXTS = [
    "Australia won the Cricket World Cup 2023",
    "India and Australia played in the finals",
    "Australia won the sixth time having last won in 2015",
]

QUERY = "won"


def main():
    #  Build vectorizer
    vectorizer = TfidfVectorizer()

    #  Learn vocabulary + compute document vectors
    tfidf_matrix = vectorizer.fit_transform(TEXTS)

    #  Show vocabulary
    print("=== Vocabulary ===")
    print(vectorizer.vocabulary_)
    print()

    #  Show TF-IDF matrix
    print("=== TF-IDF Matrix (Documents) ===")
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    print(df)
    print()

    #  Transform query into vector
    query_vector = vectorizer.transform([QUERY])

    print("=== Query Vector ===")
    print(pd.DataFrame(
        query_vector.toarray(),
        columns=vectorizer.get_feature_names_out()
    ))
    print()

    #  Compute cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    print("=== Cosine Similarity Scores ===")
    for i, score in enumerate(similarities[0]):
        print(f"Doc {i + 1}: {score:.4f}")

    #   Rank results
    ranked_indices = similarities[0].argsort()[::-1]

    print("\n=== Ranked Results ===")
    for idx in ranked_indices:
        print(TEXTS[idx])


if __name__ == "__main__":
    main()
