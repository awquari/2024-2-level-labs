"""
Lab 2.

Text retrieval with BM25
"""

import json
import math

# pylint:disable=too-many-arguments, unused-argument


def tokenize(text: str) -> list[str] | None:
    """
    Tokenize the input text into lowercase words without punctuation, digits and other symbols.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str] | None: A list of words from the text.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None
    tokens = []
    text = text.lower()
    for word in text.split():
        token = []
        for symbol in word:
            if symbol.isalpha():
                token.append(symbol)
            else:
                if token:
                    tokens.append("".join(token))
                token = []
        if token:
            tokens.append("".join(token))
    return tokens


def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str] | None:
    """
    Remove stopwords from the list of tokens.

    Args:
        tokens (list[str]): List of tokens.
        stopwords (list[str]): List of stopwords.

    Returns:
        list[str] | None: Tokens after removing stopwords.

    In case of corrupt input arguments, None is returned.
    """
    right_input = isinstance(tokens, list) and isinstance(stopwords, list) and all(
        isinstance(token, str) for token in tokens) and all(
        isinstance(word, str) for word in stopwords) and tokens and stopwords
    if not right_input:
        return None
    return list(filter(lambda word: word not in stopwords, tokens))


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) \
            or not all(isinstance(doc, list) for doc in documents) \
            or not documents:
        return None
    for doc in documents:
        if not all(isinstance(word, str) for word in doc):
            return None
    vocab = []
    for tokenize_doc in documents:
        for word in tokenize_doc:
            if word not in vocab:
                vocab.append(word)
    return vocab


def calculate_tf(vocab: list[str], document_tokens: list[str]) -> dict[str, float] | None:
    """
    Calculate term frequency for the given tokens based on the vocabulary.

    Args:
        vocab (list[str]): Vocabulary list.
        document_tokens (list[str]): Tokenized document.

    Returns:
        dict[str, float] | None: Mapping from vocabulary terms to their term frequency.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = (isinstance(vocab, list) and isinstance(document_tokens, list)
                      and all(isinstance(word, str) for word in vocab)
                      and all(isinstance(token, str) for token in document_tokens)
                      and vocab and document_tokens)
    if not is_valid_input:
        return None
    tf_dict = {}
    for word in vocab:
        tf_dict[word] = document_tokens.count(word) / abs(len(document_tokens))
    for word in document_tokens:
        tf_dict[word] = document_tokens.count(word) / abs(len(document_tokens))
    return tf_dict


def calculate_idf(vocab: list[str], documents: list[list[str]]) -> dict[str, float] | None:
    """
    Calculate inverse document frequency for each term in the vocabulary.

    Args:
        vocab (list[str]): Vocabulary list.
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        dict[str, float] | None: Mapping from vocabulary terms to its IDF scores.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = (isinstance(vocab, list) and isinstance(documents, list)
                      and all(isinstance(word, str) for word in vocab)
                      and all(isinstance(doc, list) for doc in documents) and vocab != []
                      and documents != [])
    if not is_valid_input:
        return None
    for doc in documents:
        if not all(isinstance(word, str) for word in doc):
            return None
    idf_dic = {}
    len_doc = len(documents)
    for word in vocab:
        amount = 0
        for tokenize_doc in documents:
            if word in tokenize_doc:
                amount += 1
        idf_dic[word] = math.log((len_doc - amount + 0.5) / (amount + 0.5))
    return idf_dic


def calculate_tf_idf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    """
    Calculate TF-IDF scores for a document.

    Args:
        tf (dict[str, float]): Term frequencies for the document.
        idf (dict[str, float]): Inverse document frequencies.

    Returns:
        dict[str, float] | None: Mapping from terms to their TF-IDF scores.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = (((isinstance(tf, dict) and isinstance(idf, dict)
                        and all((isinstance(key, str) and isinstance(value, float)
                                 for key, value in tf.items())))
                       and all((isinstance(key, str) and isinstance(value, float)
                                for key, value in idf.items())))
                      and tf and idf)
    if not is_valid_input:
        return None

    tf_idf_dict = {}
    for word in tf:
        tf_idf_dict[word] = tf[word] * idf[word]
    return tf_idf_dict


def calculate_bm25(
    vocab: list[str],
    document: list[str],
    idf_document: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float | None = None,
    doc_len: int | None = None,
) -> dict[str, float] | None:
    """
    Calculate BM25 scores for a document.

    Args:
        vocab (list[str]): Vocabulary list.
        document (list[str]): Tokenized document.
        idf_document (dict[str, float]): Inverse document frequencies.
        k1 (float): BM25 parameter.
        b (float): BM25 parameter.
        avg_doc_len (float | None): Average document length.
        doc_len (int | None): Length of the document.

    Returns:
        dict[str, float] | None: Mapping from terms to their BM25 scores.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = ((((isinstance(idf_document, dict) and isinstance(k1, float)
                         and isinstance(b, float) and isinstance(doc_len, int))
                        and isinstance(avg_doc_len, float) and isinstance(vocab, list))
                       and isinstance(document, list) and document and vocab)
                      and idf_document and not isinstance(doc_len, bool))
    if not is_valid_input or doc_len is None or avg_doc_len is None:
        return None
    if not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(token, str) for token in document):
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    bm25_dict = {}
    for word in vocab:
        n = document.count(word)
        if idf_document.get(word):
            bm25_dict[word] = idf_document[word] * (n * (k1 + 1)) / (
                    n + k1 * (1 - b + (b * doc_len / avg_doc_len)))
        else:
            bm25_dict[word] = 0.0
    for word in document:
        n = document.count(word)
        if idf_document.get(word):
            bm25_dict[word] = idf_document[word] * (n * (k1 + 1)) / (
                    n + k1 * (1 - b + (b * doc_len / avg_doc_len)))
        else:
            bm25_dict[word] = 0.0
    return bm25_dict


def rank_documents(
    indexes: list[dict[str, float]], query: str, stopwords: list[str]
) -> list[tuple[int, float]] | None:
    """
    Rank documents for the given query.

    Args:
        indexes (list[dict[str, float]]): List of BM25 or TF-IDF indexes for the documents.
        query (str): The query string.
        stopwords (list[str]): List of stopwords.

    Returns:
        list[tuple[int, float]] | None: Tuples of document index and its score in the ranking.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = (isinstance(query, str) and isinstance(stopwords, list)
                      and isinstance(indexes, list) and stopwords != []
                      and indexes != [] and query != '')
    if not is_valid_input:
        return None
    if not all(isinstance(stopword, str) for stopword in stopwords) or \
            not all(isinstance(vocab, dict) for vocab in indexes) or \
            not all(
                (isinstance(key, str) and isinstance(value, float) for key, value in vocab.items())
                for vocab in indexes):
        return None

    tokenized_query = tokenize(query)
    if tokenized_query is None:
        return None
    tokenize_str = remove_stopwords(tokenized_query, stopwords)
    if not tokenize_str:
        return None
    result = []
    n = 0
    for text in indexes:
        sum_result = 0.0
        for word in tokenize_str:
            if word in text.keys():
                sum_result += text[word]
        tuple_result = (n, sum_result)
        n += 1
        result.append(tuple_result)
    result.sort(key=lambda a: a[1], reverse=True)
    return result


def calculate_bm25_with_cutoff(
    vocab: list[str],
    document: list[str],
    idf_document: dict[str, float],
    alpha: float,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float | None = None,
    doc_len: int | None = None,
) -> dict[str, float] | None:
    """
    Calculate BM25 scores for a document with IDF cutoff.

    Args:
        vocab (list[str]): Vocabulary list.
        document (list[str]): Tokenized document.
        idf_document (dict[str, float]): Inverse document frequencies.
        alpha (float): IDF cutoff threshold.
        k1 (float): BM25 parameter.
        b (float): BM25 parameter.
        avg_doc_len (float | None): Average document length.
        doc_len (int | None): Length of the document.

    Returns:
        dict[str, float] | None: Mapping from terms to their BM25 scores with cutoff applied.

    In case of corrupt input arguments, None is returned.
    """
    is_valid_input = (isinstance(idf_document, dict) and isinstance(k1, float)
                      and isinstance(b, float) and isinstance(doc_len, int)
                      and isinstance(avg_doc_len, float) and isinstance(vocab, list)
                      and isinstance(document, list) and isinstance(alpha, float)
                      and document != [] and vocab != [] and idf_document != {}
                      and not isinstance(doc_len, bool)
                      and not isinstance(avg_doc_len, bool) and doc_len > 0)
    if not is_valid_input or doc_len is None or avg_doc_len is None:
        return None

    if k1 < 1.2 or k1 > 2.0 or b < 0 or b > 1:
        return None
    if not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(token, str) for token in document):
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    bm25_dict = {}
    for word in vocab:
        n = document.count(word)
        if idf_document[word] > alpha:
            bm25_dict[word] = idf_document[word] * (n * (k1 + 1)) / (
                    n + k1 * (1 - b + (b * doc_len / avg_doc_len)))
    return bm25_dict


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not isinstance(file_path, str) or file_path == '':
        return None
    if not isinstance(index, list) or index == []:
        return None

    for idx in index:
        if isinstance(idx, dict):
            if not all(isinstance(key, str) and isinstance(val, float) for key, val in idx.items()):
                return None
        else:
            return None

    with open(file_path, "w", encoding="UTF-8") as file:
        json.dump(index, file, indent=4)
    return None


def load_index(file_path: str) -> list[dict[str, float]] | None:
    """
    Load the index from a file.

    Args:
        file_path (str): The path to the file from which to load the index.

    Returns:
        list[dict[str, float]] | None: The loaded index.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(file_path, str) or file_path == '':
        return None
    with open(file_path, 'r', encoding="UTF-8") as file:
        result: list[dict[str, float]] = json.load(file)
        return result


def calculate_spearman(rank: list[int], golden_rank: list[int]) -> float | None:
    """
    Calculate Spearman's rank correlation coefficient between two rankings.

    Args:
        rank (list[int]): Ranked list of document indices.
        golden_rank (list[int]): Golden ranked list of document indices.

    Returns:
        float | None: Spearman's rank correlation coefficient.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(rank, list) or not isinstance(golden_rank, list) or \
            not all(isinstance(indices, int) for indices in rank) or \
            not all(isinstance(golden_indices, int) for golden_indices in golden_rank):
        return None
    if not rank or not golden_rank or len(rank) != len(golden_rank):
        return None

    len_rank = len(rank)
    return 1 - (6 * sum((rank.index(golden_rank[i]) - i) ** 2 for i in range(len_rank)
                        if golden_rank[i] in rank) / (len_rank * (len_rank * len_rank - 1)))
