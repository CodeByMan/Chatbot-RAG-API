# RAG evaluation protocol

The repository does not contain a representative document set or labelled
questions, so no retrieval or answer-quality score is claimed. Create a local,
non-sensitive evaluation set before publishing benchmark results.

Each question should record:

```json
{"question":"...", "expected_answer":"...", "expected_sources":["file.pdf"]}
```

For each item, record retrieval recall (whether an expected source was returned),
answer correctness, citation/source correctness, and the behaviour for an
unanswerable question. Run the same set after changes to chunking, embeddings,
retrieval thresholds, prompts, or models. Report the dataset size, model names,
configuration, and failure cases alongside any measured results.

Until that process is completed, describe this project as a working RAG API with
an evaluation protocol, not as a quantitatively validated RAG system.
