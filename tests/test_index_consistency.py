import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from app.services.rag import RAGService


class IndexConsistencyTests(unittest.TestCase):
    def test_load_resets_when_index_checksum_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / "index.bin"
            meta_path = Path(directory) / "meta.json"

            service = RAGService.__new__(RAGService)
            service.dimension = 2
            service.index = faiss.IndexFlatL2(2)
            service.documents = ["one"]
            service.metadatas = [{"doc_id": "doc-1"}]
            service.index_path = str(index_path)
            service.meta_path = str(meta_path)
            service.index.add(np.array([[1.0, 0.0]], dtype="float32"))
            service._save_index()

            meta_path.write_text(
                json.dumps({
                    "documents": ["tampered"],
                    "metadatas": [{"doc_id": "doc-1"}],
                    "index_sha256": "not-the-file-hash",
                }),
                encoding="utf-8",
            )

            restored = RAGService.__new__(RAGService)
            restored.dimension = 2
            restored.index = faiss.IndexFlatL2(2)
            restored.documents = []
            restored.metadatas = []
            restored.index_path = str(index_path)
            restored.meta_path = str(meta_path)
            restored._load_index()

            self.assertEqual(restored.index.ntotal, 0)
            self.assertEqual(restored.documents, [])
            self.assertFalse(index_path.exists())
            self.assertFalse(meta_path.exists())


if __name__ == "__main__":
    unittest.main()
