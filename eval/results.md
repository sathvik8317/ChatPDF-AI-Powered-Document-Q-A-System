Model `qwen2.5:7b`, 23 questions (21 answerable, 2 not in the paper).

| Search | Reranker | Retrieval hit rate | Answer accuracy | Seconds / question |
|---|---|---|---|---|
| hybrid | cross-encoder | 100% | 100% | 4.7 |
| hybrid | llm | 100% | 100% | 12.9 |
| hybrid | none | 100% | 96% | 3.9 |
| vector | cross-encoder | 100% | 100% | 4.9 |
| vector | llm | 100% | 96% | 16.7 |
| vector | none | 100% | 100% | 3.9 |
