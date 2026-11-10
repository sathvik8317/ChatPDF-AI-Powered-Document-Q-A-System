Model `qwen2.5:7b`, 23 questions (21 answerable, 2 not in the paper).

| Search | Reranker | Retrieval hit rate | Answer accuracy | Seconds / question |
|---|---|---|---|---|
| hybrid | cross-encoder | 100% | 96% | 4.6 |
| hybrid | llm | 100% | 87% | 12.9 |
| hybrid | none | 100% | 96% | 3.5 |
| vector | cross-encoder | 100% | 96% | 4.7 |
| vector | llm | 100% | 83% | 16.7 |
| vector | none | 95% | 91% | 3.2 |
