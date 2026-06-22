# input/

- `eval_set.jsonl` — готовая стратифицированная выборка (18 пар), на ней работает `eval.py` из коробки.
- Чтобы пересобрать выборку другого размера/seed, положите сюда `train.csv` и
  `test.csv` из датасета Hugging Face `cnamuangtoun/resume-job-description-fit`
  и запустите `python prepare_data.py --per-class N`.

Скачать датасет:
    from datasets import load_dataset
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    ds["test"].to_csv("input/test.csv", index=False)
    ds["train"].to_csv("input/train.csv", index=False)
