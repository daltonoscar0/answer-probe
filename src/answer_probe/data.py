"""Question sampling for the two datasets.

The notebook calls `sample_all()` once, writes the result to `questions.jsonl`
and `data_summary.csv`, and from then on the questions file is the single source
of truth. Nothing on the analysis side needs `datasets` installed or a network
connection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Sampling seeds. Changing these changes which questions the study runs on.
TRIVIAQA_SEED = 20260101
SHIFT_SEED = 20260102

N_TRIVIAQA = 6000
N_SHIFT = 2000


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to pull a split, and everything to credit it."""

    key: str  # short name used in file columns: "triviaqa" | "nq_open"
    hf_path: str
    hf_config: str | None
    split: str
    license: str
    citation: str
    url: str
    n_sample: int
    seed: int


TRIVIAQA = DatasetSpec(
    key="triviaqa",
    hf_path="mandarjoshi/trivia_qa",
    hf_config="rc.nocontext",
    split="validation",
    # The Hugging Face card reports "unknown"; the TriviaQA repository and paper
    # release the data under Apache 2.0.
    license="Apache-2.0 (per the TriviaQA repository; the HF card says 'unknown')",
    citation="Joshi, Choi, Weld & Zettlemoyer (2017), TriviaQA, ACL",
    url="https://huggingface.co/datasets/mandarjoshi/trivia_qa",
    n_sample=N_TRIVIAQA,
    seed=TRIVIAQA_SEED,
)

NQ_OPEN = DatasetSpec(
    key="nq_open",
    hf_path="google-research-datasets/nq_open",
    hf_config="nq_open",
    split="validation",
    license="CC BY-SA 3.0",
    citation=(
        "Kwiatkowski et al. (2019), Natural Questions, TACL; "
        "open variant from Lee, Chang & Toutanova (2019), ACL"
    ),
    url="https://huggingface.co/datasets/google-research-datasets/nq_open",
    n_sample=N_SHIFT,
    seed=SHIFT_SEED,
)

SPECS = {TRIVIAQA.key: TRIVIAQA, NQ_OPEN.key: NQ_OPEN}


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    question: str
    aliases: tuple[str, ...]


def choose_indices(n_total: int, n_sample: int, seed: int) -> np.ndarray:
    """Pick `n_sample` row indices out of `n_total`, reproducibly and sorted.

    Sorted so the order a dataset happens to iterate in cannot change the
    sample, and so the notebook reads rows in one forward pass.
    """
    if n_sample > n_total:
        raise ValueError(f"asked for {n_sample} questions but the split has {n_total}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_sample, replace=False))


def _triviaqa_record(row: dict, index: int) -> Question:
    answer = row["answer"]
    aliases = [answer["value"], *answer.get("aliases", [])]
    qid = row.get("question_id") or f"triviaqa-{index}"
    return Question(
        qid=f"triviaqa:{qid}",
        dataset="triviaqa",
        question=row["question"],
        aliases=tuple(dict.fromkeys(a for a in aliases if a)),
    )


def _nq_open_record(row: dict, index: int) -> Question:
    return Question(
        qid=f"nq_open:{index}",
        dataset="nq_open",
        question=row["question"],
        aliases=tuple(dict.fromkeys(a for a in row["answer"] if a)),
    )


_RECORD_BUILDERS = {"triviaqa": _triviaqa_record, "nq_open": _nq_open_record}


def sample_questions(spec: DatasetSpec) -> tuple[list[Question], dict]:
    """Load a split from Hugging Face and take the fixed sample from it.

    Requires the `extract` extra (`pip install -e ".[extract]"`). Returns the
    questions and a summary row describing provenance.
    """
    import datasets  # imported lazily: the analysis side never needs it

    ds = datasets.load_dataset(spec.hf_path, spec.hf_config, split=spec.split)
    indices = choose_indices(len(ds), spec.n_sample, spec.seed)
    builder = _RECORD_BUILDERS[spec.key]
    rows = ds.select(indices)
    questions = [builder(row, int(i)) for row, i in zip(rows, indices, strict=True)]

    n_no_alias = sum(1 for q in questions if not q.aliases)
    summary = {
        "dataset": spec.key,
        "hf_path": spec.hf_path,
        "hf_config": spec.hf_config or "",
        "split": spec.split,
        "split_rows": len(ds),
        "n_sampled": len(questions),
        "sample_seed": spec.seed,
        "questions_without_aliases": n_no_alias,
        "mean_aliases": round(float(np.mean([len(q.aliases) for q in questions])), 3),
        "license": spec.license,
        "citation": spec.citation,
        "url": spec.url,
        "datasets_version": datasets.__version__,
        "hf_revision": getattr(ds.info, "version", "") and str(ds.info.version) or "",
    }
    return questions, summary


def sample_all() -> tuple[list[Question], list[dict]]:
    """Both datasets: the TriviaQA main set and the NQ-open shift set."""
    questions: list[Question] = []
    summaries: list[dict] = []
    for spec in (TRIVIAQA, NQ_OPEN):
        qs, summary = sample_questions(spec)
        questions.extend(qs)
        summaries.append(summary)
    qids = [q.qid for q in questions]
    if len(set(qids)) != len(qids):
        raise RuntimeError("duplicate question ids across the sampled datasets")
    return questions, summaries


def question_to_dict(q: Question) -> dict:
    d = asdict(q)
    d["aliases"] = list(q.aliases)
    return d
