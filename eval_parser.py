"""eval_parser.py: the Expert Nitpicker evaluation pipeline for LLM chat logs.

Target data is SALT-NLP/SWE-chat, conversations config. These are Claude Code
agent session logs. Each row is one turn. Most turns are tool_use, tool_result,
or metadata; the real chat turns carry is_conversational true and role user or
assistant. The set also ships its own labels: prompt_intent and prompt_pushback,
where a user pushback of correction or failure_report is direct evidence that the
prior assistant turn was flawed.

Two halves.

Data processing (token efficient). Logs load through datasets into a pandas
DataFrame, one row per turn, keyed by session_id and turn_number. Search by
keyword, regex, turn index, or sentence transformer similarity. Views return only
role and content, truncated, so nothing dumps raw JSON into the terminal. The flag
function marks likely rejections and corrections with regex, and also surfaces
assistant turns that drew a user pushback.

The four layer architecture (the pipeline). Each conversational assistant turn is
judged:
  Layer 0  Base Task    restate the original user prompt and the core objective
  Layer 1  Memory       track conversational state and the expected next action
  Layer 2  Nitpicker    strict auditor, hunts hallucinations, lazy code, false rejections
  Layer J  Judge        on a flaw, emits BAD CLAUDE feedback, a category, a correction

The LLM loop is fully defined here but never run on import. Call run_pipeline
yourself when you are ready to spend tokens.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Guarded optional imports, same pattern as alpha_evolver.py. The module stays
# importable with only numpy and the standard library. Each function checks its
# flag and raises a clear error if the dep is missing.
try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False

try:
    import anthropic
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False

try:
    import datasets
    HAVE_DATASETS = True
except ImportError:
    HAVE_DATASETS = False

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False


# ----------------------------------------------------------------------------
# Part 1: data processing
# ----------------------------------------------------------------------------

SWE_CHAT = "SALT-NLP/SWE-chat"

# Canonical frame. idx is a stable row reference after sorting by session then
# turn. turn is the raw turn_number; conv_turn is the conversation only index.
COLUMNS = ("idx", "session_id", "turn", "conv_turn", "role", "turn_type",
           "is_conversational", "tool_name", "content", "prompt_intent",
           "prompt_pushback")

# User pushback values that mark the prior assistant turn as suspect.
PUSHBACK_FLAGS = ("correction", "failure_report", "rejection", "clarification")

# Regex signals. Word boundaries keep them tight. Case is ignored at match time.
REJECTION_PATTERNS = (
    r"\bI cannot\b",
    r"\bI can'?t\b",
    r"\bAs an AI\b",
    r"\bI'?m unable\b",
    r"\bI am unable\b",
    r"\bI'?m not able\b",
    r"\bI will not\b",
    r"\bI'?m sorry,? but\b",
    r"\bagainst (?:my|our) (?:guidelines|policy|programming)\b",
    r"\bI do not feel comfortable\b",
)

CORRECTION_PATTERNS = (
    r"\bApolog",
    r"\bI apologize\b",
    r"\bYou'?re right\b",
    r"\bYou are right\b",
    r"\bMy mistake\b",
    r"\bI was wrong\b",
    r"\bLet me correct\b",
    r"\bgood catch\b",
    r"\bmy apologies\b",
    r"\bsorry for the\b",
)

_REJECTION_RE = re.compile("|".join(REJECTION_PATTERNS), re.IGNORECASE)
_CORRECTION_RE = re.compile("|".join(CORRECTION_PATTERNS), re.IGNORECASE)


def _require_pandas():
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required. Run: pip install pandas")


def _normalize_role(role: Any) -> str:
    """Lowercase the role. SWE-chat uses user, assistant, tool_use, tool_result,
    metadata; map a few synonyms onto user and assistant, leave the rest as is."""
    r = str(role or "").strip().lower()
    if r in ("human", "prompter"):
        return "user"
    if r in ("gpt", "model", "ai", "bot", "claude"):
        return "assistant"
    return r or "unknown"


def _normalize_content(content: Any) -> str:
    """Flatten content into one string. Conversational turns are plain text; tool
    and metadata turns are JSON strings, kept verbatim and truncated at render."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, default=str)
    return str(content)


def load_swe_chat(config: str = "conversations", split: str = "train",
                  limit: Optional[int] = None, streaming: bool = True
                  ) -> "pd.DataFrame":
    """Load SALT-NLP/SWE-chat into the canonical DataFrame, one row per turn.

    The dataset is gated, so accept access on its Hub page and authenticate first
    (hf auth login, or set HF_TOKEN). limit caps rows for a quick look. Rows are
    sorted by session_id then turn, and idx is reassigned as a stable reference.
    """
    _require_pandas()
    if not HAVE_DATASETS:
        raise RuntimeError("datasets is required. Run: pip install datasets")
    ds = datasets.load_dataset(SWE_CHAT, config, split=split, streaming=streaming)

    records = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        records.append((
            ex.get("session_id"),
            ex.get("turn_number"),
            ex.get("conversation_turn_number"),
            _normalize_role(ex.get("role")),
            ex.get("turn_type"),
            bool(ex.get("is_conversational")),
            ex.get("tool_name"),
            _normalize_content(ex.get("content")),
            ex.get("prompt_intent"),
            ex.get("prompt_pushback"),
        ))

    df = pd.DataFrame.from_records(
        records,
        columns=[c for c in COLUMNS if c != "idx"],
    )
    df = df.sort_values(["session_id", "turn"], kind="stable").reset_index(drop=True)
    df.insert(0, "idx", np.arange(len(df)))
    return df


def conversational(df: "pd.DataFrame") -> "pd.DataFrame":
    """The real chat turns only: user prompts and assistant responses."""
    _require_pandas()
    return df[df["is_conversational"]]


def view(df: "pd.DataFrame", indices=None, width: int = 200, max_rows: int = 40
         ) -> "pd.DataFrame":
    """Return a compact role and content table. Content is truncated to width so
    the terminal stays clean. Pass indices to view a specific subset."""
    _require_pandas()
    sub = df if indices is None else df[df["idx"].isin(list(indices))]
    sub = sub.head(max_rows).copy()
    sub["content"] = sub["content"].astype(str).str.slice(0, width)
    return sub[["idx", "session_id", "turn", "role", "content"]]


def search_keyword(df: "pd.DataFrame", keyword: str, case: bool = False,
                   role: Optional[str] = None) -> "pd.DataFrame":
    """Substring search over content. Optionally restrict to one role."""
    _require_pandas()
    mask = df["content"].astype(str).str.contains(re.escape(keyword), case=case, regex=True)
    if role:
        mask &= df["role"].eq(_normalize_role(role))
    return view(df, df.loc[mask, "idx"])


def search_regex(df: "pd.DataFrame", pattern: str, flags: int = re.IGNORECASE,
                 role: Optional[str] = None) -> "pd.DataFrame":
    """Regex search over content. Pattern is a raw regex string."""
    _require_pandas()
    rx = re.compile(pattern, flags)
    mask = df["content"].astype(str).apply(lambda s: rx.search(s) is not None)
    if role:
        mask &= df["role"].eq(_normalize_role(role))
    return view(df, df.loc[mask, "idx"])


def get_index(df: "pd.DataFrame", idx: int) -> "pd.DataFrame":
    """One turn by global index, full content."""
    return view(df, [idx], width=10_000)


def get_turn(df: "pd.DataFrame", turn: int, session_id: Optional[str] = None
             ) -> "pd.DataFrame":
    """Turns at a given turn_number, optionally within one session, full content."""
    _require_pandas()
    sel = df[df["turn"].eq(turn)]
    if session_id is not None:
        sel = sel[sel["session_id"].eq(str(session_id))]
    return view(df, sel["idx"], width=10_000)


def flag_signals(df: "pd.DataFrame") -> "pd.DataFrame":
    """Flag suspect assistant turns from two sources.

    Regex: rejection and correction phrases inside an assistant turn.
    Pushback: a user turn whose prompt_pushback is a correction, failure_report,
    rejection, or clarification points back at the prior assistant turn.

    Returns one row per flag: idx of the assistant turn, session_id, turn,
    flag_type, the matched text, and a short snippet. These idx values are the
    natural target_indices for run_pipeline.
    """
    _require_pandas()
    out = []
    conv = conversational(df)
    for sid, grp in conv.groupby("session_id", sort=False):
        grp = grp.sort_values("turn", kind="stable")
        last_assistant = None
        for _, row in grp.iterrows():
            if row["role"] == "assistant":
                last_assistant = row
                text = str(row["content"])
                for flag_type, rx in (("rejection", _REJECTION_RE),
                                      ("correction", _CORRECTION_RE)):
                    m = rx.search(text)
                    if m:
                        a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
                        out.append({
                            "idx": int(row["idx"]), "session_id": sid,
                            "turn": int(row["turn"]), "flag_type": flag_type,
                            "match": m.group(0),
                            "snippet": text[a:b].replace("\n", " "),
                        })
            elif row["role"] == "user":
                pb = row["prompt_pushback"]
                if pb in PUSHBACK_FLAGS and last_assistant is not None:
                    out.append({
                        "idx": int(last_assistant["idx"]), "session_id": sid,
                        "turn": int(last_assistant["turn"]),
                        "flag_type": f"pushback:{pb}", "match": str(pb),
                        "snippet": str(row["content"])[:80].replace("\n", " "),
                    })
    cols = ["idx", "session_id", "turn", "flag_type", "match", "snippet"]
    return pd.DataFrame.from_records(out, columns=cols)


class SemanticIndex:
    """Sentence transformer index for similarity search over turn content.

    Lazy: the model loads on first build. Embeddings are normalized so a dot
    product is cosine similarity. Used to find turns near a query in meaning, not
    just by literal keyword.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAVE_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is required for SemanticIndex. "
                               "Run: pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        self.idx: Optional[np.ndarray] = None
        self.emb: Optional[np.ndarray] = None

    def build(self, df: "pd.DataFrame") -> "SemanticIndex":
        _require_pandas()
        texts = df["content"].astype(str).tolist()
        self.idx = df["idx"].to_numpy()
        self.emb = self.model.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)
        return self

    def search(self, df: "pd.DataFrame", query: str, k: int = 5) -> "pd.DataFrame":
        if self.emb is None:
            raise RuntimeError("Call build(df) before search.")
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        order = np.argsort(-scores)[:k]
        hits = self.idx[order]
        out = view(df, hits, width=200).copy()
        score_map = {int(self.idx[i]): float(scores[i]) for i in order}
        out["score"] = out["idx"].map(score_map).round(3)
        return out.sort_values("score", ascending=False)


# ----------------------------------------------------------------------------
# Part 2: the four layer architecture
# ----------------------------------------------------------------------------

@dataclass
class Models:
    """Model id per layer. Lighter model for context, strong models for the audit
    and the verdict. Override any field to trade cost for rigor."""
    base: str = "claude-sonnet-4-6"
    context: str = "claude-sonnet-4-6"
    nitpicker: str = "claude-opus-4-8"
    judge: str = "claude-opus-4-8"


@dataclass
class Layer0Result:
    original_prompt: str
    core_objective: str


@dataclass
class Layer1Result:
    state_summary: str
    expected_action: str


@dataclass
class Finding:
    type: str
    evidence: str
    explanation: str


@dataclass
class NitpickResult:
    passed: bool
    severity: str
    findings: list = field(default_factory=list)


@dataclass
class JudgeResult:
    error_category: str
    bad_claude_feedback: str
    correction: str


@dataclass
class EvalRecord:
    idx: int
    session_id: str
    turn: int
    layer0: Layer0Result
    layer1: Layer1Result
    nitpick: NitpickResult
    judge: Optional[JudgeResult] = None


LAYER0_SYSTEM = (
    "You reconstruct the base task from a Claude Code session log. Given the "
    "conversation, find the original user request that started the thread and "
    "distill the single core objective in one sentence. Do not judge quality. "
    "Only restate intent.\n"
    "Return JSON only: {\"original_prompt\": str, \"core_objective\": str}"
)

LAYER1_SYSTEM = (
    "You track conversational state for an audit. Given the prior turns before a "
    "target assistant turn, including tool calls and tool results, summarize the "
    "running state: facts established, files and commands touched, constraints set, "
    "instructions still in force, and any earlier correction. Then state what the "
    "assistant should do on the target turn given that history.\n"
    "Return JSON only: {\"state_summary\": str, \"expected_action\": str}"
)

NITPICKER_SYSTEM = (
    "You are an expert nitpicker auditing one assistant turn from a Claude Code "
    "session. You are strict and skeptical. You are given the base task (Layer 0), "
    "the context and expected action (Layer 1), and the assistant turn under audit. "
    "Decide whether the turn fulfilled the objective and the expected action.\n\n"
    "Hunt specifically for:\n"
    "- hallucination: invented facts, apis, files, citations, numbers, or results.\n"
    "- lazy_code: stubs, placeholders, omitted logic, 'rest stays the same', "
    "unrun assertions, code that cannot work as written.\n"
    "- false_rejection: refusing or hedging on a benign in scope request.\n"
    "- context_loss: ignoring a constraint, fact, file, or correction set earlier.\n\n"
    "Cite exact evidence quoted from the turn. If the turn is sound, set passed "
    "true with an empty findings list.\n"
    "Return JSON only: {\"passed\": bool, \"severity\": "
    "\"none\"|\"minor\"|\"major\"|\"critical\", \"findings\": "
    "[{\"type\": \"hallucination\"|\"lazy_code\"|\"false_rejection\"|"
    "\"context_loss\"|\"other\", \"evidence\": str, \"explanation\": str}]}"
)

JUDGE_SYSTEM = (
    "You are the judge and corrector. The nitpicker flagged a flaw. Deliver the "
    "verdict. Be specific and actionable, and tie the correction to the base "
    "objective and the expected action.\n"
    "Return JSON only: {\"error_category\": \"False Rejection\"|\"Context Loss\"|"
    "\"Hallucination\"|\"Lazy Code\"|\"Instruction Drift\"|\"Other\", "
    "\"bad_claude_feedback\": str that starts with 'BAD CLAUDE! ', "
    "\"correction\": str giving the precise turn or behavior the assistant should "
    "have produced}"
)


def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ. Stdlib only, no
    python-dotenv dependency. An existing environment value always wins, so an
    exported key overrides the file."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("'").strip('"'))


def _client():
    if not HAVE_ANTHROPIC:
        raise RuntimeError("anthropic is required to run the pipeline. "
                           "Run: pip install anthropic")
    _load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env or the environment.")
    if len(key) < 20:
        raise RuntimeError("ANTHROPIC_API_KEY in .env looks like a placeholder "
                           f"({len(key)} chars). Put your real key there.")
    return anthropic.Anthropic(api_key=key)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply, tolerant of code fences
    and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _call(client, system: str, user: str, model: str, max_tokens: int = 1024) -> dict:
    """One Claude call returning parsed JSON. Not invoked on import."""
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _extract_json(text)


def conversation_for(df: "pd.DataFrame", target_idx: int) -> "pd.DataFrame":
    """All prior turns up to and including target_idx in the same session, in
    turn order. Tool turns are kept so the audit sees what actually happened."""
    _require_pandas()
    row = df[df["idx"].eq(target_idx)].iloc[0]
    conv = df[df["session_id"].eq(row["session_id"]) & (df["idx"] <= target_idx)]
    return conv.sort_values("turn", kind="stable")


def _render_turn(r, width: int) -> str:
    """One transcript line. Conversational turns show content; tool and metadata
    turns collapse to a tag plus a short slice so the prompt stays small."""
    body = str(r["content"]).replace("\n", " ")
    if r["is_conversational"]:
        return f"[{r['idx']}] {r['role']}: {body[:width]}"
    tag = r["turn_type"] or r["role"]
    name = f" {r['tool_name']}" if r["tool_name"] else ""
    return f"[{r['idx']}] {tag}{name}: {body[:160]}"


def _render_conv(conv: "pd.DataFrame", target_idx: int, width: int = 1200) -> str:
    lines = []
    for _, r in conv.iterrows():
        marker = "  <-- TARGET TURN UNDER AUDIT" if r["idx"] == target_idx else ""
        lines.append(_render_turn(r, width) + marker)
    return "\n".join(lines)


def layer0_base_task(client, df, target_idx, models: Models) -> Layer0Result:
    """Layer 0: restate the original prompt and the core objective."""
    conv = conversation_for(df, target_idx)
    user = "Conversation:\n" + _render_conv(conv, target_idx)
    d = _call(client, LAYER0_SYSTEM, user, models.base)
    return Layer0Result(d.get("original_prompt", ""), d.get("core_objective", ""))


def layer1_context(client, df, target_idx, models: Models) -> Layer1Result:
    """Layer 1: summarize state and state the expected action for the target."""
    conv = conversation_for(df, target_idx)
    prior = conv[conv["idx"] < target_idx]
    user = ("Prior turns:\n" + _render_conv(prior, -1)
            + f"\n\nThe target turn to plan for is index {target_idx}.")
    d = _call(client, LAYER1_SYSTEM, user, models.context)
    return Layer1Result(d.get("state_summary", ""), d.get("expected_action", ""))


def layer2_nitpicker(client, df, target_idx, l0: Layer0Result, l1: Layer1Result,
                     models: Models) -> NitpickResult:
    """Layer 2: strict audit of the target turn against L0 and L1."""
    response = str(df[df["idx"].eq(target_idx)].iloc[0]["content"])
    user = (
        f"Layer 0 base task:\n  original_prompt: {l0.original_prompt}\n"
        f"  core_objective: {l0.core_objective}\n\n"
        f"Layer 1 context:\n  state_summary: {l1.state_summary}\n"
        f"  expected_action: {l1.expected_action}\n\n"
        f"Assistant turn under audit (index {target_idx}):\n{response}"
    )
    d = _call(client, NITPICKER_SYSTEM, user, models.nitpicker, max_tokens=1500)
    findings = [Finding(f.get("type", "other"), f.get("evidence", ""),
                        f.get("explanation", "")) for f in d.get("findings", [])]
    return NitpickResult(bool(d.get("passed", True)), d.get("severity", "none"), findings)


def layerJ_judge(client, df, target_idx, l0: Layer0Result, l1: Layer1Result,
                 nit: NitpickResult, models: Models) -> JudgeResult:
    """Layer J: the verdict. Only call when the nitpicker found a flaw."""
    response = str(df[df["idx"].eq(target_idx)].iloc[0]["content"])
    findings = "\n".join(f"- {f.type}: {f.evidence} ({f.explanation})"
                         for f in nit.findings)
    user = (
        f"Base objective: {l0.core_objective}\n"
        f"Expected action: {l1.expected_action}\n"
        f"Severity: {nit.severity}\n"
        f"Nitpicker findings:\n{findings}\n\n"
        f"Assistant turn (index {target_idx}):\n{response}"
    )
    d = _call(client, JUDGE_SYSTEM, user, models.judge, max_tokens=1024)
    return JudgeResult(d.get("error_category", "Other"),
                       d.get("bad_claude_feedback", ""),
                       d.get("correction", ""))


def run_pipeline(df: "pd.DataFrame", target_indices=None, models: Optional[Models] = None,
                 verbose: bool = True) -> list:
    """Drive all four layers over conversational assistant turns.

    target_indices limits the audit, for example to the indices flag_signals
    returned. When None, every conversational assistant turn is audited. Layer J
    runs only when Layer 2 reports a flaw. Returns a list of EvalRecord. This
    spends tokens, so it is never called on import.
    """
    _require_pandas()
    models = models or Models()
    client = _client()

    if target_indices is None:
        sel = df[df["is_conversational"] & df["role"].eq("assistant")]
        target_indices = sel["idx"].tolist()

    records = []
    for idx in target_indices:
        idx = int(idx)
        row = df[df["idx"].eq(idx)].iloc[0]
        l0 = layer0_base_task(client, df, idx, models)
        l1 = layer1_context(client, df, idx, models)
        nit = layer2_nitpicker(client, df, idx, l0, l1, models)
        judge = None
        if not nit.passed:
            judge = layerJ_judge(client, df, idx, l0, l1, nit, models)
        records.append(EvalRecord(idx, row["session_id"], int(row["turn"]),
                                  l0, l1, nit, judge))
        if verbose:
            status = "PASS" if nit.passed else f"FLAW [{judge.error_category}]"
            print(f"  index {idx}: {status}", file=sys.stderr)
    return records


# ----------------------------------------------------------------------------
# Part 3: actionable output
# ----------------------------------------------------------------------------

def _md_escape(text: str, limit: int = 220) -> str:
    text = (text or "").replace("\n", " ").replace("|", "\\|").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def summary_table(records: list) -> str:
    """Markdown table of flagged turns: index, error category, judge feedback.

    Only failed records appear. Compact by design so it can be read without
    flooding the context window.
    """
    flagged = [r for r in records if not r.nitpick.passed and r.judge is not None]
    header = ("| Index | Error Category | Layer J Feedback |\n"
              "| --- | --- | --- |")
    if not flagged:
        return (header + f"\n| _none_ | _none_ | No flaws found across "
                f"{len(records)} audited turns. |")
    rows = [
        f"| {r.idx} | {_md_escape(r.judge.error_category, 40)} "
        f"| {_md_escape(r.judge.bad_claude_feedback)} |"
        for r in flagged
    ]
    counts: dict = {}
    for r in flagged:
        counts[r.judge.error_category] = counts.get(r.judge.error_category, 0) + 1
    tally = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    footer = f"\n\n**{len(flagged)} flagged of {len(records)} audited.** {tally}"
    return header + "\n" + "\n".join(rows) + footer


def save_report(markdown: str, path: str = "nitpicker_report.md") -> str:
    with open(path, "w") as fh:
        fh.write(markdown)
    return path


# ----------------------------------------------------------------------------
# CLI: data processing only by default. The LLM loop runs only with --run.
# ----------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Expert Nitpicker log evaluator.")
    p.add_argument("--config", default="conversations", help="SWE-chat config name")
    p.add_argument("--split", default="train", help="dataset split")
    p.add_argument("--limit", type=int, default=None, help="cap rows for a quick look")
    p.add_argument("--flags", action="store_true", help="show regex flag table and exit")
    p.add_argument("--search", help="regex to search content")
    p.add_argument("--run", action="store_true",
                   help="run the four layer LLM pipeline (spends tokens)")
    p.add_argument("--only-flagged", action="store_true",
                   help="with --run, audit only the flagged indices")
    p.add_argument("--report", default="nitpicker_report.md", help="output report path")
    args = p.parse_args(argv)

    df = load_swe_chat(config=args.config, split=args.split, limit=args.limit)
    n_conv = int(df["is_conversational"].sum())
    print(f"Loaded {len(df)} turns ({n_conv} conversational) across "
          f"{df['session_id'].nunique()} sessions.", file=sys.stderr)

    if args.search:
        print(search_regex(df, args.search).to_string(index=False))
        return

    flags = flag_signals(df)
    if args.flags or not args.run:
        print(flags.to_string(index=False) if len(flags) else "No flags.")
        if not args.run:
            return

    targets = flags["idx"].drop_duplicates().tolist() if args.only_flagged else None
    records = run_pipeline(df, target_indices=targets)
    md = summary_table(records)
    path = save_report(md, args.report)
    print(f"\nReport written to {path}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
