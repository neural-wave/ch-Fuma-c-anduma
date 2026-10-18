"""Small local LLM (e.g. Qwen3.5-2B) as a *selective, structured* privacy detector.

The model never rewrites text. It returns a JSON list of {"text", "type"} candidates for a short
passage; Python then keeps only items that occur verbatim in that passage (hallucinated or
paraphrased items are rejected and counted) and the normal evidence filter + clinical guard decide
what is finally replaced. Runs fully locally (weights from the Hugging Face cache, CPU or GPU).
"""
from __future__ import annotations

import json
import os
import re
import time

from .ner import resolve_model
from .spans import Span

TYPES = {"PERSON", "LOCATION", "ORGANIZATION", "OCCUPATION", "ID", "CONTACT"}

SYSTEM = (
    "You are a de-identification assistant for doctor-patient conversation transcripts. "
    "List every personal identifier in the passage. Return ONLY a JSON array; each element is "
    '{"text": "<exact substring copied from the passage>", "type": "<TYPE>"} with TYPE one of '
    "PERSON (names of people or pets: patient, relatives, friends, coworkers, clinicians), LOCATION (cities, "
    "regions, countries, addresses), ORGANIZATION (employers, companies, schools, hospitals), OCCUPATION "
    "(the patient's or a relative's job), ID (record, member, account numbers, also spoken digit by digit), "
    "CONTACT (phone numbers, e-mails). Never list diseases, symptoms, drugs, doses, body parts, lab values, "
    "generic roles (doctor, nurse, wife, son), dates or ages. Return [] if there is none.")

FEW_SHOT = [
    ("[patient] my brother , tom , drove me . i still take the lyrica for my parkinson's disease .",
     '[{"text": "tom", "type": "PERSON"}]'),
    ("[patient] honestly hope , my neighbor , works as a welder at boeing in tacoma .",
     '[{"text": "hope", "type": "PERSON"}, {"text": "welder", "type": "OCCUPATION"}, '
     '{"text": "boeing", "type": "ORGANIZATION"}, {"text": "tacoma", "type": "LOCATION"}]'),
    ("[doctor] no chest pain , and your a1c is seven point two .", "[]"),
]


SCHEMA_TYPES = sorted(TYPES) + ["MEDICAL"]  # MEDICAL: an outlet so drugs/diseases are not forced into a PII type


class LLMDetector:
    """backend='llama_cpp' (default; 4-bit GGUF, grammar-constrained JSON) or 'transformers'."""

    def __init__(self, model: str = "unsloth/Qwen3.5-2B-GGUF", gguf_file: str = "Qwen3.5-2B-Q4_K_M.gguf",
                 backend: str = "llama_cpp", prompt: str = "few-shot", max_new_tokens: int = 128,
                 device: str = "cpu", dtype: str = "bfloat16", allow_download: bool = True, max_chars: int = 900,
                 threshold: float | None = None, label_map: str | None = None, threads: int | None = None,
                 medical_outlet: bool = True):
        self.model_name, self.prompt, self.max_new_tokens, self.max_chars = model, prompt, max_new_tokens, max_chars
        self.backend, self.device, self.medical_outlet = backend, device, medical_outlet
        self.local_path, _ = resolve_model(model, allow_download)
        self.revision = os.path.basename(self.local_path.rstrip("/\\"))
        if backend == "llama_cpp":
            from llama_cpp import Llama, LlamaGrammar

            self.model = Llama(model_path=os.path.join(self.local_path, gguf_file), n_ctx=2048,
                               n_threads=threads or max(1, (os.cpu_count() or 2) - 2), verbose=False)
            types = SCHEMA_TYPES if medical_outlet else sorted(TYPES)
            schema = {"type": "object", "properties": {"identifiers": {"type": "array", "items": {
                "type": "object", "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": types}},
                "required": ["text", "type"]}}}, "required": ["identifiers"]}
            self.grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

            if threads:
                torch.set_num_threads(threads)
            self.tok = AutoTokenizer.from_pretrained(self.local_path)
            kw = {"dtype": getattr(torch, dtype)}
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.local_path, **kw)
            except Exception:  # noqa: BLE001 - multimodal checkpoints (Qwen3.5) load via the image-text class
                self.model = AutoModelForImageTextToText.from_pretrained(self.local_path, **kw)
            self.model.to(device).eval()
        self.stats = {"calls": 0, "json_errors": 0, "items": 0, "rejected_not_in_passage": 0,
                      "rejected_bad_type": 0, "medical_outlet": 0, "accepted": 0, "seconds": 0.0,
                      "generated_tokens": 0}

    def weights_sha256(self) -> str:
        return "n/a (LLM)"

    # ------------------------------------------------------------------ prompting
    def _messages(self, passage: str) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM}]
        if self.prompt == "few-shot":
            for q, a in FEW_SHOT:
                msgs += [{"role": "user", "content": f"Passage:\n{q}"}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": f"Passage:\n{passage}"})
        return msgs

    def _chatml(self, passage: str) -> str:
        """Qwen ChatML with thinking disabled (empty think block), JSON-object output convention."""
        sys_msg = SYSTEM.replace("Return ONLY a JSON array;", 'Return ONLY a JSON object {"identifiers": [...]};')
        sys_msg = sys_msg.replace("Return [] if there is none.", 'Return {"identifiers": []} if there is none.')
        if self.medical_outlet:
            sys_msg += " If you are unsure whether a word is a drug, disease or other medical term, label it MEDICAL."
        s = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        if self.prompt == "few-shot":
            for q, a in FEW_SHOT:
                s += f"<|im_start|>user\nPassage:\n{q}<|im_end|>\n<|im_start|>assistant\n{{\"identifiers\": {a}}}<|im_end|>\n"
        return s + f"<|im_start|>user\nPassage:\n{passage}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _generate(self, passage: str) -> str:
        if self.backend == "llama_cpp":
            t0 = time.perf_counter()
            out = self.model(self._chatml(passage), max_tokens=self.max_new_tokens, temperature=0.0, grammar=self.grammar)
            self.stats["seconds"] += time.perf_counter() - t0
            self.stats["generated_tokens"] += out["usage"]["completion_tokens"]
            return out["choices"][0]["text"]
        import torch

        msgs = self._messages(passage)
        try:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                               enable_thinking=False, return_dict=True)
        except TypeError:
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, ids["input_ids"].shape[1]:]
        self.stats["seconds"] += time.perf_counter() - t0
        self.stats["generated_tokens"] += int(new.shape[0])
        return self.tok.decode(new, skip_special_tokens=True)

    @staticmethod
    def _parse(raw: str):
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("identifiers")
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.S)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
            return data if isinstance(data, list) else None

    # ------------------------------------------------------------------ detection
    def detect(self, text: str, field: str, regions: list[tuple[int, int]] | None = None) -> list[Span]:
        from .ner import _windows

        if regions is None:
            regions = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", text)]
        out, seen = [], set()
        for r0, r1 in regions:
            for s0, e0 in _windows(text, r0, r1, self.max_chars):
                passage = text[s0:e0]
                self.stats["calls"] += 1
                items = self._parse(self._generate(passage))
                if items is None:
                    self.stats["json_errors"] += 1
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    self.stats["items"] += 1
                    surf, typ = str(it.get("text", "")).strip(), str(it.get("type", "")).upper()
                    if typ == "MEDICAL":
                        self.stats["medical_outlet"] += 1
                        continue
                    if typ not in TYPES:
                        self.stats["rejected_bad_type"] += 1
                        continue
                    hits = [m for m in re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", passage)] if surf else []
                    if not hits and surf:
                        hits = list(re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", passage, re.I))
                    if not hits:
                        self.stats["rejected_not_in_passage"] += 1  # hallucinated / paraphrased
                        continue
                    for m in hits:
                        key = (s0 + m.start(), s0 + m.end(), typ)
                        if key in seen:
                            continue
                        seen.add(key)
                        self.stats["accepted"] += 1
                        out.append(Span(field, key[0], key[1], typ, "ml:llm", 0.7, m.group(0),
                                        reasons=[f"local LLM {os.path.basename(self.model_name)} listed it as {typ}"]))
        return out
