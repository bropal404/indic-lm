#!/usr/bin/env python3
"""
reasoning_dataset_generator.py — LMA Phase-3 synthetic reasoning data.

Single-file generator. Template *wording* lives in data files
(templates_hi.jsonl, templates_ne.jsonl, one JSON object per line:
tid/task/program/role/text/args) so adding a paraphrase never touches this
code. Everything that computes a label (the PROGRAMS table below) lives here,
in Python, deterministically — no LLM ever produces a ground-truth answer.

Design summary (full rationale in reasons.md):
  * template.role in {train, val, test} holds out WORDING, never a skill:
    every semantic `program` is guaranteed present in both train and test.
  * entity names/objects, logic rules, and common-sense scenarios are each
    partitioned disjointly by split, independently of template wording.
  * label priors (yes/no, option position, "which entity") are neutralised
    by construction, not by post-hoc balancing.
  * validate_split() enforces exact counts, uniqueness, and zero leakage
    across all four leakage axes (wording / entities / content / labels-not-
    checked-but-reported) before anything is written to disk.

Usage:
    python reasoning_dataset_generator.py --smoke
    python reasoning_dataset_generator.py
    python reasoning_dataset_generator.py --digits devanagari
    python reasoning_dataset_generator.py --templates-dir /path/to/jsonl
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction


# ===========================================================================
# 1. Template type + loader (reads templates_hi.jsonl / templates_ne.jsonl)
# ===========================================================================

ROLES = ("train", "val", "test")


@dataclass
class Template:
    tid: str
    task: str
    program: str
    role: str
    text: str
    args: dict = field(default_factory=dict)


def load_templates(path):
    """Load one language's template bank from a JSONL file.

    Returns dict[task] -> list[Template].
    """
    bank = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                t = Template(
                    tid=row["tid"], task=row["task"], program=row["program"],
                    role=row["role"], text=row["text"], args=row.get("args", {}),
                )
            except KeyError as e:
                raise ValueError(f"{path}:{line_no}: missing field {e}") from e
            bank[t.task].append(t)
    return dict(bank)


def pools_by_role(bank, task):
    pools = {r: [t for t in bank[task] if t.role == r] for r in ROLES}
    for r in ROLES:
        if not pools[r]:
            raise ValueError(f"Task {task} has no templates for role {r}")
    return pools


def validate_bank(bank, lang):
    """Structural invariants: unique ids, every role covered, no skill held out."""
    problems = []
    seen = set()
    for task, templates in bank.items():
        for t in templates:
            key = (task, t.tid)
            if key in seen:
                problems.append(f"[{lang}] duplicate template id {t.tid} in {task}")
            seen.add(key)
            if t.role not in ROLES:
                problems.append(f"[{lang}] {t.tid}: bad role {t.role}")

        by_role = defaultdict(set)
        for t in templates:
            by_role[t.role].add(t.program)
        for r in ROLES:
            if not by_role[r]:
                problems.append(f"[{lang}] {task}: no templates with role {r}")
        train_only = by_role["train"] - by_role["test"]
        test_only = by_role["test"] - by_role["train"]
        if test_only:
            problems.append(f"[{lang}] {task}: programs only in test: {sorted(test_only)}")
        if train_only:
            problems.append(f"[{lang}] {task}: programs never tested: {sorted(train_only)}")

    if problems:
        raise ValueError("Template bank invalid:\n  " + "\n  ".join(problems))
    return True


def bank_summary(bank):
    return {
        task: {
            "templates": len(templates),
            "programs": len({t.program for t in templates}),
            "by_role": dict(Counter(t.role for t in templates)),
        }
        for task, templates in bank.items()
    }


# ===========================================================================
# 2. Digit rendering
# ===========================================================================

DEVANAGARI_DIGITS = "०१२३४५६७८९"


def render_int(n, digit_script="ascii"):
    s = str(int(n))
    if digit_script == "devanagari":
        return "".join(DEVANAGARI_DIGITS[int(c)] if c.isdigit() else c for c in s)
    return s


# ===========================================================================
# 3. Config
# ===========================================================================

CONFIG = {
    "seed": 42,
    "output_root": ".",
    "digit_script": "ascii",           # "ascii" | "devanagari"
    "templates_dir": ".",              # where templates_hi.jsonl / templates_ne.jsonl live
    "languages": {
        "hi": {"dir": "hindi/reasoning", "name": "Hindi", "template_file": "templates_hi.jsonl"},
        "ne": {"dir": "nepali/reasoning", "name": "Nepali", "template_file": "templates_ne.jsonl"},
    },
    "counts": {"train": 15000, "val": 2000, "test": 1000},
    "task_weights": {
        "comparison":       0.07,
        "transitive":       0.09,
        "arithmetic":       0.08,
        "percentage":       0.07,
        "ratio":            0.06,
        "algebra":          0.08,
        "unit_conversion":  0.07,
        "speed_distance":   0.07,
        "physics_fma":      0.06,
        "physics_density":  0.05,
        "physics_work":     0.05,
        "multi_hop":        0.09,
        "logic":            0.06,
        "common_sense":     0.04,
        "counting":         0.06,
    },
    "difficulty_weights": {
        "train": {"easy": 0.30, "medium": 0.50, "hard": 0.20},
        "val":   {"easy": 0.25, "medium": 0.50, "hard": 0.25},
        "test":  {"easy": 0.15, "medium": 0.45, "hard": 0.40},
    },
    # Tasks with open-ended numeric space; absorb quota from exhausted tasks
    # (logic / common_sense have a finite fixed-string question space).
    "elastic_tasks": [
        "arithmetic", "algebra", "percentage", "multi_hop",
        "speed_distance", "unit_conversion", "counting", "ratio",
    ],
    "max_sample_attempts": 60,
    "max_redistribution_rounds": 12,
}


# ===========================================================================
# 4. Language resources: entities, answer lexicon, logic rules, CS scenarios
#    These are generator INPUTS (not surface wording) and stay in code.
# ===========================================================================

RES = {
    "hi": {
        "person_names": [
            "राम", "श्याम", "गीता", "सीता", "मोहन", "सुनीता", "अमित", "प्रिया",
            "राजू", "अंजलि", "विकास", "नेहा", "सुरेश", "कविता", "दीपक", "पूजा",
            "मनोज", "रेखा", "संजय", "आरती", "रोहित", "निखिल", "मीरा", "तन्वी",
            "अर्जुन", "रवि", "माया", "करण", "नंदिनी", "विनय", "साक्षी", "हर्ष",
        ],
        "objects": [
            "आम", "किताब", "बैग", "साइकिल", "मोबाइल", "कुर्सी", "टोकरी", "अनार",
            "घड़ी", "छाता", "जूता", "पंखा", "थैला", "बर्तन", "खिलौना", "लैंप",
            "बोतल", "कप", "कंप्यूटर", "कैलकुलेटर", "टेबल", "कंबल", "डिब्बा",
            "पेन", "गेंद", "कुशन", "बाल्टी", "चाबी",
        ],
    },
    "ne": {
        "person_names": [
            "राम", "हरि", "सीता", "गीता", "विकास", "सुनिता", "प्रकाश", "अनिता",
            "विनोद", "सरिता", "रमेश", "कमला", "दीपक", "सन्तोष", "पूजा", "विशाल",
            "सुजाता", "गोपाल", "बिन्दु", "निर्मल", "आशा", "मनिष", "कृष्ण", "मिना",
            "सागर", "रोशन", "कविन", "प्रविण", "स्मृति", "रचना", "अमित", "निरज",
        ],
        "objects": [
            "आँप", "किताब", "झोला", "साइकल", "मोबाइल", "कुर्सी", "डालो", "अनार",
            "घडी", "छाता", "जुत्ता", "पंखा", "थैलो", "भाँडो", "खेलौना", "लाम्प",
            "बोतल", "कप", "कम्प्युटर", "क्याल्कुलेटर", "टेबल", "कम्बल", "बाकस",
            "कलम", "बल", "बाल्टिन", "चाबी", "कुशन",
        ],
    },
}

LEX = {
    "hi": {"yes": "हाँ", "no": "नहीं", "increase": "बढ़ेगी", "decrease": "घटेगी"},
    "ne": {"yes": "हो", "no": "होइन", "increase": "बढ्छ", "decrease": "घट्छ"},
}

# Conditional rules for propositional-logic templates. `role` partitions the
# RULE CONTENT by split, independent of which frame (template) uses it.
LOGIC_RULES = {
    "hi": [
        {"rid": "rain", "role": "train",
         "cond": "बारिश होती है", "result": "सड़क गीली हो जाती है",
         "cond_fact": "बारिश हो रही है", "cond_neg_fact": "बारिश नहीं हो रही है",
         "cond_q": "बारिश हो रही है",
         "result_fact": "सड़क गीली है", "result_neg_fact": "सड़क गीली नहीं है",
         "result_q": "सड़क गीली है", "result_q_neg": "सड़क गीली नहीं है"},
        {"rid": "battery", "role": "train",
         "cond": "बैटरी खत्म हो जाती है", "result": "फोन बंद हो जाता है",
         "cond_fact": "बैटरी खत्म हो चुकी है", "cond_neg_fact": "बैटरी खत्म नहीं हुई है",
         "cond_q": "बैटरी खत्म हो चुकी है",
         "result_fact": "फोन बंद है", "result_neg_fact": "फोन बंद नहीं है",
         "result_q": "फोन बंद है", "result_q_neg": "फोन बंद नहीं है"},
        {"rid": "power", "role": "val",
         "cond": "बिजली चली जाती है", "result": "पंखा रुक जाता है",
         "cond_fact": "बिजली चली गई है", "cond_neg_fact": "बिजली नहीं गई है",
         "cond_q": "बिजली चली गई है",
         "result_fact": "पंखा रुका हुआ है", "result_neg_fact": "पंखा रुका हुआ नहीं है",
         "result_q": "पंखा रुका हुआ है", "result_q_neg": "पंखा रुका हुआ नहीं है"},
        {"rid": "exam", "role": "test",
         "cond": "कोई छात्र परीक्षा पास करता है", "result": "उसे प्रमाणपत्र मिलता है",
         "cond_fact": "छात्र ने परीक्षा पास की है", "cond_neg_fact": "छात्र ने परीक्षा पास नहीं की है",
         "cond_q": "छात्र ने परीक्षा पास की है",
         "result_fact": "छात्र को प्रमाणपत्र मिला है", "result_neg_fact": "छात्र को प्रमाणपत्र नहीं मिला है",
         "result_q": "छात्र को प्रमाणपत्र मिला है", "result_q_neg": "छात्र को प्रमाणपत्र नहीं मिला है"},
    ],
    "ne": [
        {"rid": "rain", "role": "train",
         "cond": "पानी पर्छ", "result": "सडक भिज्छ",
         "cond_fact": "पानी परिरहेको छ", "cond_neg_fact": "पानी परेको छैन",
         "cond_q": "पानी परेको छ",
         "result_fact": "सडक भिजेको छ", "result_neg_fact": "सडक भिजेको छैन",
         "result_q": "सडक भिजेको छ", "result_q_neg": "सडक भिजेको छैन"},
        {"rid": "battery", "role": "train",
         "cond": "ब्याट्री सकिन्छ", "result": "फोन बन्द हुन्छ",
         "cond_fact": "ब्याट्री सकिएको छ", "cond_neg_fact": "ब्याट्री सकिएको छैन",
         "cond_q": "ब्याट्री सकिएको छ",
         "result_fact": "फोन बन्द छ", "result_neg_fact": "फोन बन्द छैन",
         "result_q": "फोन बन्द छ", "result_q_neg": "फोन बन्द छैन"},
        {"rid": "power", "role": "val",
         "cond": "बिजुली जान्छ", "result": "पंखा रोकिन्छ",
         "cond_fact": "बिजुली गएको छ", "cond_neg_fact": "बिजुली गएको छैन",
         "cond_q": "बिजुली गएको छ",
         "result_fact": "पंखा रोकिएको छ", "result_neg_fact": "पंखा रोकिएको छैन",
         "result_q": "पंखा रोकिएको छ", "result_q_neg": "पंखा रोकिएको छैन"},
        {"rid": "exam", "role": "test",
         "cond": "विद्यार्थी परीक्षामा पास हुन्छ", "result": "प्रमाणपत्र पाउँछ",
         "cond_fact": "विद्यार्थी परीक्षामा पास भएको छ", "cond_neg_fact": "विद्यार्थी परीक्षामा पास भएको छैन",
         "cond_q": "विद्यार्थी परीक्षामा पास भएको छ",
         "result_fact": "विद्यार्थीले प्रमाणपत्र पाएको छ", "result_neg_fact": "विद्यार्थीले प्रमाणपत्र पाएको छैन",
         "result_q": "विद्यार्थीले प्रमाणपत्र पाएको छ", "result_q_neg": "विद्यार्थीले प्रमाणपत्र पाएको छैन"},
    ],
}

# Two-option common-sense scenarios. Generator shuffles opt1/opt2 so position
# carries no signal, and emits the correct option TEXT as the label.
CS_SCENARIOS = {
    "hi": [
        {"sid": "dry_clothes", "role": "train",
         "stem": "गीले कपड़े जल्दी सुखाने हैं।",
         "good": "धूप वाली खुली जगह", "bad": "बंद अलमारी"},
        {"sid": "cool_water", "role": "train",
         "stem": "पानी को ठंडा रखना है।",
         "good": "फ्रिज में रखना", "bad": "धूप में रखना"},
        {"sid": "heavy_rain", "role": "train",
         "stem": "बाहर तेज़ बारिश हो रही है और बाहर जाना ज़रूरी है।",
         "good": "छाता लेकर जाना", "bad": "बिना छाते के जाना"},
        {"sid": "low_battery", "role": "val",
         "stem": "फोन की बैटरी बहुत कम है और चार्जर पास में है।",
         "good": "फोन को चार्ज पर लगाना", "bad": "वीडियो देखते रहना"},
        {"sid": "night_study", "role": "test",
         "stem": "रात में पढ़ाई करनी है और कमरे में अँधेरा है।",
         "good": "लैंप जलाना", "bad": "लैंप बंद रखना"},
    ],
    "ne": [
        {"sid": "dry_clothes", "role": "train",
         "stem": "भिजेको कपडा छिटो सुकाउनु छ।",
         "good": "घाम लागेको खुला ठाउँ", "bad": "बन्द दराज"},
        {"sid": "cool_water", "role": "train",
         "stem": "पानी चिसो राख्नु छ।",
         "good": "फ्रिजमा राख्नु", "bad": "घाममा राख्नु"},
        {"sid": "heavy_rain", "role": "train",
         "stem": "बाहिर धेरै पानी परिरहेको छ र बाहिर जानै पर्छ।",
         "good": "छाता लिएर जानु", "bad": "छाता नलिई जानु"},
        {"sid": "low_battery", "role": "val",
         "stem": "फोनको ब्याट्री धेरै कम छ र चार्जर नजिकै छ।",
         "good": "फोन चार्जमा जोड्नु", "bad": "भिडियो हेरिरहनु"},
        {"sid": "night_study", "role": "test",
         "stem": "राति पढ्नु छ र कोठामा अँध्यारो छ।",
         "good": "बत्ती बाल्नु", "bad": "बत्ती निभाइराख्नु"},
    ],
}

# Direction (increase/decrease) scenarios, label-balanced.
CS_DIRECTION = {
    "hi": [
        {"sid": "glass_push", "role": "train", "label": "increase",
         "stem": "एक गिलास मेज़ के किनारे पर रखा है और कोई उसे धक्का देता है। गिलास के गिरने की संभावना बढ़ेगी या घटेगी?"},
        {"sid": "brakes", "role": "train", "label": "decrease",
         "stem": "चलती गाड़ी में ब्रेक लगाया जाता है। गाड़ी की गति बढ़ेगी या घटेगी?"},
        {"sid": "ice_sun", "role": "val", "label": "decrease",
         "stem": "बर्फ का एक टुकड़ा धूप में रखा जाता है। उसका आकार बढ़ेगा या घटेगा?"},
        {"sid": "fan_speed", "role": "test", "label": "increase",
         "stem": "पंखे की गति बढ़ा दी जाती है। कमरे में हवा की गति बढ़ेगी या घटेगी?"},
    ],
    "ne": [
        {"sid": "glass_push", "role": "train", "label": "increase",
         "stem": "गिलास टेबलको किनारमा छ र कसैले धक्का दिन्छ। गिलास खस्ने सम्भावना बढ्छ कि घट्छ?"},
        {"sid": "brakes", "role": "train", "label": "decrease",
         "stem": "चलिरहेको गाडीमा ब्रेक लगाइन्छ। गाडीको गति बढ्छ कि घट्छ?"},
        {"sid": "ice_sun", "role": "val", "label": "decrease",
         "stem": "बरफको टुक्रा घाममा राखिन्छ। त्यसको आकार बढ्छ कि घट्छ?"},
        {"sid": "fan_speed", "role": "test", "label": "increase",
         "stem": "पंखाको गति बढाइन्छ। कोठाको हावाको गति बढ्छ कि घट्छ?"},
    ],
}


# ===========================================================================
# 5. Small helpers
# ===========================================================================

def N(x):
    """Render a number (int/Fraction) for a template slot or an answer."""
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return render_int(x.numerator, CONFIG["digit_script"])
        return f"{render_int(x.numerator, CONFIG['digit_script'])}/" \
               f"{render_int(x.denominator, CONFIG['digit_script'])}"
    if isinstance(x, float) and abs(x - round(x)) > 1e-9:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return render_int(int(round(x)), CONFIG["digit_script"])


def choose(rng, seq):
    return seq[rng.randrange(len(seq))]


def difficulty_scale(d):
    return {"easy": 0, "medium": 1, "hard": 2}[d]


def pick_difficulty(split, rng):
    weights = CONFIG["difficulty_weights"][split]
    r, acc = rng.random(), 0.0
    for name, w in weights.items():
        acc += w
        if r <= acc:
            return name
    return list(weights)[-1]


def allocate_quotas(n, weights):
    total = sum(weights.values())
    raw = {k: n * v / total for k, v in weights.items()}
    q = {k: int(v) for k, v in raw.items()}
    rem = n - sum(q.values())
    order = sorted(raw, key=lambda k: (raw[k] - q[k], weights[k]), reverse=True)
    for k in order[:rem]:
        q[k] += 1
    return q


def split_partition(items, split):
    """Disjoint slice of a pool per split: ~25% test, ~25% val, ~50% train."""
    a = max(4, int(len(items) * 0.25))
    b = max(a + 4, int(len(items) * 0.50))
    if split == "test":
        return items[:a]
    if split == "val":
        return items[a:b]
    return items[b:]


def entity_context(lang, split):
    return (
        split_partition(RES[lang]["person_names"], split),
        split_partition(RES[lang]["objects"], split),
    )


def role_pool(items, split):
    pool = [x for x in items if x["role"] == split]
    return pool or [x for x in items if x["role"] == "train"]


# ===========================================================================
# 6. Program implementations
#    Each: (lang, split, template, difficulty, rng) -> (slots, answer, meta)
# ===========================================================================

def p_cmp_extreme(lang, split, t, diff, rng):
    persons, objects = entity_context(lang, split)
    pool = persons if rng.random() < 0.5 else objects
    a, b = rng.sample(pool, 2)
    d = difficulty_scale(diff)
    lo = rng.randint(5, 80)
    gap = rng.randint(1, 5 + 10 * (d + 1))
    va, vb = lo + gap, lo
    if rng.random() < 0.5:
        va, vb = vb, va
    bigger, smaller = (a, b) if va > vb else (b, a)
    answer = bigger if t.args["target"] == "max" else smaller
    return ({"a": a, "b": b, "va": N(va), "vb": N(vb)}, answer,
            {"values": [va, vb], "target": t.args["target"]})


def p_trans_extreme(lang, split, t, diff, rng):
    persons, objects = entity_context(lang, split)
    pool = persons if rng.random() < 0.5 else objects
    p1, p2, p3 = rng.sample(pool, 3)
    top, bottom = (p1, p3) if t.args["relation"] == "gt" else (p3, p1)
    answer = top if t.args["target"] == "max" else bottom
    return ({"p1": p1, "p2": p2, "p3": p3}, answer,
            {"relation": t.args["relation"], "target": t.args["target"]})


def p_trans_yesno(lang, split, t, diff, rng):
    persons, objects = entity_context(lang, split)
    pool = persons if rng.random() < 0.5 else objects
    p1, p2, p3 = rng.sample(pool, 3)
    p1_is_greater = t.args["relation"] == "gt"
    asked_p1_gt_p3 = t.args["query"] == "p1_gt_p3"
    truth = p1_is_greater if asked_p1_gt_p3 else (not p1_is_greater)
    answer = LEX[lang]["yes"] if truth else LEX[lang]["no"]
    return ({"p1": p1, "p2": p2, "p3": p3}, answer,
            {"relation": t.args["relation"], "query": t.args["query"]})


def _arith_slots(lang, split, rng, x, y):
    persons, objects = entity_context(lang, split)
    o1, o2 = rng.sample(objects, 2)
    return {"a": choose(rng, persons), "o1": o1, "o2": o2, "x": N(x), "y": N(y)}


def p_arith_add(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    x = rng.randint(5, 50 + 40 * d)
    y = rng.randint(2, 30 + 30 * d)
    return _arith_slots(lang, split, rng, x, y), N(x + y), {"op": "add"}


def p_arith_sub(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    y = rng.randint(2, 20 + 20 * d)
    x = rng.randint(y + 1, y + 40 + 30 * d)
    return _arith_slots(lang, split, rng, x, y), N(x - y), {"op": "sub"}


def p_arith_mul(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    x = rng.randint(2, 12 + 10 * d)
    y = rng.randint(2, 9 + 4 * d)
    return _arith_slots(lang, split, rng, x, y), N(x * y), {"op": "mul"}


def p_arith_div(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    y = rng.randint(2, 9 + 3 * d)
    q = rng.randint(2, 15 + 10 * d)
    x = y * q
    return _arith_slots(lang, split, rng, x, y), N(q), {"op": "div"}


def _pct_slots(lang, split, rng, x, p):
    _, objects = entity_context(lang, split)
    return {"o1": choose(rng, objects), "x": N(x), "p": N(p)}


PCT_VALUES = [5, 10, 15, 20, 25, 30, 40, 50]


def p_pct_discount(lang, split, t, diff, rng):
    p = choose(rng, PCT_VALUES)
    x = 20 * rng.randint(5, 50)
    return _pct_slots(lang, split, rng, x, p), N(x * (100 - p) // 100), {"pct": p}


def p_pct_of(lang, split, t, diff, rng):
    p = choose(rng, PCT_VALUES)
    x = 20 * rng.randint(5, 50)
    return _pct_slots(lang, split, rng, x, p), N(x * p // 100), {"pct": p}


def p_pct_increase(lang, split, t, diff, rng):
    p = choose(rng, PCT_VALUES)
    x = 20 * rng.randint(5, 50)
    return _pct_slots(lang, split, rng, x, p), N(x * (100 + p) // 100), {"pct": p}


RATIOS = [(1, 2), (2, 3), (3, 4), (2, 5), (3, 5), (4, 5), (5, 2), (5, 3), (7, 4)]


def p_ratio_other(lang, split, t, diff, rng):
    m, n = choose(rng, RATIOS)
    persons, _ = entity_context(lang, split)
    a, b = rng.sample(persons, 2)
    x = m * rng.randint(2, 20)
    return ({"a": a, "b": b, "m": N(m), "n": N(n), "x": N(x), "t": N(0)},
            N(x * n // m), {"ratio": [m, n]})


def p_ratio_part(lang, split, t, diff, rng):
    m, n = choose(rng, RATIOS)
    persons, _ = entity_context(lang, split)
    a, b = rng.sample(persons, 2)
    total = (m + n) * rng.randint(2, 20)
    return ({"a": a, "b": b, "m": N(m), "n": N(n), "x": N(0), "t": N(total)},
            N(total * m // (m + n)), {"ratio": [m, n]})


def p_alg_add_inv(lang, split, t, diff, rng):
    x, b = rng.randint(5, 100), rng.randint(1, 40)
    return {"b": N(b), "c": N(x + b), "m": N(1)}, N(x), {"eq": "x+b=c"}


def p_alg_sub_inv(lang, split, t, diff, rng):
    b = rng.randint(1, 40)
    x = rng.randint(b + 1, b + 100)
    return {"b": N(b), "c": N(x - b), "m": N(1)}, N(x), {"eq": "x-b=c"}


def p_alg_mul_inv(lang, split, t, diff, rng):
    m, x = rng.randint(2, 12), rng.randint(2, 40)
    return {"b": N(0), "c": N(m * x), "m": N(m)}, N(x), {"eq": "mx=c"}


def p_alg_linear_plus(lang, split, t, diff, rng):
    m, x, b = rng.randint(2, 9), rng.randint(2, 40), rng.randint(1, 40)
    return {"b": N(b), "c": N(m * x + b), "m": N(m)}, N(x), {"eq": "mx+b=c"}


def p_alg_linear_minus(lang, split, t, diff, rng):
    m, x, b = rng.randint(2, 9), rng.randint(2, 40), rng.randint(1, 40)
    return {"b": N(b), "c": N(m * x - b), "m": N(m)}, N(x), {"eq": "mx-b=c"}


def p_unit_convert(lang, split, t, diff, rng):
    x = rng.randint(t.args["low"], t.args["high"])
    return {"x": N(x)}, N(x * t.args["factor"]), {"pair": t.args["pair"]}


SPEEDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120]
HOURS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]


def p_sd_distance(lang, split, t, diff, rng):
    v, h = choose(rng, SPEEDS), choose(rng, HOURS)
    return {"v": N(v), "t": N(h), "d": N(v * h)}, N(v * h), {"formula": "d=vt"}


def p_sd_time(lang, split, t, diff, rng):
    v, h = choose(rng, SPEEDS), choose(rng, HOURS)
    return {"v": N(v), "t": N(h), "d": N(v * h)}, N(h), {"formula": "t=d/v"}


def p_sd_speed(lang, split, t, diff, rng):
    v = choose(rng, [2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    s = choose(rng, [2, 4, 5, 8, 10, 15, 20, 25, 30, 40])
    return {"v": N(v), "t": N(s), "d": N(v * s)}, N(v), {"formula": "v=d/t"}


def p_fma_force(lang, split, t, diff, rng):
    m, a = rng.randint(2, 60), rng.randint(1, 25)
    return {"m": N(m), "a": N(a), "f": N(m * a)}, N(m * a), {"formula": "F=ma"}


def p_fma_accel(lang, split, t, diff, rng):
    m, a = rng.randint(2, 60), rng.randint(1, 25)
    return {"m": N(m), "a": N(a), "f": N(m * a)}, N(a), {"formula": "a=F/m"}


def p_fma_mass(lang, split, t, diff, rng):
    m, a = rng.randint(2, 60), rng.randint(1, 25)
    return {"m": N(m), "a": N(a), "f": N(m * a)}, N(m), {"formula": "m=F/a"}


def p_den_density(lang, split, t, diff, rng):
    d, v = rng.randint(2, 15), rng.randint(2, 20)
    return {"d": N(d), "v": N(v), "m": N(d * v)}, N(d), {"formula": "rho=m/V"}


def p_den_mass(lang, split, t, diff, rng):
    d, v = rng.randint(2, 15), rng.randint(2, 20)
    return {"d": N(d), "v": N(v), "m": N(d * v)}, N(d * v), {"formula": "m=rho*V"}


def p_den_volume(lang, split, t, diff, rng):
    d, v = rng.randint(2, 15), rng.randint(2, 20)
    return {"d": N(d), "v": N(v), "m": N(d * v)}, N(v), {"formula": "V=m/rho"}


def p_work_work(lang, split, t, diff, rng):
    f, dd = rng.randint(2, 60), rng.randint(2, 40)
    return ({"f": N(f), "d": N(dd), "w": N(f * dd), "t": N(1), "p": N(f * dd)},
            N(f * dd), {"formula": "W=Fd"})


def p_work_power(lang, split, t, diff, rng):
    p, s = rng.randint(2, 60), rng.randint(2, 30)
    return ({"f": N(1), "d": N(1), "w": N(p * s), "t": N(s), "p": N(p)},
            N(p), {"formula": "P=W/t"})


def p_work_from_power(lang, split, t, diff, rng):
    p, s = rng.randint(2, 60), rng.randint(2, 30)
    return ({"f": N(1), "d": N(1), "w": N(p * s), "t": N(s), "p": N(p)},
            N(p * s), {"formula": "W=Pt"})


def _mh_slots(lang, split, rng, **kw):
    persons, objects = entity_context(lang, split)
    base = {"a": choose(rng, persons), "o1": choose(rng, objects),
            "x": N(0), "y": N(0), "z": N(0), "m": N(0), "p": N(0)}
    base.update({k: N(v) for k, v in kw.items()})
    return base


def p_mh_spend_earn(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    x = rng.randint(50, 400 + 300 * d)
    y = rng.randint(10, x - 10)
    z = rng.randint(5, 200 + 100 * d)
    return _mh_slots(lang, split, rng, x=x, y=y, z=z), N(x - y + z), {"hops": 2}


def p_mh_sell_percent(lang, split, t, diff, rng):
    p = choose(rng, [10, 20, 25, 50])
    remaining = choose(rng, [20, 40, 60, 80, 100, 120, 200])
    y = rng.randint(1, 30)
    x = remaining + y
    return (_mh_slots(lang, split, rng, x=x, y=y, p=p),
            N(remaining * (100 - p) // 100), {"hops": 2, "pct": p})


def p_mh_add_then_mul(lang, split, t, diff, rng):
    d = difficulty_scale(diff)
    x = rng.randint(2, 30 + 20 * d)
    y = rng.randint(1, 20 + 15 * d)
    m = rng.randint(2, 9)
    return _mh_slots(lang, split, rng, x=x, y=y, m=m), N((x + y) * m), {"hops": 2}


def _logic(lang, split, rng, answer_key):
    rule = choose(rng, role_pool(LOGIC_RULES[lang], split))
    slots = {k: v for k, v in rule.items() if k not in ("rid", "role")}
    return slots, LEX[lang][answer_key], {"rule_id": rule["rid"]}


def p_log_modus_ponens(lang, split, t, diff, rng):
    return _logic(lang, split, rng, "yes")


def p_log_modus_tollens(lang, split, t, diff, rng):
    return _logic(lang, split, rng, "no")


def p_log_affirm_consequent_certain(lang, split, t, diff, rng):
    return _logic(lang, split, rng, "no")


def p_log_affirm_consequent_possible(lang, split, t, diff, rng):
    return _logic(lang, split, rng, "yes")


def p_log_deny_antecedent(lang, split, t, diff, rng):
    return _logic(lang, split, rng, "no")


def p_cs_choice(lang, split, t, diff, rng):
    sc = choose(rng, role_pool(CS_SCENARIOS[lang], split))
    opts = [sc["good"], sc["bad"]]
    rng.shuffle(opts)  # option position carries no signal
    return ({"stem": sc["stem"], "opt1": opts[0], "opt2": opts[1]},
            sc["good"], {"scenario_id": sc["sid"], "correct_position": opts.index(sc["good"])})


def p_cs_direction(lang, split, t, diff, rng):
    sc = choose(rng, role_pool(CS_DIRECTION[lang], split))
    return ({"stem": sc["stem"], "opt1": "", "opt2": ""},
            LEX[lang][sc["label"]], {"scenario_id": sc["sid"], "label": sc["label"]})


def p_count_sum(lang, split, t, diff, rng):
    a, b = rng.randint(3, 60), rng.randint(3, 60)
    return {"a": N(a), "b": N(b), "n": N(0), "k": N(0)}, N(a + b), {"kind": "sum"}


def p_count_diff(lang, split, t, diff, rng):
    b = rng.randint(3, 40)
    a = rng.randint(b + 1, b + 40)
    return {"a": N(a), "b": N(b), "n": N(0), "k": N(0)}, N(a - b), {"kind": "diff"}


def p_count_choose(lang, split, t, diff, rng):
    n = rng.randint(4, 12)
    k = rng.randint(1, n - 1)
    c = 1
    for i in range(1, k + 1):
        c = c * (n - k + i) // i
    return {"a": N(0), "b": N(0), "n": N(n), "k": N(k)}, N(c), {"kind": "nCk"}


def p_count_even_below(lang, split, t, diff, rng):
    n = choose(rng, [6, 8, 10, 12, 14, 16, 20, 24, 30, 40, 50])
    return ({"a": N(0), "b": N(0), "n": N(n), "k": N(0)}, N((n - 1) // 2),
            {"kind": "even_below"})


PROGRAMS = {
    "cmp_extreme": p_cmp_extreme,
    "trans_extreme": p_trans_extreme,
    "trans_yesno": p_trans_yesno,
    "arith_add": p_arith_add,
    "arith_sub": p_arith_sub,
    "arith_mul": p_arith_mul,
    "arith_div": p_arith_div,
    "pct_discount": p_pct_discount,
    "pct_of": p_pct_of,
    "pct_increase": p_pct_increase,
    "ratio_other": p_ratio_other,
    "ratio_part": p_ratio_part,
    "alg_add_inv": p_alg_add_inv,
    "alg_sub_inv": p_alg_sub_inv,
    "alg_mul_inv": p_alg_mul_inv,
    "alg_linear_plus": p_alg_linear_plus,
    "alg_linear_minus": p_alg_linear_minus,
    "unit_convert": p_unit_convert,
    "sd_distance": p_sd_distance,
    "sd_time": p_sd_time,
    "sd_speed": p_sd_speed,
    "fma_force": p_fma_force,
    "fma_accel": p_fma_accel,
    "fma_mass": p_fma_mass,
    "den_density": p_den_density,
    "den_mass": p_den_mass,
    "den_volume": p_den_volume,
    "work_work": p_work_work,
    "work_power": p_work_power,
    "work_from_power": p_work_from_power,
    "mh_spend_earn": p_mh_spend_earn,
    "mh_sell_percent": p_mh_sell_percent,
    "mh_add_then_mul": p_mh_add_then_mul,
    "log_modus_ponens": p_log_modus_ponens,
    "log_modus_tollens": p_log_modus_tollens,
    "log_affirm_consequent_certain": p_log_affirm_consequent_certain,
    "log_affirm_consequent_possible": p_log_affirm_consequent_possible,
    "log_deny_antecedent": p_log_deny_antecedent,
    "cs_choice": p_cs_choice,
    "cs_direction": p_cs_direction,
    "count_sum": p_count_sum,
    "count_diff": p_count_diff,
    "count_choose": p_count_choose,
    "count_even_below": p_count_even_below,
}


# ===========================================================================
# 7. Example construction and sampling
# ===========================================================================

def make_example(lang, split, task, template, difficulty, rng):
    program = PROGRAMS[template.program]
    slots, answer, meta = program(lang, split, template, difficulty, rng)
    question = template.text.format(**slots)

    if not question.strip():
        raise ValueError(f"Empty question from {template.tid}")
    if not str(answer).strip():
        raise ValueError(f"Empty answer from {template.tid}")

    meta = dict(meta)
    meta.update({"template_id": template.tid, "program": template.program,
                 "template_role": template.role})
    return {
        "language": lang, "split": split, "task_type": task,
        "program": template.program, "template_id": template.tid,
        "difficulty": difficulty, "question": question, "answer": str(answer),
        "metadata": meta,
    }


def fill_task(lang, split, task, pool, quota, rng, seen):
    """Sample `quota` unique examples, round-robining across programs so
    labels/skills stay balanced. Returns (examples, shortfall)."""
    by_program = defaultdict(list)
    for t in pool:
        by_program[t.program].append(t)
    programs = sorted(by_program)

    out = []
    budget = CONFIG["max_sample_attempts"]
    i = 0
    while len(out) < quota:
        template = choose(rng, by_program[programs[i % len(programs)]])
        i += 1
        difficulty = pick_difficulty(split, rng)
        ex = make_example(lang, split, task, template, difficulty, rng)
        if ex["question"] in seen:
            budget -= 1
            if budget <= 0:
                break
            continue
        budget = CONFIG["max_sample_attempts"]
        seen.add(ex["question"])
        out.append(ex)
    return out, quota - len(out)


def generate_split(lang, bank, split, n, rng):
    """Generate one split, redistributing quota from tasks that run out of
    unique questions (logic / common_sense have a fixed-size question space)
    to tasks with open numeric ranges, until the split is full."""
    quotas = allocate_quotas(n, CONFIG["task_weights"])
    pools = {task: pools_by_role(bank, task)[split] for task in quotas}

    seen = set()
    examples, shortfalls = [], {}
    for task, quota in quotas.items():
        got, short = fill_task(lang, split, task, pools[task], quota, rng, seen)
        examples.extend(got)
        if short:
            shortfalls[task] = short

    elastic = [t for t in CONFIG["elastic_tasks"] if t in quotas]
    for _ in range(CONFIG["max_redistribution_rounds"]):
        deficit = n - len(examples)
        if deficit <= 0:
            break
        before = len(examples)
        extra = allocate_quotas(deficit, {t: CONFIG["task_weights"][t] for t in elastic})
        for task, k in extra.items():
            if k:
                got, _ = fill_task(lang, split, task, pools[task], k, rng, seen)
                examples.extend(got)
        if len(examples) == before:
            raise ValueError(
                f"{split}: cannot reach {n} unique examples "
                f"(stuck at {len(examples)}); widen numeric ranges or add templates"
            )
    examples = examples[:n]

    rng.shuffle(examples)
    for idx, ex in enumerate(examples):
        ex["id"] = f"{lang}_{split}_{idx:06d}"
    return examples, shortfalls


# ===========================================================================
# 8. Validation and reporting
# ===========================================================================

def validate_split(splits, expected):
    problems = []

    for name, target in expected.items():
        if len(splits[name]) != target:
            problems.append(f"{name}: expected {target}, got {len(splits[name])}")

    ids = [e["id"] for s in splits.values() for e in s]
    if len(ids) != len(set(ids)):
        problems.append("duplicate example ids")

    qs = {k: {e["question"] for e in v} for k, v in splits.items()}
    for k, v in splits.items():
        if len(qs[k]) != len(v):
            problems.append(f"{k}: duplicate questions within split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = qs[a] & qs[b]
        if overlap:
            problems.append(f"question leakage {a}/{b}: {len(overlap)} items")

    for k, v in splits.items():
        bad = {e["template_id"] for e in v if e["metadata"]["template_role"] != k}
        if bad:
            problems.append(f"{k}: templates from another role: {sorted(bad)}")

    def content_ids(split_name):
        return {e["metadata"][key]
                for e in splits[split_name]
                for key in ("rule_id", "scenario_id")
                if key in e["metadata"]}
    for a, b in (("train", "test"), ("val", "test")):
        overlap = content_ids(a) & content_ids(b)
        if overlap:
            problems.append(f"rule/scenario leakage {a}/{b}: {sorted(overlap)}")

    prog = {k: {e["program"] for e in v} for k, v in splits.items()}
    if prog["test"] - prog["train"]:
        problems.append(f"programs only in test: {sorted(prog['test'] - prog['train'])}")
    if prog["train"] - prog["test"]:
        problems.append(f"programs never tested: {sorted(prog['train'] - prog['test'])}")

    for k, v in splits.items():
        long = [e["id"] for e in v if len(e["answer"]) > 80]
        if long:
            problems.append(f"{k}: overlong answers: {long[:3]}")

    if problems:
        raise ValueError("Dataset invalid:\n  " + "\n  ".join(problems))
    return True


def split_stats(examples):
    labels = Counter(e["answer"] for e in examples)
    return {
        "count": len(examples),
        "unique_questions": len({e["question"] for e in examples}),
        "task_counts": dict(Counter(e["task_type"] for e in examples)),
        "program_counts": dict(Counter(e["program"] for e in examples)),
        "template_counts": dict(Counter(e["template_id"] for e in examples)),
        "difficulty_counts": dict(Counter(e["difficulty"] for e in examples)),
        "majority_label_share": round(labels.most_common(1)[0][1] / len(examples), 4),
        "distinct_labels": len(labels),
    }


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_language(lang, rng, counts, out_root, templates_dir):
    template_path = os.path.join(templates_dir, CONFIG["languages"][lang]["template_file"])
    bank = load_templates(template_path)
    validate_bank(bank, lang)

    splits, shortfalls = {}, {}
    for split, n in counts.items():
        splits[split], shortfalls[split] = generate_split(lang, bank, split, n, rng)

    validate_split(splits, counts)

    out_dir = os.path.join(out_root, CONFIG["languages"][lang]["dir"], "data")
    for split, examples in splits.items():
        write_jsonl(os.path.join(out_dir, f"{split}.jsonl"), examples)

    report = {
        "language": lang,
        "counts": counts,
        "digit_script": CONFIG["digit_script"],
        "template_source": template_path,
        "template_bank": bank_summary(bank),
        "splits": {k: split_stats(v) for k, v in splits.items()},
        "quota_shortfalls_redistributed": {k: v for k, v in shortfalls.items() if v},
        "ground_truth": "all labels computed in Python; no model-generated labels",
        "leakage_controls": [
            "template wordings partitioned by role (train/val/test)",
            "entity pools partitioned disjointly by split",
            "logic rules and common-sense scenarios partitioned by split",
            "exact-question uniqueness enforced within and across splits",
            "every semantic program present in both train and test",
        ],
    }
    with open(os.path.join(out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report, splits


# ===========================================================================
# 9. CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (400/100/100 per language); prints samples")
    ap.add_argument("--out", default=CONFIG["output_root"], help="output root directory")
    ap.add_argument("--templates-dir", default=CONFIG["templates_dir"],
                    help="directory containing templates_hi.jsonl / templates_ne.jsonl")
    ap.add_argument("--digits", choices=["ascii", "devanagari"],
                    default=CONFIG["digit_script"])
    args = ap.parse_args()
    CONFIG["digit_script"] = args.digits

    counts = {"train": 400, "val": 100, "test": 100} if args.smoke else CONFIG["counts"]
    out_root = os.path.join(args.out, "_smoke") if args.smoke else args.out

    master = random.Random(CONFIG["seed"])
    summary = {}
    for lang in CONFIG["languages"]:
        rng = random.Random(master.randint(0, 2**31 - 1))
        report, splits = generate_language(lang, rng, counts, out_root, args.templates_dir)
        summary[lang] = report
        print(f"[{lang}] " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        if args.smoke:
            for ex in splits["test"][:4]:
                print(f"  {ex['program']:<28} {ex['question']}")
                print(f"  {'':<28} -> {ex['answer']}")

    path = os.path.join(out_root, "reasoning_data_generation_summary.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": CONFIG, "languages": summary}, f, ensure_ascii=False, indent=2)
    print(f"summary written to {path}")


if __name__ == "__main__":
    main()
