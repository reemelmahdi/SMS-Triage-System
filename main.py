"""
Women's Health SMS Triage System
Pipeline: Privacy Filter → Triage → Safety Routing → Operator Notes
"""

import re, os, time, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PRIVACY FILTER (Text Sanitization)
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_text(text: str) -> str:
    """Replace PII with typed placeholders. Handles English and Kiswahili."""
    s = text

    # 1. PHONE NUMBERS
    s = re.sub(r'\+\d{1,3}[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{3,4}', '[PHONE]', s)
    s = re.sub(r'\b0[67]\d{2}[\s\-]?\d{3}[\s\-]?\d{3,4}\b', '[PHONE]', s)

    # 2. TRIBE / ETHNICITY — must run BEFORE name patterns
    tribe_names = (
        r'Maasai|Kikuyu|Luo|Kamba|Kalenjin|Luhya|Kisii|Meru|Embu|Taita|'
        r'Mijikenda|Pokomo|Nandi|Samburu|Turkana|Somali|Dinka|Nuer|Acholi|'
        r'Baganda|Sukuma|Chagga|Haya|Nyamwezi')
    tribe_keywords = r'tribe|ethnic(?:ity)?|clan|ancestry|kabila'
    s = re.sub(rf'(?i)\b(?:{tribe_names}|{tribe_keywords})\b', '[TRIBE/ETHNICITY]', s)

    # 3. NAMES (keyword-anchored)
    s = re.sub(r'(?i)\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'my name is [NAME]', s)
    s = re.sub(r'(?i)\bthis is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'this is [NAME]', s)
    s = re.sub(r"(?i)\bI(?:'m| am) called\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", 'I am called [NAME]', s)
    s = re.sub(r'(?i)\bjina\s+langu\s+ni\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'jina langu ni [NAME]', s)
    s = re.sub(r'(?i)\bnaitwa\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'naitwa [NAME]', s)
    s = re.sub(r'(?i)\bmimi\s+ni\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', 'mimi ni [NAME]', s)

    # 4. NATIONAL ID
    s = re.sub(
        r'(?i)\b(?:national[\s_]?id|id[\s_]?number|namba\s+ya\s+kitambulisho)\s*[:\-]?\s*[a-zA-Z0-9]{4,}',
        '[NATIONAL ID]', s)

    # 5. GENERIC ID
    s = re.sub(r'(?i)\b(?:id|identification)\s*[:\-]?\s*(?=[a-zA-Z0-9]*\d)[a-zA-Z0-9]{4,}\b', '[ID]', s)

    # 6. PASSPORT NUMBER
    s = re.sub(r'(?i)\bpassport(?:[\s_]?(?:number|no|namba))?\s*[:\-]?\s*[a-zA-Z0-9]{4,}', '[PASSPORT]', s)
    s = re.sub(r'\b[A-Z]{1,2}\d{6,7}\b', '[PASSPORT]', s)

    # 7. PATIENT ID
    s = re.sub(
        r'(?i)(?:\bpatient[\s_]?(?:id|no|number)\b|(?<!\w)pt\.?\s*(?:id|no)\b|'
        r'\bnamba\s+ya\s+mgonjwa\b)\s*[:\-]?\s*[a-zA-Z0-9]{4,}', '[PATIENT ID]', s)

    # 8. HEALTH INSURANCE — scheme name preserved, number stripped
    s = re.sub(r'(?i)\b(NHIF|SHA|SHIF)\b[\s\-]*(?:no\.?|number|card|namba)?\s*[:\-]?\s*\d{5,}',
               r'\1 [INSURANCE NUMBER]', s)
    s = re.sub(
        r'(?i)\b(?:health[\s_]?insurance[\s_]?(?:number|no)|'
        r'insurance[\s_]?(?:number|no)|bima[\s_]?(?:namba|no))\s*[:\-]?\s*[a-zA-Z0-9]{4,}',
        '[HEALTH INSURANCE NUMBER]', s)

    # 9. HEALTHCARE PROVIDER NAME
    s = re.sub(
        r'(?i)(?:^|(?<=\s))(?:doctor|dr\.?|daktari|dkt\.?|nurse|midwife|mw\.?)\s+'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', '[HEALTHCARE PROVIDER]', s)

    # 10. HEALTHCARE FACILITY NAME
    s = re.sub(
        r'(?i)\b(?:hospital|clinic|dispensary|health\s+cent(?:re|er)|zahanati|hospitali)\s+'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', '[HEALTHCARE FACILITY]', s)

    # 11. SOCIAL RELATIONSHIP TERMS
    s = re.sub(
        r'(?i)\b(?:husband|wife|mother|father|sister|brother|child|daughter|son|'
        r'aunt|uncle|grandparent|fianc[eé]e?|partner|boyfriend|girlfriend|'
        r'mume|mke|mama|baba|kaka|dada|mtoto|shangazi|mjomba)\b', '[RELATIVE]', s)

    return s


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TRIAGE SYSTEM (Classification)
# ═══════════════════════════════════════════════════════════════════════════════

# -- Layer 1: Keyword scoring --

TRIAGE_KEYWORDS = {
    "Medical Emergency": {
        "damu": 2, "kutokwa damu": 3, "ninatokwa damu": 3, "kutapika": 2,
        "homa kali": 3, "maumivu makali": 3, "kali sana": 2, "dharura": 3,
        "zimia": 3, "amezimia": 3, "hana fahamu": 3, "kupoteza fahamu": 3,
        "siwezi kupumua": 3, "kupumua": 2, "uchungu wa kuzaa": 3,
        "uchungu": 2, "kifafa": 3, "sumu": 3, "ajali": 3, "kuumia": 2,
        "kinauma sana": 2, "inauma sana": 2, "siponi": 2,
        "msaada wa haraka": 3, "sasa hivi": 1, "haraka": 1,
    },
    "Appointment Request": {
        "miadi": 3, "appointment": 3, "kuonana na daktari": 3,
        "naomba kuonana": 3, "nataka kuonana": 3, "naomba miadi": 3,
        "kupanga": 2, "ratiba": 2, "kufungua kadi": 2, "kujiandikisha": 2,
        "booking": 2, "kesho": 1, "wiki ijayo": 1, "saa ngapi": 1,
        "lini": 1, "kliniki": 1, "tarehe": 1,
    },
    "General Health Info": {
        "nataka kujua": 3, "naomba kujua": 3, "ni salama": 2, "swali": 2,
        "habari kuhusu": 2, "maelezo": 2, "ushauri": 2,
        "inamaanisha nini": 2, "kuzuia mimba": 2, "uzazi wa mpango": 2,
        "je,": 1, "je ": 1, "kunyonyesha": 1, "chanjo": 1, "lishe": 1,
        "dawa ya": 1, "inafaa": 1, "naweza": 1, "vipi": 1,
    },
}
STRONG_WIN = 4
EMERGENCY_FLOOR = 2


def keyword_scores(text):
    t = text.lower()
    return {label: sum(w for kw, w in kws.items() if kw in t)
            for label, kws in TRIAGE_KEYWORDS.items()}


# -- Layer 2: Embedding similarity --

PROTOTYPES = {
    "Medical Emergency": [
        "query: ninatokwa damu nyingi sana",
        "query: nina homa kali sana na kutapika",
        "query: mtoto wangu amezimia hana fahamu",
        "query: maumivu makali sana siwezi kuvumilia",
        "query: ninahitaji msaada wa dharura sasa hivi",
        "query: siwezi kupumua vizuri nahisi kifua kinabana",
    ],
    "Appointment Request": [
        "query: naomba kuonana na daktari kesho",
        "query: nataka kupanga miadi wiki ijayo",
        "query: ninaweza kupata appointment lini",
        "query: naomba kujiandikisha kliniki yenu",
        "query: nataka kufungua kadi ya kliniki",
    ],
    "General Health Info": [
        "query: nahitaji dawa ya kuzuia mimba",
        "query: je ni salama kunyonyesha nikiwa na homa",
        "query: nataka kujua kuhusu uzazi wa mpango",
        "query: naomba ushauri kuhusu lishe ya mtoto",
        "query: chanjo ya watoto inapatikana wapi",
        "query: dalili hizi zinamaanisha nini",
    ],
    "Others": [
        "query: habari za asubuhi",
        "query: asante sana kwa msaada wenu",
        "query: nashukuru kwa huduma nzuri",
        "query: sijui kama nimetuma vizuri",
    ],
}

_embed_model = None
_proto_vecs = None


def _load_embed_model():
    global _embed_model, _proto_vecs
    if _embed_model is None:
        _embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
        _proto_vecs = {}
        for label, sents in PROTOTYPES.items():
            v = _embed_model.encode(sents, normalize_embeddings=True).mean(axis=0)
            _proto_vecs[label] = v / np.linalg.norm(v)


def embedding_scores(text):
    _load_embed_model()
    vec = _embed_model.encode([f"query: {text}"], normalize_embeddings=True)[0]
    return {label: float(np.dot(vec, pv)) for label, pv in _proto_vecs.items()}


# -- Combined classifier --

def classify(text):
    """Returns {"label": str, "method": str, "confidence": float}."""
    kw = keyword_scores(text)
    ranked = sorted(kw.items(), key=lambda x: -x[1])
    best_label, best_score = ranked[0]
    runner_score = ranked[1][1]

    # Layer 1 decisive win
    if best_score >= STRONG_WIN and (best_score - runner_score) >= 2:
        return {"label": best_label, "method": "keywords", "confidence": min(best_score / 8, 1.0)}

    # Very short, no keywords → Others
    if best_score == 0 and len(text.split()) <= 4:
        return {"label": "Others", "method": "keywords", "confidence": 0.5}

    # Layer 2: embedding fallback
    emb = embedding_scores(text)
    emb_label = max(emb, key=emb.get)

    # Safety rule: emergency keywords forbid downgrade
    if kw["Medical Emergency"] >= EMERGENCY_FLOOR and emb_label != "Medical Emergency":
        return {"label": "Medical Emergency", "method": "safety-floor", "confidence": 0.7}

    return {"label": emb_label, "method": "embedding", "confidence": emb[emb_label]}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SAFETY-AWARE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

EMERGENCY_TERMS = {
    # English — medical
    'bleeding', 'unconscious', 'fainted', 'breathing', 'severe pain',
    'poisoning', 'accident', 'high fever', 'labour', 'hit', 'drunk',
    # English — GBV / danger
    'beat me', 'beaten', 'attacked', 'threatened', 'rape', 'danger',
    # Swahili — medical
    'damu', 'kutokwa damu', 'kutapika', 'homa kali', 'maumivu makali',
    'kali sana', 'dharura', 'zimia', 'amezimia', 'hana fahamu',
    'siwezi kupumua', 'uchungu wa kuzaa', 'kifafa', 'sumu', 'ajali',
    'kinauma sana', 'siponi',
    # Swahili — GBV / danger
    'alinipiga', 'ubakaji', 'ubakiwa', 'hatari', 'ninapigwa', 'nisaidie',
}
GREETING_TERMS = {'hello', 'hi', 'hey', 'habari', 'mambo', 'hujambo', 'salaam'}
HEALTH_SIGNALS = {
    'pain', 'fever', 'sick', 'help', 'doctor', 'blood', 'pregnant',
    'medicine', 'worried', 'scared', 'not sure',
    'dawa', 'daktari', 'maumivu', 'homa', 'msaada', 'mimba',
    'sijui', 'sifahamu', 'nahangaika',
}
LOW_CONFIDENCE = 0.60


def apply_safety_routing(sanitized_text, predicted_label, confidence=1.0):
    """Returns {"final_route": str, "priority": str, "reason": str}."""
    text = sanitized_text.lower().strip()

    # 1. Emergency / GBV override
    if any(term in text for term in EMERGENCY_TERMS):
        return {"final_route": "Urgent Human Review", "priority": "High",
                "reason": "Emergency or safety signal detected; overriding model prediction."}

    # 2. Bare greeting — possible cautious disclosure
    if len(text.strip('?!. ').split()) <= 2 and any(g in text for g in GREETING_TERMS):
        return {"final_route": "Human Review", "priority": "Medium",
                "reason": "Short greeting may be a cautious disclosure test; escalating out of caution."}

    # 3. Structural noise: very short, no health signal
    if len(text.strip('?!. ').split()) <= 4 and not any(w in text for w in HEALTH_SIGNALS):
        return {"final_route": "Probable Noise", "priority": "Low",
                "reason": "Short message with no health signal detected."}

    # 4. Standard classifier mapping
    if predicted_label == 'Medical Emergency':
        reason = (f'Low-confidence prediction ({confidence:.0%}); operator should verify.'
                  if confidence < LOW_CONFIDENCE else 'Model classified as Medical Emergency.')
        return {"final_route": "Urgent Human Review", "priority": "High", "reason": reason}

    if predicted_label == 'Appointment Request':
        return {"final_route": "Appointments Queue", "priority": "Low", "reason": "Routine scheduling request."}

    if predicted_label == 'General Health Info':
        return {"final_route": "General Health Queue", "priority": "Low", "reason": "General health information inquiry."}

    # 5. Other / ambiguous
    return {"final_route": "Human Review", "priority": "Medium",
            "reason": "Unrecognised message; routing to human rather than dismissing."}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: OPERATOR NOTES (Retrieval + Note Generation)
# ═══════════════════════════════════════════════════════════════════════════════

# -- Swahili → English term mapping for TF-IDF retrieval --
SW_TO_EN = {
    "damu": "bleeding", "kutokwa damu": "severe bleeding", "kutapika": "vomiting",
    "homa": "fever", "homa kali": "severe fever", "kichwa kinauma": "headache",
    "kupumua": "breathing", "zimia": "consciousness", "amezimia": "loss of consciousness",
    "kupigwa": "physical violence", "kudhulumiwa": "violence", "hatari": "danger",
    "mimba": "pregnancy", "ujauzito": "pregnancy", "hedhi": "period",
    "kunyonyesha": "breastfeeding", "kuzuia mimba": "contraception",
    "utoaji mimba": "abortion", "dawa": "medication", "daktari": "doctor",
    "kliniki": "clinic", "hospitali": "clinic", "miadi": "appointment",
    "kuonana na daktari": "appointment doctor", "siponi": "severe headache",
    "maumivu": "pain", "maumivu makali": "severe pain",
}

# -- English synonym mapping for terms not in the KB --
EN_SYNONYMS = {
    "hits me": "physical violence", "hit me": "physical violence",
    "beats me": "physical violence", "beat me": "physical violence",
    "drunk": "violence danger", "attacked": "physical violence",
    "where is the clinic": "clinic location",
    "clinic located": "clinic location",
}

_kb_paragraphs = None
_tfidf_vectorizer = None
_tfidf_matrix = None


def _load_knowledge_base(kb_dir="knowledge_base"):
    global _kb_paragraphs, _tfidf_vectorizer, _tfidf_matrix
    if _kb_paragraphs is not None:
        return

    _kb_paragraphs = []
    for filename in sorted(os.listdir(kb_dir)):
        with open(os.path.join(kb_dir, filename)) as f:
            title = ""
            for line in f:
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:]
                elif line and not line.startswith("#"):
                    _kb_paragraphs.append({"source": filename, "title": title, "text": line})

    corpus = [p["text"] for p in _kb_paragraphs]
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(corpus)


def _translate_query(text):
    t = text.lower()
    for sw, en in sorted(SW_TO_EN.items(), key=lambda x: -len(x[0])):
        t = t.replace(sw, en)
    for phrase, replacement in sorted(EN_SYNONYMS.items(), key=lambda x: -len(x[0])):
        t = t.replace(phrase, replacement)
    return t


def retrieve_and_note(sanitized_text, final_route, kb_dir="knowledge_base"):
    """Returns {"retrieved_source": str, "operator_note": str}."""
    _load_knowledge_base(kb_dir)

    translated = _translate_query(sanitized_text)
    scores = cosine_similarity(
        _tfidf_vectorizer.transform([translated]), _tfidf_matrix).flatten()
    best = scores.argmax()

    if scores[best] < 0.05:
        return {"retrieved_source": "None",
                "operator_note": "No relevant knowledge-base entry found. Use professional judgment."}

    p = _kb_paragraphs[best]
    note = f"Refer to: {p['title']} ({p['source']})."
    if final_route == "Urgent Human Review":
        note += " Priority case — review immediately."
    note += " Do not provide definitive medical or legal advice."

    return {"retrieved_source": f"knowledge_base/{p['source']}", "operator_note": note}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE: Run all steps on a single message
# ═══════════════════════════════════════════════════════════════════════════════

def process_sms(raw_text):
    """Run the full pipeline on a single SMS. Returns dict with all fields."""
    sanitized = sanitize_text(raw_text)
    triage = classify(sanitized)
    routing = apply_safety_routing(sanitized, triage["label"], triage["confidence"])
    notes = retrieve_and_note(sanitized, routing["final_route"])

    return {
        "original_text": raw_text,
        "sanitized_text": sanitized,
        "predicted_label": triage["label"],
        "confidence": round(triage["confidence"], 2),
        "method": triage["method"],
        "final_route": routing["final_route"],
        "priority": routing["priority"],
        "reason": routing["reason"],
        "retrieved_source": notes["retrieved_source"],
        "operator_note": notes["operator_note"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING: Run pipeline on CSV with progress bar
# ═══════════════════════════════════════════════════════════════════════════════

def progress_bar(current, total, label="", bar_len=40):
    pct = current / total
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {label} |{bar}| {current}/{total}", end="", flush=True)
    if current == total:
        print(" ✓")


def run_batch(csv_path="sandbox_health_sms.csv", kb_dir="knowledge_base"):
    """Run full pipeline on CSV dataset with progress display."""
    print("Loading dataset...")
    df = pd.read_csv(csv_path, dtype={"id": str})
    n = len(df)

    # Step 1: Privacy Filter
    print("\n[Step 1/4] Privacy Filter")
    for i, idx in enumerate(df.index):
        df.at[idx, "sanitized_text"] = sanitize_text(df.at[idx, "original_text"])
        progress_bar(i + 1, n, "Sanitizing")

    # Step 2: Triage System
    print("\n[Step 2/4] Triage System")
    labels, confidences, methods = [], [], []
    for i, idx in enumerate(df.index):
        r = classify(df.at[idx, "sanitized_text"])
        labels.append(r["label"])
        confidences.append(r["confidence"])
        methods.append(r["method"])
        progress_bar(i + 1, n, "Classifying")
    df["predicted_label"] = labels
    df["confidence"] = confidences
    df["method"] = methods

    # Step 3: Safety-Aware Routing
    print("\n[Step 3/4] Safety-Aware Routing")
    routes, priorities, reasons = [], [], []
    for i, idx in enumerate(df.index):
        r = apply_safety_routing(df.at[idx, "sanitized_text"],
                                 df.at[idx, "predicted_label"],
                                 df.at[idx, "confidence"])
        routes.append(r["final_route"])
        priorities.append(r["priority"])
        reasons.append(r["reason"])
        progress_bar(i + 1, n, "Routing")
    df["final_route"] = routes
    df["priority"] = priorities
    df["reason"] = reasons

    # Step 4: Operator Notes
    print("\n[Step 4/4] Operator Notes")
    sources, notes = [], []
    for i, idx in enumerate(df.index):
        r = retrieve_and_note(df.at[idx, "sanitized_text"],
                              df.at[idx, "final_route"], kb_dir)
        sources.append(r["retrieved_source"])
        notes.append(r["operator_note"])
        progress_bar(i + 1, n, "Retrieving")
    df["retrieved_source"] = sources
    df["operator_note"] = notes

    print(f"\nDone — processed {n} messages.\n")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE: Test individual SMS messages
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_mode():
    """Let the user type SMS messages and see each pipeline stage."""
    print("\n" + "=" * 60)
    print("  SMS Triage System — Interactive Mode")
    print("=" * 60)
    print("Type an SMS message to test, or 'exit' to quit.\n")

    while True:
        raw = input("SMS > ").strip()
        if raw.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye.")
            break
        if not raw:
            continue

        result = process_sms(raw)

        print(f"\n  ┌─ Step 1: Privacy Filter")
        print(f"  │  {result['sanitized_text']}")
        print(f"  │")
        print(f"  ├─ Step 2: Triage System")
        print(f"  │  Label:      {result['predicted_label']}")
        print(f"  │  Confidence: {result['confidence']}")
        print(f"  │  Method:     {result['method']}")
        print(f"  │")
        print(f"  ├─ Step 3: Safety-Aware Routing")
        print(f"  │  Route:    {result['final_route']}")
        print(f"  │  Priority: {result['priority']}")
        print(f"  │  Reason:   {result['reason']}")
        print(f"  │")
        print(f"  └─ Step 4: Operator Notes")
        print(f"     Source: {result['retrieved_source']}")
        print(f"     Note:   {result['operator_note']}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = run_batch()
    print(df[["id", "predicted_label", "final_route", "priority", "retrieved_source"]].to_string(index=False))
    interactive_mode()