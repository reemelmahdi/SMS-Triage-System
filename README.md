# SMS Triage System

A lightweight AI-assisted triage system for a women's health SMS hotline in East Africa. The system processes incoming messages in English and Kiswahili, removes personally identifiable information, classifies messages by urgency, applies safety-aware routing, and generates operator notes — all designed to support human operators, never replace them.

## Project Structure

```
├── main.py                     # Full pipeline: Privacy → Triage → Routing → Notes
├── requirements.txt            # Python dependencies
├── sandbox_health_sms.csv      # Dataset (18 SMS messages)
└── knowledge_base/
    ├── clinic_services.md      # Appointment and clinic guidance
    ├── emergency_escalation.md # Emergency escalation policy
    └── reproductive_health.md  # Reproductive health guidance
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

The program processes all 18 messages with a progress bar for each step, prints a summary table, then enters interactive mode where you can type any SMS message and see the output of each pipeline stage. Type `exit` to quit.

---

## Step 1: Privacy Filter

The first step is to identify what needs to be filtered out of the original text to keep it as anonymous as possible. I compiled a list of PII categories and used regular expressions (regex) to detect and replace them with generic placeholders.

The function `sanitize_text` takes a raw SMS message as input and returns the sanitised version. It handles both English and Kiswahili, redacting the following:

- Phone numbers (e.g. `+254712345678`)
- Tribe or ethnicity references (e.g. Maasai, Kikuyu, kabila)
- Names, anchored to keyword triggers (e.g. "My name is Sarah", "mimi ni Jane")
- National ID, generic ID, passport numbers, and patient ID
- Health insurance member numbers, while preserving scheme names like NHIF
- Healthcare provider names (e.g. Dr. John, Daktari Aisha)
- Healthcare facility names (e.g. Nairobi Hospital, Zahanati ya Kijiji)
- Social relationship terms (e.g. husband, sister, mama, kaka)

I chose to keep locations and dates in the text. They are not personally identifiable on their own, and they provide important context. If a patient requires immediate attention, their location and age should remain available for medical support.

**AI Disclosure:** After creating an initial list, I consulted Claude to review it and suggest additional keywords that should be included in the privacy filter. This helped ensure I was not missing any important identifiers that could compromise anonymity. Although Claude suggested removing locations and dates, I decided to keep them for the reasons stated above.

---

## Step 2: Triage System

Classifying messages into triage categories proved to be the most challenging step. I needed a model that could handle both English and Kiswahili, run within limited hardware capacity, and accurately classify messages into four labels: **Medical Emergency**, **Appointment Request**, **General Health Info**, and **Others**.

### What I tried and why it did not work

I started with `paraphrase-multilingual-MiniLM-L12-v2`, which performed well on English text. However, it misclassified several Kiswahili messages. For example, "Nahitaji dawa ya kuzuia mimba" (I need contraceptive medication) was classified as Medical Emergency instead of General Health Info.

I then attempted to split the pipeline by language — detecting the language first using `fast_langdetect`, then routing English and Kiswahili messages to separate models. However, `fast_langdetect` did not perform well on short texts and code-switched messages, often misclassifying Kiswahili as English.

I switched to `langid` as an alternative. It detected the majority of the dataset correctly, but still misclassified some Kiswahili texts as Ethiopian or other languages. It was an improvement, but not reliable enough on its own.

### The final approach: a hybrid two-layer classifier

Rather than relying on language segregation as a prerequisite, I combined keyword scoring and embedding similarity into a single pipeline:

- **Layer 1 — Keyword scoring:** A pure Python keyword matcher with weighted Swahili and English health vocabulary. If a category wins decisively (score ≥ 4 and leads the runner-up by ≥ 2), it returns immediately. This layer runs in under 1 ms and handles the majority of messages.

- **Layer 2 — Embedding cosine similarity:** For ambiguous cases where keywords are insufficient, the system falls back to `intfloat/multilingual-e5-small` (117 MB, 384-dim, supports 100 languages including Swahili). It compares the message against prototype embeddings for each category. Note: e5 models require the `"query: "` prefix on all inputs.

- **Safety rule:** If Layer 1 detects any emergency keyword (score ≥ 2), the message can never be downgraded below Medical Emergency by Layer 2. False positives cost an operator seconds; false negatives can cost lives.

**CPU footprint:** ~400 MB RAM with both layers loaded. Latency is under 1 ms for keyword-only decisions and 30–50 ms when the embedding model is invoked.

### Classification performance

```
                     precision    recall  f1-score   support

Appointment Request       1.00      1.00      1.00         4
General Health Info       0.86      1.00      0.92         6
  Medical Emergency       1.00      1.00      1.00         6
              Other       0.00      0.00      0.00         2

           accuracy                           0.89        18
       weighted avg       0.84      0.89      0.86        18
```

**AI Disclosure:** I consulted Claude for recommendations on which approach would best fit the hardware limitations and bilingual requirements. Claude initially suggested `all-MiniLM-L12-v2`, but after checking, I found it only supports English. It also suggested `bigscience/mt0-base` for zero-shot multilingual classification, but this model did not follow instructions reliably, which is critical for a triage system. Finally, when I developed the language-split approach, Claude suggested combining both layers into a single pipeline — keywords first, then embeddings as a fallback — which is the architecture I adopted.

---

## Step 3: Safety-Aware Routing

This step ensures the system never blindly trusts the classifier on sensitive or ambiguous messages. The routing function takes the sanitised text and the model prediction, then returns a structured decision with a final route, priority level, and reason.

The logic follows four paths in order of priority:

1. **Emergency and GBV override:** If the text contains any emergency or violence keyword (in English or Kiswahili), the message is routed to `Urgent Human Review` with `High` priority, regardless of the model's prediction.

2. **Trauma-informed greeting handling:** A bare greeting like "Hello?" on a health or GBV hotline may be a cautious test from someone checking whether it is safe to disclose. These are routed to `Human Review` with `Medium` priority rather than being dismissed.

3. **Standard classifier mapping:** Messages classified as Medical Emergency go to `Urgent Human Review` (High), Appointment Request to `Appointments Queue` (Low), and General Health Info to `General Health Queue` (Low). Low-confidence Medical Emergency predictions are still escalated but flagged in the reason for the operator to verify.

4. **Noise detection:** Very short messages (≤ 4 words) with no health-related signal are routed to `Probable Noise` with `Low` priority. All other unrecognised messages go to `Human Review` rather than being silently dismissed.

**AI Disclosure:** I used Claude to identify the keyword sets, rewrite the functions to be more concise and efficient, and generate the test cases. Claude initially suggested using a keyword list for filtering noise messages, but I removed it because the number of possible noise words is unlimited. Instead, I implemented a structural check based on message length and the presence of health signals.

---

## Step 4: Operator Notes (Retrieval and Note Generation)

The final step retrieves the most relevant paragraph from the knowledge base and generates a short note the operator can scan quickly before responding.

The retrieval uses TF-IDF with cosine similarity over the three provided knowledge-base files. For English messages, this works directly — the SMS vocabulary overlaps with the knowledge-base content. For Kiswahili messages, however, there is no word overlap with the English knowledge base, so all Kiswahili queries returned no match initially.

To solve this without modifying the provided knowledge-base files, I added a Swahili-to-English term mapping dictionary. Before TF-IDF runs, the function replaces common Swahili health terms in the query with their English equivalents (e.g. "homa kali" → "severe fever", "kutapika" → "vomiting"). This bridges the language gap at the query level while leaving the knowledge base untouched.

I initially considered using a full translation model, but given the resource constraints, term-level translation proved both faster and sufficient for this task.

The generated operator note includes a reference to the matched knowledge-base file, a priority flag for urgent cases, and a reminder not to provide definitive medical or legal advice.

**AI Disclosure:** I used Claude to improve the initial retrieval function and to suggest the term-level translation approach as an alternative to a full translation model, which turned out to be faster and more practical. I also used Claude to transform the separate functions developed across the notebook into a unified pipeline in `main.py`.