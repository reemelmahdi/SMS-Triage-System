# Reflections

## 1. The Safety Paradox

This system already handles this correctly. Terms like "contraception", "abortion", and "kuzuia mimba" are not in the emergency keyword list — they sit in the triage keyword layer under General Health Info, which routes them to the General Health Queue, not to a block or a filter. The safety routing step only escalates messages containing genuine danger signals like bleeding, violence, or loss of consciousness.

The key design decision is that the privacy filter and the safety router operate on different word lists with different purposes. The privacy filter removes identifiers. The safety router detects urgency. Neither blocks reproductive health vocabulary. Row 015 ("Is abortion legal in this context?") flows cleanly to the General Health Queue with a note pointing to `reproductive_health.md`, which explicitly instructs operators not to censor or judge.

The real risk is not in this system — it is in upstream SMS gateway filters or downstream LLM safety layers that might flag these terms. Any integration must whitelist reproductive health vocabulary at every layer, because a false positive here silences the person who most needs help.

## 2. Swahili Scalability

At 1 million messages, the keyword layer remains the strongest asset — it runs in under 1 ms per message, scales linearly, and handles the majority of Kiswahili messages correctly. The bottleneck is Layer 2: embedding inference at 30–50 ms per message on CPU means ~14 hours for a million messages if every message hits the fallback.

The most resource-efficient path is to reduce how often Layer 2 is invoked. I would work with local partners to collect 500–1000 validated Kiswahili examples across the four categories, then expand the keyword dictionary to cover the patterns those examples reveal. Every new keyword that resolves a message at Layer 1 is one fewer embedding call.

For the messages that still need Layer 2, I would fine-tune the e5-small model using LoRA on those same validated examples — a few hours on a single GPU, no full retraining. Translation pipelines are tempting but risky: they introduce hallucination, add latency, and create a dependency on a model that was never trained on informal health SMS.

## 3. Pilot Gate and Production Watch

**Before launch**, the evaluation suite must test three things: that every emergency keyword in both languages triggers Urgent Human Review regardless of classifier output, that reproductive health queries are never blocked or misrouted, and that the sanitisation function catches all PII patterns present in real messages. The test set must be validated by Kiswahili-speaking health workers from the partner organisation — not by engineers or by AI. The single result that blocks launch: any Medical Emergency message routed to Probable Noise or General Health Queue. One missed emergency is enough to stop.

**Once live**, I would log only the sanitised text, predicted label, final route, confidence score, and operator override decision — never the original text or any PII. I would monitor the override rate: how often operators change the system's routing. A spike in overrides means the classifier is drifting. I would alert on two conditions: emergency override rate exceeding 10% (operators are catching emergencies the system missed), and a sudden increase in Probable Noise volume (legitimate messages being dismissed).
