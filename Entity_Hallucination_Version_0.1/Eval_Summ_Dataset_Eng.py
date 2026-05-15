import streamlit as st
import json
import spacy
import stanza
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from decimal import Decimal
from copy import deepcopy
from scipy.stats import pearsonr, ttest_rel
from rouge_score import rouge_scorer
import numpy as np

# ================================================================
# GLOBAL EMBEDDING CACHE
# ================================================================
embedding_cache = {}

# ================================================================
# LOAD MODELS (CACHED)
# ================================================================
@st.cache_resource
def load_models():
    nlp_spacy = spacy.load("en_core_web_sm")
    stanza.download("en", verbose=False)
    nlp_stanza = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=False,
        verbose=False
    )
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp_spacy, nlp_stanza, sbert

nlp_spacy, nlp_stanza, sbert = load_models()
# ================================================================
# ROUGE SCORER
# ================================================================
rouge_scorer_obj = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)
# ================================================================
# CONSTANTS
# ================================================================
MODELS = [
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "t5-large",
    "sshleifer/distilbart-cnn-12-6",
    "RefSum"
]

GENERIC_SUBJECTS = {"government", "company", "authority", "official", "body"}
REPORTING_VERBS = {"say", "tell", "report", "add", "warn", "claim"}
VALID_POS_SUBJECTS = {"NOUN", "PROPN"}
VALID_NER = {"PERSON", "ORG", "GPE", "NORP"}

# ================================================================
# TEXT CLEANING & CHUNKING
# ================================================================
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s.,!?'-]", "", text).strip()

def sentence_chunks(text, max_tokens=250):
    doc = nlp_spacy(text)
    chunks = []
    for sent in doc.sents:
        words = sent.text.split()
        for i in range(0, len(words), max_tokens):
            chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks
# ================================================================
# COHERENCE METRIC (SAFE VERSION - CPU NUMPY ONLY)
# ================================================================
def compute_coherence(text):
    text = clean_text(text)
    doc = nlp_spacy(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if len(sentences) < 2:
        return 1.0

    embeddings = sbert.encode(sentences)  # returns numpy array
    similarities = []

    for i in range(len(embeddings) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i + 1]

        # cosine similarity (manual safe version)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            sim = 0
        else:
            sim = np.dot(v1, v2) / denom

        similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 1.0
# ================================================================
# NORMALIZATION
# ================================================================
def normalize(tok):
    if not tok or tok.is_stop or tok.is_punct:
        return None
    if tok.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ"}:
        return None
    return tok.lemma_.lower()

# ================================================================
# NER-ALIGNED SUBJECT GROUNDING
# ================================================================
def ground_subject(tok, sent):
    if tok.text.lower() not in GENERIC_SUBJECTS:
        return normalize(tok)
    for ent in sent.ents:
        if ent.label_ in VALID_NER:
            return ent.text.lower()
    return normalize(tok)

# ================================================================
# NOMINAL RELATION RECOVERY
# ================================================================
def extract_nominal_relations(sent):
    rels = []
    for tok in sent:
        if tok.pos_ == "NOUN":
            for c in tok.children:
                if c.dep_ == "prep":
                    for pobj in c.children:
                        if pobj.pos_ == "NOUN":
                            rels.append(
                                (normalize(tok), c.lemma_, normalize(pobj))
                            )
    return rels

# ================================================================
# IMPROVED SVO EXTRACTION (ALL FIXES)
# =======================================F=========================
@st.cache_data(show_spinner=False)
def extract_svo(text):
    relations = set()
    seen = set()

    for chunk in sentence_chunks(clean_text(text)):
        doc = nlp_spacy(chunk)

        for sent in doc.sents:

            # ---- VERB-BASED SVO ----
            for verb in sent:
                if verb.pos_ != "VERB":
                    continue

                vlemma = verb.lemma_.lower()
                if vlemma in REPORTING_VERBS:
                    continue

                neg = any(c.dep_ == "neg" for c in verb.children)
                vfinal = f"not_{vlemma}" if neg else vlemma

                subjects, objects = [], []

                for c in verb.children:
                    if c.dep_ in {"nsubj", "nsubjpass"}:
                        subjects.append(c)
                    if c.dep_ in {"dobj", "obj", "attr"}:
                        objects.append(c)

                # Passive agent recovery
                for c in verb.children:
                    if c.dep_ == "agent":
                        for gc in c.children:
                            if gc.dep_ == "pobj":
                                subjects.append(gc)

                for s in subjects:
                    if s.pos_ not in VALID_POS_SUBJECTS:
                        continue
                    s_norm = ground_subject(s, sent)
                    if not s_norm:
                        continue

                    for o in objects:
                        o_norm = normalize(o)
                        if not o_norm:
                            continue

                        key = (s.i, verb.i, o.i)
                        if key in seen:
                            continue
                        seen.add(key)
                        relations.add((s_norm, vfinal, o_norm))

            # ---- NOMINAL FALLBACK ----
            for r in extract_nominal_relations(sent):
                if all(r):
                    relations.add(r)

    return list(relations)

# ================================================================
# FILTER
# ================================================================
def filter_unwanted_symbols(tuples):
    return set(tuples)

from sentence_transformers import util

SIM_THRESHOLD = 0.85

def get_embedding(text, model):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True)
    return embedding_cache[text]

def component_match(a, b, model):
    if a == b:
        return 1.0

    emb_a = get_embedding(a, model)
    emb_b = get_embedding(b, model)

    sim = util.cos_sim(emb_a, emb_b).item()

    return 1.0 if sim >= SIM_THRESHOLD else 0.0

def calculate_weighted_intersection(set1, set2, model):
    """
    Computes partial (1/3, 2/3, 1) SVO intersection score.
    Ensures one-to-one matching (no duplicates).
    """

    set1 = list(set1)
    set2 = list(set2)

    used_1 = set()
    total_score = 0.0

    for (s2, v2, o2, *_) in set2:

        best_score = 0.0
        best_index = None

        for i, (s1, v1, o1, *_) in enumerate(set1):
            if i in used_1:
                continue

            m_s = component_match(s1, s2, model)
            m_v = component_match(v1, v2, model)
            m_o = component_match(o1, o2, model)

            match_score = (m_s + m_v + m_o) / 3.0

            if match_score > best_score:
                best_score = match_score
                best_index = i

        if best_index is not None and best_score > 0:
            total_score += best_score
            used_1.add(best_index)

    return total_score

def calculate_weighted_triple_intersection(I, R, G, model):
    """
    Computes triple intersection with partial + semantic scoring.
    No duplicate usage.
    """

    used_I = set()
    used_R = set()
    total_score = 0.0

    for (sg, vg, og, *_) in G:

        best_score = 0.0
        best_i = None
        best_r = None

        for r_idx, (sr, vr, or_, *_) in enumerate(R):
            if r_idx in used_R:
                continue

            for i_idx, (si, vi, oi, *_) in enumerate(I):
                if i_idx in used_I:
                    continue

                m_s = min(
                    component_match(si, sg, model),
                    component_match(sr, sg, model)
                )

                m_v = min(
                    component_match(vi, vg, model),
                    component_match(vr, vg, model)
                )

                m_o = min(
                    component_match(oi, og, model),
                    component_match(or_, og, model)
                )

                score = (m_s + m_v + m_o) / 3.0

                if score > best_score:
                    best_score = score
                    best_i = i_idx
                    best_r = r_idx

        if best_score > 0:
            total_score += best_score
            used_I.add(best_i)
            used_R.add(best_r)

    return total_score

# ================================================================
# FULL RHI METRIC (EXACT FORMULA)
# ================================================================
def compute_hallucination_metrics(I, R, G):
    I, R, G = map(filter_unwanted_symbols, (I, R, G))
    len_I, len_R, len_G = len(I), len(R), len(G)

    IR = calculate_weighted_intersection(I, R, sbert)
    IG = calculate_weighted_intersection(I, G, sbert)
    RG = calculate_weighted_intersection(R, G, sbert)
    IRG = calculate_weighted_triple_intersection(I, R, G, sbert)

    precision = IG / len_G if len_G else 0
    recall = RG / len_R if len_R else 0
    ef1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    ef = (3 * IRG) / (len_I + len_R + len_G) if (len_I + len_R + len_G) else 0
    ph = (2 * RG) / (len_R + len_G) if (len_R + len_G) else 0
    of = (2 * (IG - IRG)) / (len_I + len_G) if (len_I + len_G) else 0
    nh = abs(len_G - (RG + IG - IRG)) / len_G if len_G else 0
    lf = (len_R - (IR - IRG)) / (len_R + len_G) if (len_R + len_G) else 0
    lh = (len_I - IG) / (len_I + len_G) if (len_I + len_G) else 0

    rhi = 1 + ((ef + ph) / 2) - ((of + lh + lf + nh) / 4)
    rhi = max(0, min(1, rhi))

    return ef1, rhi, {"ef": ef, "ph": ph, "of": of, "nh": nh, "lf": lf, "lh": lh}


# ================================================================
# ROUGE COMPUTATION
# ================================================================
def compute_rouge(reference_text, generated_text):
    reference_text = clean_text(reference_text)
    generated_text = clean_text(generated_text)

    if not reference_text or not generated_text:
        return 0.0, 0.0, 0.0

    scores = rouge_scorer_obj.score(reference_text, generated_text)

    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    return rouge1, rouge2, rougeL
# ================================================================
# DATASET PROCESSING 
# ================================================================
def process_dataset(data):
    global embedding_cache
    embedding_cache = {}
    results = deepcopy(data)
    metrics = {
    m: {
        "EF1": [],
        "RHI": [],
        "Coherence": [],  # ADDED
        "ROUGE1": [],
        "ROUGE2": [],
        "ROUGEL": [],
        "ef": [],
        "ph": [],
        "of": [],
        "nh": [],
        "lf": [],
        "lh": []
    }
    for m in MODELS
    }
    for _, entry in results.items():
        input_svo = extract_svo(entry.get("InputText", ""))
        ref_svo = extract_svo(entry.get("ReferenceSummary", ""))

        entry["InputText_SVO"] = input_svo
        entry["ReferenceSummary_SVO"] = ref_svo

        I = [(s, v, o, "1") for s, v, o in input_svo]
        R = [(s, v, o, "1") for s, v, o in ref_svo]

        for model in MODELS:
            gen_svo = extract_svo(entry.get(model, ""))
            entry[f"{model}_SVO"] = gen_svo

            G = [(s, v, o, "1") for s, v, o in gen_svo]
            ef1, rhi, sub = compute_hallucination_metrics(I, R, G)
            coherence_score = compute_coherence(entry.get(model, ""))
            entry[f"{model}_EF1"] = ef1
            entry[f"{model}_RHI"] = rhi
            entry[f"{model}_Coherence"] = coherence_score
            rouge1, rouge2, rougeL = compute_rouge(
            entry.get("ReferenceSummary", ""),
            entry.get(model, "")
            )
            entry[f"{model}_submetrics"] = sub

            metrics[model]["EF1"].append(ef1)
            metrics[model]["RHI"].append(rhi)
            metrics[model]["Coherence"].append(coherence_score)
            entry[f"{model}_ROUGE1"] = rouge1
            entry[f"{model}_ROUGE2"] = rouge2
            entry[f"{model}_ROUGEL"] = rougeL

            metrics[model]["ROUGE1"].append(rouge1)
            metrics[model]["ROUGE2"].append(rouge2)
            metrics[model]["ROUGEL"].append(rougeL)
            for k in sub:
                metrics[model][k].append(sub[k])

    return results, metrics

# ================================================================
# AGGREGATE TABLE
# ================================================================
def aggregate_table(metrics):
    return {
        m: {f"{k}": np.mean(v[k]) for k in v}
        for m, v in metrics.items()
    }

# ================================================================
# PLOTTING (LEGEND BELOW X-AXIS)
# ================================================================

def plot_recordwise(metrics, key):

    plt.figure(figsize=(10, 4))

    global_max = 0

    for m, v in metrics.items():

        vals = np.array(v[key])   # ✅ NO CLIPPING
        global_max = max(global_max, np.max(vals))

        plt.plot(vals, label=m)

    # ✅ Auto-scale instead of forcing 0–1
    plt.ylim(0, global_max * 1.05)

    plt.xlabel("Record Index")
    plt.ylabel(key)
    plt.title(f"{key} per Record")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(plt)

def plot_cdf(metrics):
    plt.figure(figsize=(8, 4))

    for m, v in metrics.items():
        x = np.sort(np.clip(v["RHI"], 0, 1))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=m)

    plt.xlabel("RHI")
    plt.ylabel("CDF")
    plt.title("CDF of RHI")

    # Legend BELOW X-axis
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(plt)


def plot_cdf_coherence(metrics):
    plt.figure(figsize=(8, 4))

    for m, v in metrics.items():
        x = np.sort(np.clip(v["Coherence"], -1, 1))
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=m)

    plt.xlabel("Coherence")
    plt.ylabel("CDF")
    plt.title("CDF of Coherence")

    # Legend BELOW X-axis
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(plt)
# ================================================================
# CORRELATION ANALYSIS (RHI vs Coherence)
# ================================================================
def compute_correlation(metrics):
    correlation_results = {}

    for model in MODELS:
        rhi_vals = metrics[model]["RHI"]
        coh_vals = metrics[model]["Coherence"]

        if len(rhi_vals) > 1:
            corr, p_value = pearsonr(rhi_vals, coh_vals)
        else:
            corr, p_value = 0, 1

        correlation_results[model] = {
            "pearson_r": round(corr, 4),
            "p_value": round(p_value, 6)
        }

    return correlation_results
# ================================================================
# LINE PLOT (RHI vs Coherence Per Record)
# ================================================================
def plot_line_rhi_coherence(metrics):
    plt.figure(figsize=(10, 5))

    for model in MODELS:
        rhi_vals = np.clip(metrics[model]["RHI"], 0, 1)
        coh_vals = np.clip(metrics[model]["Coherence"], 0, 1)

        # Plot RHI
        plt.plot(rhi_vals, linestyle='-', marker='o',
                 label=f"{model} - RHI")

        # Plot Coherence
        plt.plot(coh_vals, linestyle='--', marker='x',
                 label=f"{model} - Coherence")

    plt.xlabel("Record Index")
    plt.ylabel("Score")
    plt.title("RHI vs Coherence per Record")
    plt.ylim(0, 1)

    # Legend BELOW X-axis
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(plt)
# ================================================================
# PAIRED T-TEST (RHI Between Models)
# ================================================================
def paired_ttest_models(metrics):
    significance_results = {}

    for i in range(len(MODELS)):
        for j in range(i+1, len(MODELS)):
            m1 = MODELS[i]
            m2 = MODELS[j]

            vals1 = metrics[m1]["RHI"]
            vals2 = metrics[m2]["RHI"]

            if len(vals1) == len(vals2) and len(vals1) > 1:
                t_stat, p_val = ttest_rel(vals1, vals2)
            else:
                t_stat, p_val = 0, 1

            significance_results[f"{m1} vs {m2}"] = {
                "t_stat": round(t_stat, 4),
                "p_value": round(p_val, 6)
            }

    return significance_results
def extract_case_studies(results, metrics, top_k=2):

    case_studies = {}

    # ✅ stable ordering
    result_list = [
        results[k] for k in sorted(results.keys(), key=int)
    ]

    for model in MODELS:

        if model not in metrics:
            continue

        rhi_vals = metrics[model]["RHI"]
        coh_vals = metrics[model]["Coherence"]

        indices = list(range(len(rhi_vals)))

        # High coherence + Low RHI
        combo1 = sorted(
            indices,
            key=lambda i: (-coh_vals[i], rhi_vals[i])
        )[:top_k]

        # High RHI + Low Coherence
        combo2 = sorted(
            indices,
            key=lambda i: (-rhi_vals[i], coh_vals[i])
        )[:top_k]

        examples = {
            "High_Coherence_Low_RHI": [],
            "High_RHI_Low_Coherence": []
        }

        for idx in combo1:
            examples["High_Coherence_Low_RHI"].append({
                "RHI": rhi_vals[idx],
                "Coherence": coh_vals[idx],
                "Generated_Summary":
                    result_list[idx].get(model, "").strip()
            })

        for idx in combo2:
            examples["High_RHI_Low_Coherence"].append({
                "RHI": rhi_vals[idx],
                "Coherence": coh_vals[idx],
                "Generated_Summary":
                    result_list[idx].get(model, "").strip()
            })

        case_studies[model] = examples

    return case_studies
# ================================================================
# STREAMLIT UI
# ================================================================
st.title(
"🧠 Refining Relation Hallucination Evaluation in Text Summarization: "
"A Grounded and Decomposed RHI Framework - FOR DATASET SumEval 100 records"
)

uploaded = st.file_uploader(
    "Upload dataset (JSON format)",
    type=["json"]
)

if uploaded:

    # ------------------------------------------------
    # LOAD JSON DATASET (NEW FORMAT)
    # ------------------------------------------------
    data = json.load(uploaded)

    # ✅ Ensure required keys exist
    for idx, entry in data.items():

        entry.setdefault("InputText", "")
        entry.setdefault("ReferenceSummary", "")
        entry.setdefault("RefSum", entry.get("ReferenceSummary", ""))

        # ensure all models exist
        for model in MODELS:
            entry.setdefault(model, "")

    st.success("✅ Dataset loaded successfully")

    # ------------------------------------------------
    # RECORD COUNT
    # ------------------------------------------------
    total_records = len(data)
    st.info(f"Total records available: {total_records}")

    # ------------------------------------------------
    # RECORD LIMIT CONTROL
    # ------------------------------------------------
    record_limit = st.number_input(
        "Enter number of records to process",
        min_value=1,
        max_value=total_records,
        value=min(50, total_records),
        step=1
    )

    data_subset = dict(
        list(data.items())[:record_limit]
    )

    st.warning(f"🚀 Processing first {record_limit} records...")

    # ------------------------------------------------
    # RUN RHI PIPELINE
    # ------------------------------------------------
    with st.spinner(
        "Running improved SVO extraction + RHI evaluation…"
    ):
        results, metrics = process_dataset(data_subset)
        avg_table = aggregate_table(metrics)

    st.success("✅ Completed")

    # =================================================
    # VISUALIZATION
    # =================================================
    st.subheader("EF1 per Record")
    plot_recordwise(metrics, "EF1")

    st.subheader("RHI per Record")
    plot_recordwise(metrics, "RHI")

    st.subheader("Coherence per Record")
    plot_recordwise(metrics, "Coherence")

    st.subheader("CDF of RHI")
    plot_cdf(metrics)

    st.subheader("CDF of Coherence")
    plot_cdf_coherence(metrics)

    st.subheader("Average Hallucination Metrics")
    st.table(avg_table)

    # =================================================
    # ANALYSIS
    # =================================================
    st.subheader("Correlation Analysis (RHI vs Coherence)")
    corr_results = compute_correlation(metrics)
    st.json(corr_results)

    st.subheader("RHI vs Coherence Scatter Plot")
    plot_line_rhi_coherence(metrics)

    st.subheader("Statistical Significance (Paired t-test on RHI)")
    ttest_results = paired_ttest_models(metrics)
    st.json(ttest_results)

    # =================================================
    # QUALITATIVE CASE STUDIES
    # =================================================
    st.subheader("Qualitative Case Studies")

    cases = extract_case_studies(results, metrics)

    for model in MODELS:

        st.markdown(f"### {model}")

        st.markdown("**High Coherence + Low RHI**")
        for ex in cases[model]["High_Coherence_Low_RHI"]:
            st.write(
                f"RHI: {ex['RHI']} | Coherence: {ex['Coherence']}"
            )
            st.write(ex["Generated_Summary"])
            st.write("---")

        st.markdown("**High RHI + Low Coherence**")
        for ex in cases[model]["High_RHI_Low_Coherence"]:
            st.write(
                f"RHI: {ex['RHI']} | Coherence: {ex['Coherence']}"
            )
            st.write(ex["Generated_Summary"])
            st.write("---")

    # =================================================
    # DOWNLOAD RESULTS
    # =================================================
    st.download_button(
        "Download Full Results JSON",
        json.dumps(
            {"results": results,
             "average_metrics": avg_table},
            indent=2
        ),
        "full_rhi_results.json",
        "application/json"
    )