# In-House Legal Intelligence System for Indian Case Law
**CS781: Advanced Natural Language Processing — Capstone Project (Autumn 2026)**  
**Department of Computer Science and Engineering, IIT Kanpur**

* **Author**: Brinda Fadadu
* **Roll Number**: 230307
* **Email**: [brindaf23@iitk.ac.in](mailto:brindaf23@iitk.ac.in)
* **Codabench Username**: `b230307`
* **Total Valid Codabench Submissions**: 51

---

## Executive Summary

Legal practices and corporate legal counsel operate under strict statutory obligations of confidentiality and legal professional privilege, strictly precluding the transmission of sensitive litigation files to public or commercial third-party LLM APIs. This capstone project designs, trains, evaluates, and deploys a fully self-hosted, modular, and privacy-preserving **Legal NLP Stack** specifically optimized for Indian Case Law under strict computational constraints (single Tesla T4 GPU, 16GB VRAM).

The system follows a strict transfer-learning protocol across **seven canonical legal intelligence tasks**:
1. **Phase 1: Foundation Language Model Pretraining** — Causal LM pretraining from scratch on Indian Supreme Court judgments across four custom transformer architectures.
2. **Phase 2: Legal Named Entity Recognition (NER)** — Span-level identification across 12 legal entity classes using constrained BIO Viterbi decoding.
3. **Phase 3: Rhetorical Role Prediction (RR)** — Sentence-level structural parsing across 13 rhetorical roles using hierarchical BiLSTM context modelling.
4. **Phase 4: Court Judgment Prediction (CJPE)** — Binary outcome classification (accepted vs. dismissed) on long unstructured judgments using multi-chunk hierarchical pooling.
5. **Phase 5: Legal Statute Identification (LSI)** — Extreme multi-label statute violation tagging across 116 target acts using label-wise attention and Asymmetric Loss.
6. **Phase 6: Abstractive Legal Summarization** — Generation of structured legal headnotes from long judgments using lead-tail-SumBasic prompt caching.
7. **Phase 7: Prior Case Retrieval (PCR)** — Precedent discovery across candidate pools (150–400 cases) using GradCache chunk-contrastive learning and hybrid BM25 fusion.

Across all adaptation phases, **Architecture 4 (ALiBi + QK-Norm + SwiGLU + Pre-RMSNorm)** served as the foundational backbone, and **Low-Rank Matrix Adaptation (LoRMA)** consistently outperformed full fine-tuning and additive LoRA by preventing catastrophic forgetting.

---

## Official Codabench Results

The official scores recorded on the final evaluation benchmark (recorded 7/11/26) across all seven phases are summarized below:

| Phase | Task | Primary Metric | Official Score | Benchmark Rank | Valid Submissions | Winning Configuration |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| **Phase 1** | Legal LM Pretraining | Word Continuation Coherence | **0.8263** | 110 | 7 | Arch4 (ALiBi + QK-Norm + SwiGLU) |
| **Phase 2** | Legal Named Entity Recognition | Strict Macro-F1 (Span) | **0.4145** | 101 | 12 | Arch4 + LoRMA + Constrained Viterbi |
| **Phase 3** | Rhetorical Role Prediction | Macro-F1 (13 classes) | **0.4122** | 118 | 10 | Arch4 + LoRMA + BiLSTM + Logit Bias |
| **Phase 4** | Court Judgment Prediction | Macro-F1 (Binary) | **0.6377** | 108 | 12 | Arch4 + LoRMA + Multi-Chunk BiLSTM Head |
| **Phase 5** | Legal Statute Identification | Micro-F1 (116 statutes) | **0.3524** | 87 | 6 | 4-Model Probability Ensemble + Cross-Fitted Thresholds |
| **Phase 6** | Abstractive Summarization | BERTScore | **0.8255** | 108 | 7 | Arch4 + LoRMA + Lead-Tail-SumBasic Greedy |
| **Phase 7** | Prior Case Retrieval | Micro-F1@7 | **0.2735** | 85 | 7 | Arch4 Bi-Encoder MaxSim + BM25 Hybrid Top-7 |

---

## Repository Structure

```text
.
├── code/
│   ├── phase1/
│   │   └── Phase 1.ipynb          # Pretraining Arch 1-4, BPE tokenizer, word generation
│   ├── phase2/
│   │   └── Phase 2.ipynb          # Legal NER: Token classification, Viterbi, post-processing
│   ├── phase3/
│   │   └── Phase 3.ipynb          # Rhetorical Roles: LoRMA adaptation, BiLSTM sentence chunker
│   ├── phase4/
│   │   └── Phase 4.ipynb          # Judgment Prediction: Head+Tail chunking, 2-stage LP-FT
│   ├── phase5/
│   │   └── Phase 5.ipynb          # Statute Identification: Keyword windows, ASL loss, ensemble
│   ├── phase6/
│   │   └── Phase 6.ipynb          # Summarization: SumBasic extraction, prompt caching, decoding
│   └── phase7/
│       └── Phase 7.ipynb          # Retrieval: GradCache contrastive bi-encoder, BM25 hybrid
├── report/
│   └── CS781_CP_Brinda_Fadadu_230307.pdf   # Complete formal project report (PDF)
└── README.md                      # Comprehensive documentation & technical appendix
```

---

## Phase-by-Phase Technical Breakdown

### Phase 1: Foundation Legal Language Model Pretraining

#### 1. Task Formulation & Data
* **Objective**: Train a causal decoder-only language model from scratch on Indian Supreme Court legal texts to model statutory clauses, procedural verbiage, and citation syntax.
* **Corpus & Split**: 35M training tokens ($68{,}359$ non-overlapping 512-token blocks) and 1.5M validation tokens ($2{,}929$ blocks) sampled from `train_supreme_court.json`. Supreme Court texts were selected over High Court or Central Acts due to syntactic stability and clean OCR.
* **Tokenizer**: Byte-level Byte Pair Encoding (ByteBPE) trained on the corpus with a vocabulary size of $8{,}000$ tokens.

#### 2. Architectures Explored
Four architectures were designed, constructed, and pre-trained under identical context windows ($S=512$) to identify the optimal legal foundation backbone:
1. **Architecture 1 (Baseline)**: Sinusoidal positional embeddings, Pre-LayerNorm, standard GELU activation ($d_{\text{model}}=768$, 12 layers, 12 heads; 91.16M params).
2. **Architecture 2 (Modern Standard)**: Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, Pre-RMSNorm, and per-head QK-norm ($d_{\text{model}}=768$, 12 layers, 12 heads; 105.26M params).
3. **Architecture 3 (Efficient Attention)**: Grouped-Query Attention (GQA, 12 query heads, 4 KV heads), learned absolute position embeddings, Pre-LayerNorm ($d_{\text{model}}=768$, 12 layers; 119.41M params).
4. **Architecture 4 (Custom Winning Foundation)**: Attention with Linear Biases (ALiBi), per-head QK-norm ($\text{RMSNorm}(Q), \text{RMSNorm}(K)$), SwiGLU non-linear FFN ($d_{\text{ff}}=2560$), Pre-RMSNorm, and tied input-output embeddings ($d_{\text{model}}=768$, 12 layers, 12 heads; 105.26M params).

#### 3. Key Findings & Metric Comparison
* **Validation Perplexity**:
  * **Arch 4**: **23.66** (Validation loss: 3.164, Training loss: 3.11 after 2,136 steps / 49.7 minutes).
  * **Arch 2**: 120.75 (30M tokens, 1,831 steps).
  * **Arch 1**: 190.77 (60M tokens, 3,662 steps).
  * **Arch 3**: Did not achieve stability; stopped at step 2,800.
* **Why Arch 4 Won**:
  * *QK-Norm*: Bounds query-key dot products to $\le \sqrt{d_{\text{head}}}$, eliminating attention entropy collapse under high learning rates ($4\times 10^{-4}$) on specialized legal syntax.
  * *ALiBi*: Injects relative linear distance penalties ($m \cdot (j - i)$) directly into attention logits, providing an inductive bias for local syntactic dependencies without consuming parameter capacity.
  * *SwiGLU*: Multiplicative bilinear gating acts as a dynamic content filter, suppressing redundant legal boilerplate.
* **Codabench Submission**: Continuation coherence score of **0.8263** (Rank 110).

---

### Phase 2: Legal Named Entity Recognition (NER)

#### 1. Task Formulation & Data
* **Task**: Token-level classification and character-level span boundary reconstruction across 12 legal categories: `COURT`, `PETITIONER`, `RESPONDENT`, `JUDGE`, `LAWYER`, `DATE`, `ORG`, `GPE`, `STATUTE`, `PROVISION`, `PRECEDENT`, `CASE_NUMBER`.
* **Data**: Two folds of 35 documents each ($2{,}845$ sentences, $14{,}820$ entity spans). Severe class imbalance: 90.2% of tokens are background class `O`.

#### 2. Models & Adaptation Techniques Tried
* **Arch 4 + LoRMA (Winning Run)**: Low-Rank Matrix Adaptation ($r=32$, $\alpha=64$, applied to all linear projections) on Arch 4 backbone with a token classification head.
* **Arch 4 Full Fine-Tuning**: Updating all 105.87M parameters with differential learning rates ($5\times 10^{-5}$ backbone, $10^{-3}$ head, $2\times 10^{-5}$ embeddings; 10 epochs).
* **Arch 2 + LoRA**: Additive low-rank update ($r=16$) on RoPE backbone.
* **Arch 3 Distilled**: Student model distilled from four teacher backbones.
* **Arch 1 Full Fine-Tuning**: Baseline full fine-tuning.

#### 3. Post-Processing & Constrained Decoding Pipeline
1. **Bidirectional Attention Mask**: Enabled bidirectional context across the 512-token input sequence during token classification.
2. **Constrained BIO Viterbi Decoding**: Enforces grammatical BIO transition rules, strictly forbidding illegal transitions (e.g., $O \to I\text{-TAG}$ or $B\text{-COURT} \to I\text{-DATE}$).
3. **Class-Calibrated Logit Offsets**: Adjusted logit biases on validation data ($O$ bias: 2.5, continuation bonus: 3.0, merge gap: 1 token).
4. **Boundary Punctuation Cleanup**: Stripped leading/trailing brackets, quotes, and punctuation artifacts from reconstructed character spans.
5. **Document-Level Entity Consistency**: Enforced majority voting across identical surface strings within the same judgment to eliminate missed mentions.

#### 4. Results & Ablations
| Configuration | Validation Macro-F1 | Codabench Score | Rank |
|:---|:---:|:---:|:---:|
| **Arch 4 + LoRMA (Winning)** | 0.3950 | **0.4145** | **101** |
| Arch 4 Full Fine-Tuning | 0.4007 | 0.3900 | 114 |
| Logit Ensemble (LoRMA + FT) | 0.4020 | 0.4000 | 108 |
| Arch 2 + LoRA | 0.2850 | 0.2200 | 130 |
| Arch 3 KD (4 Teachers) | 0.2100 | 0.1764 | 138 |
| Arch 1 Full Fine-Tuning | 0.1900 | 0.1578 | 142 |

* **Failed Attempts & Failure Modes**:
  * *Actor Class Ambiguity*: Procedural roles (`PETITIONER`, `RESPONDENT`, `LAWYER`) achieved low F1 (0.085–0.152) because plain person names lack distinct semantic identifiers outside of intro clauses.
  * *Overfitting in Full Fine-Tuning*: Full fine-tuning over-fitted rapidly after epoch 4 (validation loss degraded while training loss dropped from 0.0202 to 0.0021), whereas LoRMA regularized the update space and won on test data.
  * *Ensembling Degradation*: Ensembling LoRMA and Full Fine-Tuning degraded the Codabench score from 0.4145 to 0.4000 due to correlated false-positive noise.

---

### Phase 3: Rhetorical Role Prediction (RR)

#### 1. Task Formulation & Data
* **Task**: Sentence-level functional classification assigning each sentence of a court case to one of 13 canonical rhetorical roles: `PREAMBLE`, `FACTS`, `ISSUE`, `ARG_PETITIONER`, `ARG_RESPONDENT`, `ANALYSIS`, `STATUTE`, `PRECEDENT_RELIED`, `PRECEDENT_NOT_RELIED`, `RATIO`, `RULING`, `NONE`.
* **Corpus**: 80 training documents ($10{,}482$ sentences), 10 dev documents ($1{,}248$ sentences), and 10 test documents across Corporate Law (CL) and Information Technology (IT) domains.

#### 2. Architectural Design & Hierarchical Modeling
* **Backbone**: Phase-1 Architecture 4 with LoRMA ($r_{\text{attn}}=32, r_{\text{ffn}}=16, \alpha=32$; 11.4M / 116.6M trainable params, 9.75%).
* **Context Modeling**: Each sentence was encoded with a sliding $\pm 2$ sentence neighbor context window.
* **Document-Level Sequential Head**: A 2-layer Bidirectional LSTM ($d_{\text{hidden}}=256$) processed the sequence of sentence vectors across the entire judicial narrative to capture long-range structural dependencies (e.g., `PREAMBLE` $\to$ `FACTS` $\to$ `ARGUMENTS` $\to$ `ANALYSIS` $\to$ `RULING`).
* **Loss Function**: Class-weighted, label-smoothed ($\epsilon=0.05$) cross-entropy:
  $$\mathcal{L} = -\sum_{c} w_c y_c \log \hat{y}_c, \quad w_c = \left(\frac{1}{N_c}\right)^{0.7}$$

#### 3. Results & Ablations
| Model / Configuration | Trainable Params | Dev Macro-F1 | Codabench Macro-F1 | Rank |
|:---|:---:|:---:|:---:|:---:|
| **Arch 4 + LoRMA + BiLSTM (Winning)** | **11.4M** | **0.4122** | **0.4122** | **118** |
| Arch 2 + LoRA + BiLSTM | 8.2M | 0.3910 | 0.4000 | 125 |
| Arch 3 + Prefix Tuning | 26.5M | 0.1643 | 0.3783 | 129 |
| Arch 1 + Linear Probe | 2.1M | 0.3320 | 0.3486 | 135 |

* **Failed Attempts & Debugging**:
  * *Label Dictionary Inversion*: In early iterations of Run 2 (Arch 3), the evaluation pipeline inverted the string-to-ID vocabulary mapping, causing a severe score crash before the pipeline was debugged and corrected.
  * *Ablation on Logit Bias Calibration*: Direct argmax decoding produced 0.3840 Macro-F1; applying coordinate ascent to tune additive per-class logit biases on dev improved rare-class recall and raised Macro-F1 to 0.4122.

---

### Phase 4: Court Judgment Prediction (CJPE)

#### 1. Task Formulation & Data
* **Task**: Binary classification predicting whether an appeal before the Supreme Court is accepted ($y=1$) or dismissed ($y=0$) from raw textual records.
* **Corpus**: $37{,}387$ training judgments, $1{,}988$ dev judgments, and $1{,}517$ test judgments. Severe document length: legal judgments exceed $5{,}000$ words, far beyond standard transformer context windows.

#### 2. Multi-Chunk Hierarchical Architecture
* **Chunking Strategy**: Extracted $4 \times 512$ token chunks per document: **1 Head chunk** (procedural context, lower court holdings) + **3 Tail chunks** (final arguments, ratio decidendi, operative decree).
* **Multi-Layer Feature Pooling**: Extracted and concatenated RMS-normalized mean-pooled hidden states from layers 3, 6, and 9 of Arch 4, capturing low-level syntactic markers, intermediate entity patterns, and high-level legal semantic states.
* **Document Head**: Hierarchical BiLSTM across the 4 chunk vectors followed by self-attention pooling and an MLP classifier.

#### 3. Two-Stage Training Protocol (LP-FT)
* **Stage A (Linear Probe)**: Froze the Arch 4 backbone and trained only the hierarchical document head on pre-cached chunk features for 30 epochs.
* **Stage B (LoRMA End-to-End Fine-Tuning)**: Injected LoRMA ($r_{\text{attn}}=16, r_{\text{ffn}}=8$) initialized at zero ($B=0$, starting identically at the Stage-A optimum) and fine-tuned jointly with differential learning rates ($2\times 10^{-4}$ adapter, $10^{-4}$ head) under strict GPU time budgets.

#### 4. Results & Ablations
| Run / System | Description | Dev Macro-F1 | Codabench Score |
|:---|:---|:---:|:---:|
| **Run 1 (Winning)** | **Arch 4 + LoRMA + Quantile Thresholding** | **0.6465** | **0.6377** |
| Run 1 (Ablation) | Prior-matched thresholding | 0.6310 | 0.6210 |
| Run 1 (Re-run) | Default 0.5 threshold | 0.6180 | 0.6102 |
| Run 1 (Early) | Direct end-to-end (no Stage A) | 0.4850 | 0.3344 |
| Master Ensemble | Rank-averaged fusion of Runs 1–3 + KD | 0.6568 | 0.6279 |
| Run 2 | Arch 3 + AdaLoRA | 0.5840 | 0.5579 |
| Run 3 | Arch 2 + Layer-wise Fine-Tuning | 0.5790 | 0.5612 |
| Run 4 | Knowledge Distillation (Arch 4 $\to$ Arch 1) | 0.5120 | Failed / Incomplete |

* **Failed Attempts & Critical Insights**:
  * *The Prior-Shift Ensemble Trap*: The multi-architecture ensemble gained +0.010 Macro-F1 on dev (0.6465 to 0.6568) but dropped -0.010 on Codabench test (0.6377 to 0.6279). Root cause: the dev set had an artificial 50/50 class balance, while the test population had a 59/41 skew; quantile calibration over-predicted positive outcomes on test.
  * *Distillation Capacity Failure*: Attempting to distill Arch 4 into a half-depth Arch 1 student failed to converge because the compressed student lacked capacity to retain multi-chunk sequence dependencies.
  * *Frozen Linear Probe Baseline*: Training a linear probe on frozen Arch 4 representations yielded an AUC of 0.517 (near random chance), proving that adaptation of the backbone weights is strictly necessary for legal reasoning.

---

### Phase 5: Legal Statute Identification (LSI)

#### 1. Task Formulation & Data
* **Task**: Multi-label classification predicting which of 116 target statutory provisions (e.g., IPC Section 302, CrPC Section 482, Article 226) were violated or applied.
* **Corpus**: $25{,}487$ training judgments, $2{,}000$ validation judgments, and $1{,}500$ test judgments. Severe label sparsity: median of 2.1 positive statutes per case out of 116 possibilities.

#### 2. Architectural Design & Training
* **Context Selection via Keyword Density**: Sliding windows ($L=512$, overlap 128) scored by legal regex markers (`section \d+`, `u/s`, `IPC`, `Article \d+`, `punishable`); the top 6 keyword-dense windows were selected.
* **Label-Wise Attention Head**: Learned 116 query vectors (one per statute, $d=384$) attending across the selected chunk sequence, allowing each statute to query the specific passages referencing it.
* **Loss Function**: Asymmetric Loss (ASL) with $\gamma_-=4, \gamma_+=0$, heavily penalizing false positives while preventing easy negative statutory provisions from overwhelming gradients.

#### 3. Results & Ensembling Strategy
Phase 5 was the one phase where **multi-architecture ensembling triumphed**:
$$\bar{P}_l = \frac{1}{4} \sum_{m=1}^{4} \sigma(z_m^{(l)})$$
* **Models Ensembled**:
  1. *Run 1*: Arch 4 + LoRMA ($r=8, \alpha=16$)
  2. *Run 2*: Arch 3 + Houlsby Bottleneck Adapters ($d_{\text{bottleneck}}=64$)
  3. *Run 3*: Arch 2 + Layer-wise Fine-Tuning (top 4 blocks)
  4. *Run 4*: Arch 1 Half-Depth Student distilled from Run 1
* **Decision Rule**: 2-fold cross-fitted per-statute thresholds $\tau_l$ with an argmax fallback $\arg\max_l (\bar{P}_l - \tau_l)$ to guarantee every document receives at least one statute.
* **Official Codabench Score**: **0.3524 Micro-F1** (Rank 87).

---

### Phase 6: Abstractive Summarization

#### 1. Task Formulation & Data
* **Task**: Generate concise, legally accurate abstractive summaries (headnotes) from full-length court judgments.
* **Corpus**: $6{,}878$ training documents (150 held-out dev) and 100 test documents. Median source text: $3{,}135$ words; median summary: 638 words ($4.91\times$ compression ratio).

#### 2. Architecture & Long-Document Prompt Caching
* **Unified Prompt Layout**: Left-padded source tokens (budget 315) + `[SEP]` + target summary tokens (budget 192) + `[END]`.
* **Lead-Tail-SumBasic Extractive Selection**: Populated the 315 source tokens by selecting the opening procedural paragraphs, operative concluding orders, and intermediate sentences selected via greedy SumBasic with redundancy penalties.
* **Pre-Tokenization Finding**: Testing four tokenization schemes revealed that preserving punctuation and casing gave a validation perplexity of **29.7**, whereas lowercasing (46.1) or punctuation splitting (60.3) ruined legal citation structure.
* **Inference Decoding**: Autoregressive greedy decoding with **no-repeat 3-gram blocking** and a 30% minimum length constraint.

#### 3. Results & Metric Divergence
| Model / System | Trainable Params | Dev ROUGE-L | Codabench BERTScore | Rank |
|:---|:---:|:---:|:---:|:---:|
| **Run 1: Arch 4 + LoRMA (Winning)** | **3.44M** | 0.1778 | **0.8255** | **108** |
| Run 3: Arch 2 + LoRA | 3.44M | **0.1825** | 0.8136 | 115 |
| Run 4: Arch 1 Distilled Student | 48.7M | 0.1712 | 0.8128 | 117 |
| Run 2: Arch 3 + Houlsby | 4.12M | 0.1690 | 0.7830 | 124 |
| Logit Ensemble (Runs 1–3) | — | 0.1795 | 0.8205 | 111 |
| Minimum Bayes Risk (MBR) | — | 0.1780 | 0.8129 | 116 |

* **Failed Attempts & Metric Divergence**:
  * *ROUGE vs. BERTScore Conflict*: Local ROUGE-L preferred Arch 2 (0.1825 vs. 0.1778), but official Codabench BERTScore ranked Arch 4 decisively higher (0.8255 vs. 0.8136). ROUGE over-rewarded verbatim n-gram copying, whereas BERTScore rewarded legal semantic fidelity.
  * *Extractive Baseline Paradox*: Simple extractive sentence selection scored higher ROUGE-L than 108M generative models because Indian legal headnotes frequently reproduce passages verbatim.

---

### Phase 7: Prior Case Retrieval (PCR)

#### 1. Task Formulation & Data
* **Task**: Given a query case narrative, retrieve the top 7 legally relevant precedent cases from candidate pools of 150–400 judgments.
* **Corpus**: $2{,}872$ training queries, 300 dev queries, and 300 test queries. Evaluated by **Micro-F1@7**.

#### 2. Architecture & Training Pipeline
* **Dense Bi-Encoder**: Siamese encoder built on Arch 4. Documents were split into 512-token chunks, encoded into token-level representations, and mean-pooled.
* **MaxSim Multi-Vector Aggregation**: Measured query-candidate similarity via late-interaction MaxSim:
  $$\text{Sim}(Q, D) = \frac{1}{|Q|} \sum_{q \in Q} \max_{d \in D} \left(\frac{q \cdot d}{\|q\| \|d\|}\right)$$
* **GradCache Chunk-Contrastive Learning**: InfoNCE loss ($\tau=0.05$) with 3 BM25 hard negatives per query. Enabled large batch contrastive learning ($B=16$ queries, 64 total documents) on a single 16GB GPU without OOM.
* **Hybrid Score Fusion**: Combined z-score normalized dense similarity with z-score normalized Okapi BM25 lexical similarity:
  $$S_{\text{hybrid}} = z(S_{\text{dense}}) + \lambda \cdot z(S_{\text{BM25}}), \quad \lambda = 0.35$$

#### 3. Results & Ablations
| Configuration | Dense Dev F1@7 | Hybrid Dev F1@7 | Codabench Score |
|:---|:---:|:---:|:---:|
| **Run 1: Arch 4 + LoRMA + BM25 Hybrid (Winning)** | 0.2380 | **0.2742** | **0.2735** |
| Run 4: Arch 1 Student + BM25 Hybrid | 0.2180 | 0.2540 | 0.2534 |
| Multi-Run Ensemble + BM25 | 0.2290 | 0.2450 | 0.2434 |
| Run 3: Arch 2 Partial FT + BM25 | 0.1980 | 0.2210 | 0.2191 |
| Run 2: Arch 3 IA3 + BM25 | 0.1910 | 0.2150 | 0.2138 |
| Standalone BM25 Baseline | — | 0.2110 | 0.2105 |

* **Failed Attempts & Key Findings**:
  * *Dense-Only vs. Hybrid*: Standalone dense retrieval reached 0.2380, missing cases matching specific statute numbers; adding BM25 provided an immediate +0.035 boost to 0.2735.
  * *Random vs. Hard Negatives*: Training with in-batch random negatives yielded dev F1 of only 0.1840; injecting BM25 hard negatives forced the encoder to discriminate between legally substantive precedents and superficially similar cases.

---

## Comprehensive Hyperparameter Reference

| Hyperparameter | Phase 1 (Pretrain) | Phase 2 (NER) | Phase 3 (Roles) | Phase 4 (CJPE) | Phase 5 (Statutes) | Phase 6 (Summary) | Phase 7 (Retrieval) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base Architecture** | Arch 4 | Arch 4 | Arch 4 | Arch 4 | Ensemble (1–4) | Arch 4 | Arch 4 |
| **Total Parameters** | 105.3M | 112.7M | 116.6M | 108.5M | 110.4M | 108.7M | 105.3M |
| **Trainable Parameters** | 105.3M | 2.72M | 11.4M | 3.24M | 5.12M | 3.44M | 3.44M |
| **Adaptation Technique** | From scratch | LoRMA | LoRMA | LoRMA | LoRMA / Adapters | LoRMA | LoRMA |
| **Adapter Rank ($r$) / Dim** | — | $r=32$ | $r_a=32, r_f=16$ | $r_a=16, r_f=8$ | $r=8$ | $r=16$ | $r=16$ |
| **Scaling Factor ($\alpha$)** | — | $\alpha=64$ | $\alpha=32$ | $\alpha=32$ | $\alpha=16$ | $\alpha=32$ | $\alpha=32$ |
| **Context Layout** | $1\times 512$ | $1\times 512$ | $1\times 512$ | $4\times 512$ | $6\times 512$ | $1\times 512$ | $4\times 512$ |
| **Peak Learning Rate** | $4\times 10^{-4}$ | $3\times 10^{-4}$ | $2.5\times 10^{-4}$ | $2\times 10^{-4}$ | $5\times 10^{-4}$ | $6\times 10^{-4}$ | $3\times 10^{-4}$ |
| **Effective Batch Size** | 32 ($4\times 8$) | 16 ($4\times 4$) | 20 ($4\times 5$) | 16 ($4\times 4$) | 32 ($8\times 4$) | 16 ($4\times 4$) | 16 (GradCache) |
| **Epochs / Budget** | 1 (2,136 steps) | 16 epochs | 20 epochs | 30 ep (A) + budget | 120 min budget | 40 min budget | 60 min budget |
| **Optimizer & Weight Decay** | AdamW (0.05) | AdamW (0.01) | AdamW (0.05) | AdamW (0.05) | AdamW (0.05) | AdamW (0.05) | AdamW (0.05) |
| **Loss Function** | Cross-Entropy | Token CE | Weighted CE | Binary CE | Asymmetric (ASL) | Causal CE | InfoNCE ($\tau=0.05$) |
| **Decoding / Post-Proc.** | Nucleus ($p=0.9$) | Viterbi BIO | Dev Logit Bias | Quantile Thresh. | Per-Statute Thresh. | Greedy (3-gram block)| BM25 Hybrid Top-7 |

---

## Detailed Appendix

### Appendix A: Mathematical Formulations of Foundation Architectures

#### 1. Architecture 4 (Winning Backbone)
* **Attention with Linear Biases (ALiBi)**: For head $i$ and positions $j, k$:
  $$\text{AttnBias}_{i,j,k} = -m_i \cdot |j - k|, \quad m_i = 2^{-\frac{8i}{H}}$$
* **Per-Head QK-Normalization**:
  $$\tilde{Q} = \text{RMSNorm}(Q), \quad \tilde{K} = \text{RMSNorm}(K)$$
  $$\text{Attention}(\tilde{Q}, \tilde{K}, V) = \text{Softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d_{\text{head}}}} + \text{AttnBias} + M_{\text{causal}}\right)V$$
* **SwiGLU Feed-Forward Network**:
  $$\text{SwiGLU}(x) = \left(\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}\right) W_{\text{down}}$$

#### 2. Architecture 2 (RoPE + SwiGLU)
* Applies 2D Givens rotation matrices to query and key vector pairs at index $m$:
  $$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, \dots, R_{\theta_{d/2}, m}\right)$$

#### 3. Architecture 3 (Grouped-Query Attention)
* Groups 12 query heads into 4 key-value heads, repeating KV heads across 3 query heads each:
  $$\text{GQA}(Q, K, V) = \text{Concat}\left(\text{head}_1, \dots, \text{head}_{12}\right)W^O$$

---

### Appendix B: PEFT Mathematical Formulations

#### 1. Low-Rank Matrix Adaptation (LoRMA)
LoRMA applies a multiplicative rank-stabilized low-rank update with a learned scalar gate $s$:
$$W' = W_0 + s \cdot \left(\frac{\alpha}{\sqrt{r}}\right) B A$$
where $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is frozen, $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, and $B=0$ at initialization, guaranteeing that the model starts identically at the pretrained checkpoint.

#### 2. Houlsby Bottleneck Adapters
Inserted after self-attention and FFN sub-layers:
$$h' = h + f(h W_{\text{down}}) W_{\text{up}}$$
where $W_{\text{down}} \in \mathbb{R}^{d \times m}$, $W_{\text{up}} \in \mathbb{R}^{m \times d}$ ($m \ll d$).

#### 3. $(IA)^3$
Learned coordinate-wise scaling vectors $l_k, l_v, l_{\text{ffn}}$ that multiply activations directly:
$$\tilde{K} = l_k \odot K, \quad \tilde{V} = l_v \odot V, \quad \tilde{\text{FFN}}(x) = \left(l_{\text{ffn}} \odot \sigma(x W_1)\right) W_2$$

---

### Appendix C: Context Selection & Extraction Heuristics

#### 1. Head + Tail Chunking (Phase 4)
Legal judgments state procedural context and lower court orders in the opening paragraphs, and operative decrees in the concluding paragraphs. We allocate:
$$\text{Tokens} = \text{Head}(1 \times 512) \cup \text{Tail}(3 \times 512)$$

#### 2. Keyword-Dense Statutory Windows (Phase 5)
Given candidate windows $\{W_1, \dots, W_k\}$, score each by regex matching density:
$$\text{Score}(W_i) = \sum_{r \in \text{Regex}} c_r \cdot \text{Count}(r, W_i)$$
The top 6 non-overlapping windows with the highest scores are selected.

#### 3. Lead-Tail-SumBasic Extractive Selector (Phase 6 & 7)
* Allocate 2 lead sentences and 2 tail sentences.
* For the remaining budget, iteratively select sentence $S$ maximizing word probabilities:
  $$\text{Score}(S) = \frac{1}{|S|} \sum_{w \in S} P(w)$$
* Update probabilities of words in selected sentences via squaring: $P(w) \leftarrow P(w)^2$ to eliminate redundancy.

---

### Appendix D: Detailed Failure Modes & Post-Mortem Analysis

1. **Failure Mode 1: The Dev-to-Test Prior Shift in Ensembling (Phase 4)**
   * *Symptom*: Ensembling 4 models improved dev Macro-F1 from 0.6465 to 0.6568 (+0.010), but degraded Codabench test Macro-F1 from 0.6377 to 0.6279 (-0.010).
   * *Root Cause*: The local validation set was artificially balanced (50% accepted, 50% dismissed), whereas the real test distribution was 59% dismissed, 41% accepted. Quantile thresholding calibrated to dev priors over-predicted the minority positive class on test data.
   * *Resolution*: Standalone Arch 4 with conservative probability thresholding generalized significantly better than multi-model rank averaging.

2. **Failure Mode 2: Inverted Vocabulary Crash (Phase 3)**
   * *Symptom*: Run 2 (Arch 3 prefix tuning) produced near-zero dev macro-F1 (0.012).
   * *Root Cause*: The evaluation script inverted the label dictionary keys and values (`id2label` mapped back into `id2label` instead of `label2id`).
   * *Resolution*: Reconstructed dictionary serialization with bidirectional assertions, restoring performance to 0.3783.

3. **Failure Mode 3: Student Capacity Collapse in Distillation (Phase 4)**
   * *Symptom*: Distilling Arch 4 into a 6-layer Arch 1 student caused loss divergence and NaN gradients during multi-chunk backpropagation.
   * *Root Cause*: The 6-layer student lacked the parameter depth required to preserve 4 disparate 512-token chunk trajectories simultaneously.
   * *Resolution*: Knowledge distillation was restricted to single-chunk extraction tasks (Phase 5 and Phase 6), where student models performed reliably.

4. **Failure Mode 4: Tokenizer Merge Mismatch Exploding Perplexity (Phase 6)**
   * *Symptom*: Pre-tokenization variants caused validation perplexity on identical text to skyrocket from 29.7 to 78.8.
   * *Root Cause*: Aggressive lowercasing or punctuation stripping fractured legal citation strings (`(2021) 4 SCC 12`) into isolated byte sequences that were absent from ByteBPE merges.
   * *Resolution*: Retained exact whitespace-split punctuation preservation across all generation prompts.

---

### Appendix E: Environment Setup & Reproducibility Guide

#### 1. Hardware & Software Requirements
* **GPU**: 1x NVIDIA Tesla T4 (16GB VRAM) or NVIDIA RTX 3090/4090.
* **OS**: Linux (Ubuntu 20.04/22.04) or Windows 10/11 with CUDA 12.x.
* **Python**: 3.10+
* **Core Libraries**:
  ```bash
  pip install torch==2.4.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
  pip install transformers==4.44.0 datasets==2.21.0 evaluate==0.4.2 scikit-learn==1.5.1
  pip install huggingface-hub rank-bm25 bert-score rouge-score
  ```

#### 2. Executing the Notebooks
Each notebook is completely self-contained and pre-configured for Kaggle GPU environments:
1. Clone this repository.
2. Ensure Kaggle secrets or environment variables provide `HF_TOKEN` with access to `Exploration-Lab/CS-781-Capstone`.
3. Open any notebook (`code/phase1/Phase 1.ipynb` through `code/phase7/Phase 7.ipynb`) in JupyterLab or Kaggle and select the **GPU T4 x2** accelerator.
4. Run all cells sequentially. The notebooks automatically handle data download, checkpoint resolution, adapter injection, evaluation, and CodaBench submission ZIP packaging.
