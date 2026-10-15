# Urdu Sarcasm Detection 

Binary classifier for Urdu social-media posts: sarcastic (1) vs non-sarcastic (0).
Notebook: `urdu_sarcasm_pipeline_v4.ipynb`

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Weighted ensemble | 0.834 | **0.830** |
| TF-IDF + phrases + handcrafted (LogReg) | 0.831 | 0.826 |
| XLM-RoBERTa-large (fine-tuned) | 0.830 | 0.825 |
| BiLSTM-attn + char-CNN + features | 0.829 | 0.825 |
| TF-IDF LogReg (BPE4k + char) | 0.827 | 0.821 |

## Run on Google Colab

1. **Open the notebook** in Colab (File → Upload notebook, or open it from Drive/GitHub).
2. **Turn on the GPU:** Runtime → Change runtime type → **T4 GPU**.
   Without a GPU the XLM-R cells are skipped automatically.
3. **Put two files in your Google Drive** under `MyDrive/`:
   - `Copy of urdu_sarcastic_dataset.csv` — the dataset (columns `urdu_text`, `is_sarcastic`)
   - `NotoNastaliqUrdu-VariableFont_wght.ttf` — Urdu font for plot labels
     ([Google Fonts: Noto Nastaliq Urdu](https://fonts.google.com/noto/specimen/Noto+Nastaliq+Urdu))

   Different names or folders? Edit `DATA_PATH` and `FONT_PATH` in the setup cells.
4. **Run all:** Runtime → Run all. Approve the Google Drive access prompt.
   - The pip cell installs everything in `requirements.txt` that Colab does not already have
     (`pip install -r requirements.txt` works too if you upload the file).
   - The Urdu stopword list downloads automatically through `kagglehub`
     (`itsnobita/urdu-stopwords`); no Kaggle login is needed on Colab.
   - `xlm-roberta-large` (~2.2 GB) downloads from Hugging Face on first run.
     If the T4 runs out of memory or time, set `TF_MODEL = "xlm-roberta-base"`.
5. **Outputs** are written to `/content/sarcasm_artifacts/`:
   `results.csv`, `error_analysis.csv`, `fe_lr.joblib`, `tfidf_lr.joblib`,
   `bilstm_charcnn.pt`, `nn_vocab.json`.
   Colab deletes `/content` when the session ends, so copy them to Drive:
   `!cp -r /content/sarcasm_artifacts /content/drive/MyDrive/`

## Notebook structure

1. Load data → drop missing labels, duplicates and contradictory duplicates
2. Surface cues, emoji and hashtag analysis, word distribution (EDA)
3. Urdu-aware normalization (letter unification, cue tokens, leak-hashtag removal)
4. Stratified 70/15/15 split (seed 42)
5. TF-IDF ablation (6 text variants × word / char / BPE features)
6. Classical models: LogReg, LinearSVM, ComplementNB
7. Bonus: PMI phrases + handcrafted pragmatic features
8. BiLSTM-attention + char-CNN (PyTorch, FastText-CBOW embeddings)
9. XLM-RoBERTa-large fine-tuning
10. Weighted ensemble, results table, confusion matrices, McNemar test
11. Error analysis

## Notes

- Hashtags that state the label (`#sarcasm`, `#طنز`, `#sarcastic`, `#satire`, `#مزاح` …) are removed from
  model inputs to avoid leakage (`DROP_LEAK_TAGS = True`).
- All vectorizers, tokenizers, vocabularies and scalers are fit on the training split only;
  thresholds and ensemble weights are tuned on validation only.

## Acknowledgements

scikit-learn, PyTorch, Hugging Face `transformers`/`tokenizers`, gensim, `emoji`, `arabic_reshaper`,
`python-bidi`; pretrained `xlm-roberta-large` (Conneau et al., 2020); Urdu stopwords from Kaggle
`itsnobita/urdu-stopwords`. Parts of the code and documentation were drafted with AI assistance
(Claude, Anthropic) and reviewed by the author.
