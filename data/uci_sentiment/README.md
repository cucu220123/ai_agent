# UCI Sentiment Labelled Sentences

Source: [UCI dataset 331](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).
Citation: Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. https://doi.org/10.24432/C57604.
License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Associated paper: Kotzias, Denil, de Freitas and Smyth, *From Group to Individual Labels Using Deep Features*, KDD 2015.

The original dataset contains 3000 sentences from product, film and restaurant reviews. Labels are negative (0) or positive (1). The three original labeled text files are retained under `raw/` with attribution. No source-domain column is supplied as a model feature.

Our modifications: parse each record at its final tab (do not interpret embedded quotes as CSV syntax); remove NFKC/casefold/whitespace duplicate groups and conflicting-label groups; stratify by original source and class, with fixed seed 20260908. `development.csv` is the only dataset supplied to agents. `final_test.csv` is used once after code selection; its results are not available to repair agents. Per-record source file and line remain in the provenance JSON files. Exact counts, hashes and split policy are in `manifest.json`.

Recreate using `python scripts/prepare_sentiment_data.py` from the repository root. The downloaded archive must match the pinned SHA256. The original 16-row text demo remains a separate smoke test; these records do not overwrite its historical evidence.
