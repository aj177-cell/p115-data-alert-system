# Literature grounding for the p115 pilot (appendix to v3)

This file is the literature appendix for the merged repo. It maps the four papers in [research.md](research.md) onto the batch checks. Pilot counts are in [results/flags_by_check.csv](results/flags_by_check.csv) and [results/run_index.csv](results/run_index.csv). A flag is a candidate for verification, not a confirmed data error.

**Document role.** This note adds academic grounding and Level-1 caveats to the existing pilot methodology. It does **not** change detector code, thresholds, data, or the flag counts in `pilot_results_v3.pdf`. Pages 1–4 of the integrated PDF are the original v3 report.

**Provenance of the literature list.** The four academic items below appear under “Literature Being Reviewed for the Next Milestone” in the team’s public Milestone 1 notes:

- live path: `https://github.com/yuyangan/p115-data-alert-system/blob/main/docs/research.md`
- repo path: `docs/research.md` (commit from 2026-10-30)

**Pilot scope reminder (unchanged).** Seven core checks: crossed books, extreme prices, timestamp inversions, missing intervals, stale quotes, unusual spreads, sequence discontinuities; vendor flags is an extra feed-condition check. Results cover 95 symbol-days of Level-1 best bid/ask plus trades (stocks `XNAS.ITCH`, futures `GLBX.MDP3`). The missing-interval rule is

$$g_t > \min\big(\max(-\ln\alpha/\hat\lambda_t,\ 5\,\mathrm{s}),\ 60\,\mathrm{s}\big).$$

About 97% recall is for **injected** outages only, not claimed accuracy on real errors. Medium missing flags fell about 73% versus v2 on the calibration sample. Candidate events (including CL 2026-10-08 ~12:17 ET) remain **unconfirmed**. A later exact-team-notebook rerun matched all v3 counts except vendor flags 14→15 (one ES Aug 30 Level-1 record differed at source, bitflags 128→168, including a bad-receive-time quality flag).

---

## 1. Background

Ultra-high-frequency market data routinely contain recording artefacts as well as genuine market moves. Brownlees and Gallo (2006) argue, in the published abstract, that cleaning and screening of ultra-high-frequency trades and quotes matter: standard filters remove bad records, and failure to purge “wrong” ticks can distort durations and ACD estimates (full journal PDF not retrieved here). Cont, Kukanov and Stoikov (*Journal of Financial Econometrics* 12(1), Winter 2014, 47–88; arXiv preprint Mar 2011) and Gould and Bonart (2015 preprint) study **order-flow / queue dynamics at the best quotes**—useful context for what Level-1 looks like when the market is behaving normally, but not proofs that a flagged record is bad data. Barndorff-Nielsen and Shephard (2006 journal; 2004 Nuffield working paper read here) formalize **jump tests** that separate discontinuous moves from continuous paths; those tests are different from a simple MAD spike screen.

## 2. Mapping our seven checks to the literature

| Check | Support type | What the literature actually supports | What we do **not** claim |
| --- | --- | --- | --- |
| Extreme prices | Direct (screening idea) / contextual (jumps) | Brownlees & Gallo (abstract): high-frequency screening and removal of bad/outlier ticks. Barndorff-Nielsen & Shephard: genuine jumps exist and often coincide with macro news (abstract; WP §6). | Our \(k=12\) MAD rule, session gate, and floors are **project-tuned choices**, not taken from either paper’s statistic; injected-outage recall evaluates missing intervals only. |
| Unusual spreads | Contextual / engineering | Cont et al. Appendix A remove extreme spreads (95th percentile filter) before studying Level-1 impact; wide spreads also appear in thin books. FOMC-wide spreads in the pilot may reflect genuine market behavior rather than errors. | No paper sets our spread-tick threshold; no episode is classified as confirmed error or confirmed benign without verification. |
| Crossed books | Engineering / vendor + microstructure caution | Team `docs/research.md` treats `bid>ask` as a warning to review, not as executable arbitrage; Cont et al. work with filtered positive bid/ask Level-1 quotes. | A cross is a review signal, not confirmed corruption. |
| Timestamp inversions | Contextual / vendor | Team research notes (citing Databento) state that `ts_event` can be non-monotonic due to gateway clocks; a reversal alone is not transmission failure. | Not a formal result in the four papers. |
| Missing intervals | Engineering (rate-adaptive outage screen) | Brownlees & Gallo (abstract) stress cleaning of irregular ultra-high-frequency records; they do not supply our Poisson/\(g_{\min}\) rule. | \(\alpha\), \(g_{\min}=5\)s, 60s cap, and ~97% **injected** recall are pilot choices. |
| Stale quotes | Engineering | Thin-name inactivity is a practical data-quality concern; not a named formal test in these four papers. | Thresholds remain project choices. |
| Sequence discontinuities | Vendor / feed engineering | On symbol-filtered channel pulls, per-instrument sequence steps need not equal 1 (v3 §2/§6); such gaps are not by themselves evidence of missing records—source channel semantics matter. Regressions remain review candidates. | Outside the academic quartet; gaps are not treated as proof of lost data. |
| Vendor flags (extra) | Vendor | Vendor-provided quality flags carried on market-data records (e.g. record bitflags), separate from status/definition metadata. The notebook rerun’s extra flag came from one ES Level-1 record whose bitflags changed 128→168, including a bad-receive-time flag. | Not academic; not sourced from status/definition schemas. |

**Cont (contextual, not a DQ proof).** Cont et al. show short-horizon mid-price changes relate linearly to order-flow imbalance at the best quotes (journal abstract; arXiv §3.1 Level-1 TAQ; Appendix A). That explains why large Level-1 moves can be **real** liquidity events. It does **not** justify adding OFI prediction features to the data-quality detector.

**Gould & Bonart (contextual, do not add as a DQ feature).** They define queue imbalance \(I(t)=(n^b-n^a)/(n^b+n^a)\) and show it predicts the **direction** of the next mid move (abstract; §3). Useful for understanding best-quote pressure; **not** evidence a record is erroneous. We deliberately do **not** add imbalance predictors to the DQ system.

## 3. Current implementation vs optional future work

**Current (unchanged by this appendix).** Rule-based Level-1 screens with project-tuned thresholds; missing intervals use a causal trailing Poisson rate with a 5 s floor; validation includes constructed tests and injected-outage recall on real symbol-days.

**Optional future (from team research notes / literature, not implemented here).** Past-only robust move / size / spread baselines; formal jump statistics (bipower-style) as a **second** layer after cleaning; richer cleaning diagnostics in the spirit of Brownlees & Gallo; dashboard/event replay. Queue-imbalance **prediction** remains out of scope for data-quality detection.

## 4. Level-1 assumptions and caveats

1. Best bid/ask + trades only—no full-book depth (Cont et al. arXiv §3.1 likewise use Level-1 TAQ for accessibility).
2. Single-venue publisher streams (`XNAS.ITCH`, `GLBX.MDP3`), not consolidated NBBO construction as in Cont’s appendix.
3. Flags are review candidates. No check in the v3 pilot has a **confirmed** vendor error; CL 2026-10-08 ~12:17 ET remains the leading candidate pending verification.
4. Macroeconomic / thin-book episodes (e.g. FOMC wide spreads; ADTN/PLAB/LWAY gaps) may reflect genuine market behavior rather than errors—consistent with BNS’s observation that many detected jumps align with announcements—but are not classified here without verification.
5. Literature does **not** validate numerical thresholds, the tuned Poisson missing model, or claimed precision on real errors. Injected-outage recall evaluates missing intervals only.
6. Source-access notes: Cont arXiv PDF, Gould arXiv PDF, and the 2004 Nuffield working-paper PDF of Barndorff-Nielsen & Shephard were read in full. Brownlees & Gallo (2006) journal PDF was **not** publicly downloadable here (Flore record closed); claims for that paper are limited to the published abstract/record.

## References

1. C. T. Brownlees and G. M. Gallo (2006). Financial econometric analysis at ultra-high frequency: Data handling concerns. *Computational Statistics & Data Analysis* 51, 2232–2245. https://doi.org/10.1016/j.csda.2006.09.030 — record also at https://flore.unifi.it/handle/2158/210321. Full journal PDF not retrieved; abstract-level citation only.
2. R. Cont, A. Kukanov and S. Stoikov (2014). The price impact of order book events. *Journal of Financial Econometrics* 12(1), Winter 2014, 47–88 (published online 7 June 2013). https://doi.org/10.1093/jjfinec/nbt003. Archived preprint: arXiv:1011.6402 (v3, 13 Apr 2011), https://arxiv.org/abs/1011.6402.
3. M. D. Gould and J. Bonart (2015 preprint). Queue imbalance as a one-tick-ahead price predictor in a limit order book. https://arxiv.org/abs/1512.03492.
4. O. E. Barndorff-Nielsen and N. Shephard (2006). Econometrics of testing for jumps in financial economics using bipower variation. *Journal of Financial Econometrics* 4(1), 1–30. https://doi.org/10.1093/jjfinec/nbi022 — full text read: Nuffield Economics Paper 2003-W21 (Sept 2004 WP), https://www.nuffield.ox.ac.uk/economics/papers/2003/w21/split.pdf.

Team literature list source: https://github.com/yuyangan/p115-data-alert-system/blob/main/docs/research.md
