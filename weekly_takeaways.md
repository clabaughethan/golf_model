# Weekly Takeaways

Track lessons learned each week for model tuning.

---

## Week 1: Travelers Championship (June 25-29, 2026)

**Winner:** Viktor Hovland (playoff over Scheffler)

**Picks:** 22 total (14 cross-market, 8 single-market)
**Results:** 1/22 won (Ben Griffin top-20 at +110)
**P/L:** -6.49u on 10.69u staked (-60.7% ROI)

### What worked
- Ben Griffin top-20 (71.7% model, +110 odds) — model's highest-confidence pick hit. Market had him at 47.6%, model at 71.7%. The 24% edge was real.
- Scheffler win pick at 33.3% — he finished 2nd, lost in playoff. Close.

### What didn't
- Cross-market confidence filter: 0/14 picks hit. Worse than random.
- Model overconfident across all markets: predicted 25.8% avg win rate, got 4.5%.
- Hovland won at 1.3% model probability — wasn't in any pick.
- Harry Hall top-20 at 62.7% model — finished 53rd.
- Cameron Young top-10 at 52.3% — finished 47th.

### Takeaways for tuning
1. **Cross-market confidence is not a valid signal.** Being high on a player in multiple markets reflects general feature strength, not independent confirmation. Drop this as a pick filter.
2. **Focus on edge size, not model confidence.** The one win came from a big model-vs-market gap (24%), not from the model being precisely right.
3. **Calibration != accuracy.** Platt scaling fixed population-level calibration but didn't fix individual player ranking. The model assigns wrong probabilities to specific players.
4. **Small-field no-cut events may need separate treatment.** Travelers was 72 players, no cut. Top-20 base rate is ~30%, not ~13%. The model's field_size feature helps but may not be enough.
5. **One tournament is not a sample.** Need 20+ weeks before drawing statistical conclusions. This could be pure variance.
6. **The model is a ranking tool, not a probability tool.** It's better at "Player A > Player B" than "Player A = X%."
7. **Unit sizing should cap long shots.** Edge-based sizing without probability caps leads to over-betting on unlikely outcomes. New system: >50% prob = max 2u, 25-50% = max 1u, 10-25% = max 0.5u, <10% = max 0.25u.

### Open questions
- Does the model perform better on large-field cut events vs small-field no-cut?
- Is the edge (model - implied) predictive over time?
- Should we weight recent form more heavily?
- Are the stats features (SG, traditional) adding value or noise?
