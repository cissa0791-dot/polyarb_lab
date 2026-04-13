# information_testing_line — running notes
# polyarb_lab / research_line

---

## 2026-03-21 — Line created

Initial registry populated with 10 items (INFO-001 through INFO-010).

**Classification summary at creation:**
- pending: 9 items (INFO-001 through INFO-005, INFO-007 through INFO-010)
- park: 1 item (INFO-006 — blocked by REWARDS_API_UNVERIFIED)

**Priority order for first tests:**
1. INFO-003 — outcomePrices p_yes band validation
   - Uses existing scanner tooling. No new deps. Diagnostic mode only.
   - Fastest to run. No Docker, no external repos needed.

2. INFO-004 — structural slug filter as quality validator
   - Uses diagnostic copy of scanner. No production changes.
   - One-time run. Clear evidence standard.

3. INFO-001 — polyrec cross-feed aggregation
   - Gated on: Python 3.10+, clone to exploration/polyrec/
   - Adds genuinely new data class (Binance + Chainlink).

4. INFO-010 — floor-adjusted CAD for extreme-distance BTC markets
   - Uses existing external-ref branch tooling.
   - Blocked until BTC market inventory returns.

5. INFO-007 — whale/depth asymmetry signal
   - Requires small CLOB ingest extension (annotation only).
   - Low cost, clear test method.

**Blocked items:**
- INFO-006: REWARDS_API_UNVERIFIED (405 on /rewards/epoch)
- INFO-010: BTC market inventory absent
- INFO-009: polyrec CSV data needed; repo identity unconfirmed
- INFO-002: Docker availability unconfirmed; repo identity unconfirmed
- INFO-005: repo identity unconfirmed

---

## Iteration log

(append entries below as testing proceeds)

