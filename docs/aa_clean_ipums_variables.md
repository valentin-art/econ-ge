# IPUMS CPS variables needed for `aa_clean/`

`aa_clean/` (`launcher.do`, `clean6275km.do`, `clean7678km.do`, `clean7909km.do`,
`cps_exper_post92.do`, `cpi.do`, `gdpmar.do`) cleans March CPS (ASEC)
microdata 1962–2009 for an Acemoglu–Autor (2008)-style wage analysis. It
currently reads a local NBER-style extract (`../../../march/source_data/marY`)
using pre-IPUMS variable names that change across year blocks (e.g. `_grdhi`
vs `grdatn`, `_clslyr` vs `clslyr`, `_incwag` vs `incwg1`/`incer1`).

This doc inventories, for every raw field the cleaning code touches, the
IPUMS CPS variable that supplies the same information, so a real
`ipumspy` extract can be built later. **Scope: variable inventory only —
no cleaning/harmonization logic is changed or evaluated here.**

Two source-year files (`cpi.do`, `gdpmar.do`) only build deflator lookup
tables keyed on `year` and don't reference any CPS microdata variables, so
they're excluded from the tables below.

## Variables to request

Flat, deduplicated list, ready to paste into an `IPUMSExtractRequest(variables=[...])`
(see `src/config/sources.py`):

```
YEAR, STATEFIP, ASECWT, ASECWTH,
AGE, SEX, RACE,
HIGRADE, EDUC,
POPSTAT, EMPSTAT, LABFORCE, FULLPART, AHRSWORKT, UHRSWORKLY,
WKSWORK1, WKSWORK2, WHYNWLY, INDLY, CLASSWLY, OCCLY,
INCWAGE, INCBUS, INCFARM, SRCEARN
```

`SERIAL`, `PERNUM`, `CPSID`, `CPSIDP` are technical record identifiers IPUMS
attaches automatically regardless of what's requested — no need to list them.
With `data_quality_flags=True` (already the default in
`IPUMSExtractor.extract`), IPUMS auto-attaches each income variable's `Q*`
allocation flag (`QINCWAGE`, `QINCBUS`, `QINCFARM`, …), so those aren't
listed separately either.

"Conf." below: ✅ = confirmed against `cps.ipums.org` variable descriptions;
⚠ = best match, worth a manual double-check before submitting the real
extract; ❌ = unresolved.

## Identifiers / weights / geography

| aa_clean name | Found in | IPUMS variable | Conf. | Note |
|---|---|---|---|---|
| `_year` → `year` | all `clean*.do` | `YEAR` | ✅ | survey year; drives the year-block branching throughout |
| `_state`, `state` | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` | `STATEFIP` | ⚠ | aa_clean carries two differently-named state fields through to output unchanged |
| `wgt` | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` | `ASECWT` | ✅ | ASEC person weight; pre-1976 `wgt`/100 scaling and the 1966 half-weight are cleaning-logic details, not variable-choice issues |

## Demographics

| aa_clean name | Found in | IPUMS variable | Conf. | Note |
|---|---|---|---|---|
| `age` | all `clean*.do` | `AGE` | ✅ | topcoded 90–99→90 in aa_clean; feeds `agely`, `exp` |
| `sex` | all `clean*.do` | `SEX` | ✅ | recoded to `female`, then dropped |
| `_race` | all `clean*.do`, `cps_exper_post92.do` | `RACE` | ✅ | feeds `white`/`black`/`other` dummies |

## Education

| aa_clean name | Found in | IPUMS variable | Conf. | Note |
|---|---|---|---|---|
| `_grdhi` + `grdcom` (pre-1992) | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` (1979–87 block) | `HIGRADE` | ✅ | IPUMS already merges "grade attended" + "completed" into one field for 1962, 1964–91; no separate `GRDCOM` is exposed on IPUMS |
| `grdatn` (1992+) | `clean7909km.do` (1992–2009 block), `cps_exper_post92.do` | `EDUC` | ✅ | general "educational attainment recode", spans 1962+; `EDUC99` also exists as a 1990-recode-only variant for 1992+ if a stricter match is wanted |

## Employment / labor supply

| aa_clean name | Found in | IPUMS variable | Conf. | Note |
|---|---|---|---|---|
| `_popstat` | all `clean*.do` | `POPSTAT` | ✅ | adult civilian / armed forces / child; used to exclude armed forces |
| `_esr` (→ builds `NILF`, 1962–87 only) | `clean6275km.do`, `clean7678km.do` | `EMPSTAT` + `LABFORCE` | ⚠ | real `ESR` only covers 1968+ in two incompatible variants (`UH_ESR_A3` 1968–88, `UH_ESR_A4` 1989+); `EMPSTAT`/`LABFORCE` cover the full 1962+ range and can rebuild the same NILF flag |
| `ftpt` | all `clean*.do` | `FULLPART` | ✅ | full/part-time last year, 35+ hrs/wk threshold |
| `hours` (hours last week) | all `clean*.do` | `AHRSWORKT` | ⚠ | hours worked last week, all jobs |
| `hrslyr` | all `clean*.do` | `UHRSWORKLY` | ✅ | usual hours/week last year; ASEC-only, available from 1976 — matches aa_clean's own 1962–75 imputation-from-1976–78 logic |
| `_wkslyr` | all `clean*.do` | `WKSWORK1` + `WKSWORK2` | ✅ | `WKSWORK1` = continuous weeks, `WKSWORK2` = bracketed intervals — matches aa_clean's own pre/post-1976 bracketed-vs-continuous branch |
| `_pyrsn` | all `clean*.do` | `WHYNWLY` | ✅ | reason not working last year; not available 1964–67 in both aa_clean's source and IPUMS's own universe notes |
| `wklkun` | `clean7678km.do`, `clean7909km.do` | *(unresolved)* | ❌ | dropped by aa_clean without being used anywhere, no comment explaining it; recommend omitting from the extract rather than guessing |
| `indlyr` | all `clean*.do` | `INDLY` | ✅ | industry of the longest job held last year |
| `clslyr` / `_clslyr` | `clean6275km.do` (as `_clslyr`), `clean7678km.do`/`clean7909km.do` (as `clslyr`) | `CLASSWLY` | ✅ | class of worker, last year |
| `occ` | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` | `OCCLY` | ⚠ | grouped with `indlyr`/`clslyr` in aa_clean's keep-lists, so almost certainly the "last year" occupation, not current-week `OCC` |

## Earnings / income

| aa_clean name | Found in | IPUMS variable | Conf. | Note |
|---|---|---|---|---|
| `_incwag` (pre-1988) / `incwg1` (1988+) | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` | `INCWAGE` | ✅ | wage & salary income, harmonized across the 1988 redesign |
| `_incse` | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` (1979–87 block) | `INCBUS` | ✅ | nonfarm self-employment/business income |
| `_incfrm` | `clean6275km.do`, `clean7678km.do`, `clean7909km.do` (1979–87 block) | `INCFARM` | ✅ | farm self-employment income |
| `incer1` (1988+ "earnings/self-employment") | `clean7909km.do` (1988–91, 1992–2009 blocks) | `INCBUS` (+`INCFARM` if kept split) | ⚠ | aa_clean treats this as one post-88 field; IPUMS keeps business and farm income split |
| `ernsrc` | `clean7909km.do` (1988–91, 1992–2009 blocks) | `SRCEARN` | ✅ | source of earnings on the longest job held last year |
| `aincwag` / `aincwg1` / `aincer1` | all `clean*.do` | *(auto-attached via `data_quality_flags=True`)* | ✅ | allocation/edit flags for the income vars above — don't list as separate `variables` entries |

## Open questions to confirm before submitting a real extract

- **`STATEFIP`** — confirm coverage/coding matches aa_clean's `_state`/`state` across all year blocks (some early years may only have region-level geography, not full state detail, in IPUMS CPS due to Census disclosure rules).
- **`AHRSWORKT`** — confirm this (not some other "hours last week" variable) is what aa_clean's `hours` represents, and confirm it's available back to 1962.
- **`EMPSTAT`/`LABFORCE` vs `ESR`** — decide whether to reconstruct aa_clean's `NILF` flag from the harmonized `EMPSTAT`/`LABFORCE` pair (full 1962+ coverage) or from the raw `ESR` variants (closer 1:1 match to aa_clean's variable, but split across two incompatible codings and starting only in 1968).
- **`OCCLY`** — confirm aa_clean's unadorned `occ` really is last-year's occupation and not a current-week `OCC` pull that happens to sit next to `indlyr`/`clslyr` in the keep-list.
- **`incer1` split** — decide whether to keep `INCBUS`/`INCFARM` separate on the IPUMS side (as IPUMS provides them) or sum them to match aa_clean's single post-88 `incer1` concept; this is a cleaning-logic decision, out of scope here.
- **`wklkun`** — no IPUMS equivalent proposed; aa_clean never uses this field, so it's recommended to leave out of the extract entirely.
