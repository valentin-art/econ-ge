# BEA silver panel

`data/silver/bea/bea_silver.parquet` — one row per year, `EFF_START`
(`YEAR_START + 2`) through `YEAR_END` (see `src/config/sources.py`).
Column short names are documented one-line in
`data/silver/bea/bea_silver_columns.json` (generated from
`src/schemas/silver/bea_silver.py`).

Built by `src/pipelines/bea_silver_pipeline.py::run_bea_silver_pipeline`, a
**pure bronze → silver transform**: it reads already-parsed bronze parquet
(`data/bronze/bea/FixedAssets/*.parquet`, `data/bronze/bea/NIPA/*.parquet`)
and does no network I/O. It is an alternative to
`src/pipelines/bea_pipeline.py`, which additionally does the BEA API
extraction and produces the narrower CES-model input panel (`ces_data`). Both
pipelines call the same `src/features/bea/*` functions, so methodology never
diverges between them.

Notation: $j$ indexes a BEA Fixed-Assets `LineNumber` (one detailed asset
type); $s \in \{IT, nonIT\}$ indexes the two capital buckets assigned in
`src/parsers/bea/asset_dim.py`. Sums over $j \in s$ run over nonresidential
assets in bucket $s$ only. `ref` = `REF_YEAR` (1990) — the BEA chain-index
base year.

## 1. Asset dimension and wide matrices

Each asset $j$ has a fixed geometric (Hulten–Wykoff) depreciation rate
$\delta_j$ and a bucket assignment (`build_asset_dim`). Residential assets
are excluded. The four Fixed-Assets bronze tables are pivoted to
`LineNumber × Year` matrices: current-cost net stock $V_{j,t}$
(FAAt201), chain-quantity net-stock index $Q^N_{j,t}$ (FAAt204),
current-cost investment $X_{j,t}$ (FAAt205), chain-quantity investment index
$Z_{j,t}$ (FAAt206).

## 2. Investment price deflator

$$
p^I_{j,t} = \frac{X_{j,t}/X_{j,\text{ref}}}{Z_{j,t}/Z_{j,\text{ref}}}
\qquad (p^I_{j,\text{ref}} = 1)
$$

## 3. Real net stock

$$
K^N_{j,t} = V_{j,\text{ref}} \cdot \frac{Q^N_{j,t}}{Q^N_{j,\text{ref}}}
$$

In `ref`, $p^I_{j,\text{ref}}=1$ so nominal and real coincide:
$V_{j,\text{ref}} = K^N_{j,\text{ref}}$.

## 4. Smoothed capital gains

$$
\bar\pi_{j,t} = \operatorname{MA}_3\!\left(\frac{p^I_{j,t}}{p^I_{j,t-1}} - 1\right),
\qquad \bar\pi_{j,t} \ge \text{PI\_FLOOR}
$$

Centred 3-year moving average, floored at −40%/yr — raw year-over-year
IT price changes are too noisy to use directly in the user-cost formula.

## 5. Internal rate of return $r_t$

Solved from the capital-income exhaustion identity (NOS already excludes
CFC, so $\delta_j$ does not appear):

$$
r_t = \frac{\text{NOS}_t + \sum_j \bar\pi_{j,t} V_{j,t}}{\sum_j V_{j,t}}
$$

$\text{NOS}_t$ = T11600 line 2 (Net operating surplus, nonfarm nonfinancial
private business).

## 6. Rental price (user cost) and rental income

$$
p^K_{j,t} = p^I_{j,t}\,(r_t + \delta_j - \bar\pi_{j,t}),
\qquad
RI_{j,t} = p^K_{j,t}\, K^N_{j,t}
$$

## 7. Within-bucket rental shares and effective depreciation

$$
\omega_{j,t} = \frac{RI_{j,t}}{\sum_{i \in s} RI_{i,t}},
\qquad
\delta^s_t = \sum_{j \in s} \omega_{j,t}\,\delta_j
$$

→ columns `delta_it`, `delta_nonit`.

## 8. Capital services index (Tornqvist), real

$$
\Delta \ln K^s_t = \sum_{j \in s} \bar\omega_{j,t}\,\Delta \ln K^N_{j,t},
\qquad \bar\omega_{j,t} = \tfrac12(\omega_{j,t}+\omega_{j,t-1})
$$

$K^s_t = \exp\!\left(\sum_{\tau \le t} \Delta \ln K^s_\tau\right)$, normalized
to $K^s_{\text{ref}} = 1$. Superlative (chain, rental-weighted) index — used
instead of a raw sum to avoid fixed-weight aggregation bias across
heterogeneous assets (e.g. computers vs. structures). → columns
`cap_it_real_idx`, `cap_nonit_real_idx`.

## 9. Capital stock: levels and indices

$$
\underbrace{\sum_{j \in s} V_{j,t}}_{\texttt{cap\_\{it,nonit\}\_nom}},
\qquad
\underbrace{\sum_{j \in s} K^N_{j,t}}_{\texttt{cap\_\{it,nonit\}\_real}}
$$

Nominal index is the simple ratio of the nominal level to its `ref`-year
value:

$$
\texttt{cap\_\{it,nonit\}\_nom\_idx}_t = \frac{\sum_{j\in s} V_{j,t}}{\sum_{j\in s} V_{j,\text{ref}}}
$$

The real index (`cap_..._real_idx`) is $K^s_t$ from step 8, *not* a ratio of
the summed level — the Tornqvist chain avoids the substitution bias that a
naive $\sum K^N_{j,t} / \sum K^N_{j,\text{ref}}$ ratio would carry.

## 10. Rental price: nominal and real

Raw (dollar-denominated, un-normalized) dual price of capital services:

$$
p^s_t = \frac{RI^s_t}{K^s_t}, \qquad RI^s_t = \sum_{j\in s} RI_{j,t}
$$

Dollar-denominated because $K^s_{\text{ref}} = 1$. → `rent_it_nom`,
`rent_nonit_nom`. Deflating by the output-value price index (step 12):

$$
\texttt{rent\_\{it,nonit\}\_real}_t = \frac{p^s_t}{P^Y_t}
$$

## 11. Capital income share of output

$$
\texttt{share\_\{it,nonit\}}_t = \frac{RI^s_t}{Y^{\text{nom}}_t}
$$

## 12. Output value aggregate

Nonfarm business gross value added (T10305 nominal, T10306 real, line 3):

$$
P^Y_t = \frac{Y^{\text{nom}}_t}{Y^{\text{real}}_t} \Big/ \frac{Y^{\text{nom}}_{\text{ref}}}{Y^{\text{real}}_{\text{ref}}},
\qquad
\pi^Y_t = \frac{P^Y_t}{P^Y_{t-1}} - 1
$$

$$
\texttt{y\_nom\_idx}_t = \frac{Y^{\text{nom}}_t}{Y^{\text{nom}}_{\text{ref}}},
\qquad
\texttt{y\_real\_idx}_t = \frac{Y^{\text{real}}_t}{Y^{\text{real}}_{\text{ref}}}
$$

→ `y_nom`, `y_real`, `y_nom_idx`, `y_real_idx`, `p_output`, `pi_output`.

## 13. Investment: nominal and real

$$
\texttt{inv\_\{it,nonit\}\_nom}_t = \sum_{j\in s} X_{j,t},
\qquad
\texttt{inv\_\{it,nonit\}\_real}_t = \sum_{j\in s} \frac{X_{j,t}}{p^I_{j,t}}
$$

## 14. Schema and validation

`src/schemas/silver/bea_silver.py::BeaSilverRow` (patito) validates every row
before write: levels/indices/deflators `> 0`, depreciation rates in
$(0,1)$, `r_t`/`pi_output`/rental prices/shares unconstrained (the net-cost
rate `r_t + \delta_j - \bar\pi_{j,t}` can occasionally go negative for
fast-depreciating IT assets — see `diagnostics.net_cost_rate_negative` in
`bea_pipeline`). `column_descriptions()` derives the short-name →
description JSON directly from the same field `description=`s, so the
schema is the single source of truth for both validation and documentation.
