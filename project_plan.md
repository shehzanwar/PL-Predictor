# Premier League Match Outcome Predictor — Project Plan

---

## Scratchpad: Statistical Edge Cases in Premier League Data

Before architecting this system, it is critical to reason through the domain-specific edge cases that will directly impact model accuracy and data pipeline reliability.

### 1. Newly Promoted Teams
Each season, 3 teams are promoted from the Championship. These teams have **zero Premier League history** for head-to-head records, PL-level form metrics, and xG baselines. Their Championship stats are not directly comparable due to the quality gap between divisions.

**Mitigation Strategy:**
- Use Championship data as a *cold-start proxy*, applying a **league-quality discount factor** (empirically, promoted teams concede ~0.3 more xGA/90 than their Championship average in the first 10 PL games).
- Weight recent form heavily (last 5-6 matches) so promoted teams build their own PL profile rapidly.
- Assign initial Elo-style ratings based on Championship finishing position, then let the rating system self-correct after ~6 matchweeks.

### 2. Mid-Season Managerial Changes
A managerial sacking fundamentally alters a team's tactical identity. The "new manager bounce" is well-documented: teams average ~1.8 PPG in the first 7 games under a new manager vs. ~1.1 PPG in the final 7 under the sacked manager (academic estimates from PL data 2010-2024).

**Mitigation Strategy:**
- Track managerial tenure as a feature: `days_since_manager_change`.
- Apply a **form reset** flag that down-weights pre-change results in rolling averages.
- Maintain a lookup table of managerial changes per season, updated manually or via news API.

### 3. Blank Gameweeks & Schedule Congestion
FA Cup replays, European competition, and COVID-era rescheduling create irregular gaps. A team playing Thursday-Sunday has measurably worse performance (~4-7% lower win probability) vs. a team with a full week's rest.

**Mitigation Strategy:**
- Engineer `days_since_last_match` for both home and away teams.
- Create a `matches_in_last_14_days` congestion metric.
- Flag "rest advantage" as a signed differential feature.

### 4. FBRef Data Availability Crisis (January 2026)
**CRITICAL:** FBRef's advanced data provider terminated their agreement in January 2026. FBRef has removed xG, xA, and other advanced Opta-sourced metrics. Only basic historical stats remain.

**Mitigation Strategy:**
- **Primary xG source:** Understat (scraping) — still serves xG, xGA, deep stats for top 6 European leagues.
- **Backup/supplementary:** API-Football (free tier: 100 requests/day for all endpoints including xG).
- **FBRef role:** Downgraded to basic match results, scores, and schedule data only.

### 5. Transfer Windows & Player Availability
Teams transform mid-season during the January window. A key signing (or departure) can shift expected performance by 5-15% in specific match contexts (e.g., losing a starting GK).

**Mitigation Strategy:**
- Scrape injury/suspension lists weekly.
- Weight "squad strength" features by minutes played, not just roster membership.
- Track transfer activity as a binary flag for "significant squad change in last 30 days."

### 6. Home Advantage Post-COVID
Post-pandemic home advantage has been verified to be *lower* than pre-2020 levels (~52% home win rate vs. historic ~46% away, now closer to 48% home, 28% draw, 24% away in 2023-2025 data). Models trained on older data will overweight home advantage.

**Mitigation Strategy:**
- Use only post-2020 seasons as training data (or apply recency weighting).
- Explicitly model `home_advantage_factor` as a time-decayed feature.

---

## 1. Architecture & Tech Stack

### 1.1 Project Structure

Per `skills.sh` Rule 3 — data ingestion, model training, and UI are in **distinct directories**:

```
PL-Predictor/
├── config/
│   ├── settings.yaml          # API keys, scraping intervals, model params
│   └── team_mappings.json     # Standardized team name mappings across sources
├── data/
│   ├── raw/                   # Unprocessed scraped data (CSV/JSON)
│   ├── processed/             # Feature-engineered, model-ready datasets
│   └── cache/                 # API response caching (rate-limit mitigation)
├── ingestion/
│   ├── __init__.py
│   ├── base_scraper.py        # Abstract base class for all scrapers
│   ├── fbref_scraper.py       # Basic match results & schedules
│   ├── understat_scraper.py   # xG, xGA, deep shooting stats
│   ├── api_football_client.py # REST client for API-Football
│   ├── injury_scraper.py      # Player availability data
│   └── orchestrator.py        # Pipeline coordinator with retry logic
├── features/
│   ├── __init__.py
│   ├── form_features.py       # Rolling form, streak analysis
│   ├── h2h_features.py        # Head-to-head record computation
│   ├── xg_features.py         # xG differential, overperformance metrics
│   ├── schedule_features.py   # Rest days, congestion, travel
│   ├── squad_features.py      # Injury impact, squad strength index
│   └── pipeline.py            # Feature pipeline orchestration
├── model/
│   ├── __init__.py
│   ├── trainer.py             # Model training, cross-validation
│   ├── predictor.py           # Match prediction interface
│   ├── evaluator.py           # Accuracy, log-loss, calibration metrics
│   └── registry.py            # Model versioning & serialization
├── ui/
│   ├── __init__.py
│   ├── app.py                 # Streamlit main application
│   ├── components/
│   │   ├── match_card.py      # Individual match prediction display
│   │   ├── form_chart.py      # Team form visualization
│   │   ├── h2h_panel.py       # Head-to-head comparison panel
│   │   └── league_table.py    # Current standings view
│   └── assets/
│       └── style.css          # Custom Streamlit theming
├── tests/
│   ├── test_ingestion/
│   ├── test_features/
│   ├── test_model/
│   └── test_ui/
├── requirements.txt
├── pyproject.toml
├── README.md
└── skills.sh
```

### 1.2 UI Framework Decision: Streamlit (Recommended)

| Criterion | Streamlit | Flet |
|---|---|---|
| **Data science integration** | Native Pandas/Plotly support | Requires manual wiring |
| **Learning curve** | Minimal (pure Python scripts) | Moderate (reactive paradigm) |
| **Visualization ecosystem** | Plotly, Altair, Matplotlib built-in | Manual chart embedding |
| **Deployment** | Streamlit Cloud (free), Docker | Requires custom hosting |
| **Community / examples** | Massive data-science community | Smaller, app-development focused |
| **Real-time updates** | Polling-based (acceptable for our use case) | WebSocket (overkill here) |

**Verdict: Streamlit** — Our application is a data dashboard for match predictions, not a real-time mobile app. Streamlit's native Pandas/Plotly integration, zero-boilerplate charting, and one-click deployment to Streamlit Cloud make it the clear winner. Flet's strengths (multi-platform, offline, pixel-perfect UI) solve problems we don't have.

### 1.3 Core Libraries

| Category | Library | Purpose |
|---|---|---|
| **Data Processing** | `pandas`, `numpy` | DataFrames, numerical computation |
| **Web Scraping** | `requests`, `beautifulsoup4`, `lxml` | HTTP requests + HTML parsing |
| **API Client** | `httpx` | Async HTTP for API-Football |
| **ML Framework** | `scikit-learn`, `xgboost` | Model training & evaluation |
| **Visualization** | `plotly` | Interactive charts in Streamlit |
| **Configuration** | `pyyaml`, `pydantic` | Settings validation & management |
| **Caching** | `diskcache` | On-disk API response caching |
| **Testing** | `pytest`, `pytest-cov` | Unit & integration testing |
| **Type Checking** | `mypy` | PEP 484 enforcement (per `skills.sh`) |

### 1.4 ML Model Strategy: Ensemble (XGBoost Primary, Random Forest Secondary)

Research findings on soccer match prediction:
- XGBoost achieves ~63-79% accuracy on PL match outcomes depending on feature quality.
- Random Forest achieves ~63-81% in comparable studies but can be less stable.
- CatBoost and LightGBM show minor improvements in specific contexts.

**Our approach:**
1. **Primary model: XGBoost Classifier** — Superior handling of feature interactions, built-in regularization to prevent overfitting on small-ish datasets (~380 matches/season), native handling of missing values (critical for partial injury data).
2. **Secondary model: Random Forest** — Used as a calibration reference and for feature importance analysis.
3. **Ensemble method:** Soft-voting ensemble of XGBoost + Random Forest predictions, weighted by cross-validated log-loss performance.
4. **Output:** Probability distribution over {Home Win, Draw, Away Win}, not just a class label.

**Target architecture for the prediction pipeline:**

```
Raw Data → Feature Engineering → Train/Test Split (time-series aware)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               XGBoost              RandomForest       Logistic Reg.
                    │                   │               (baseline)
                    └───────┬───────────┘                   │
                            ▼                               ▼
                    Soft-Vote Ensemble              Baseline Comparison
                            │
                            ▼
                  Calibrated Probabilities
                    {P(H), P(D), P(A)}
```

---

## 2. Feature Engineering Map

### 2.1 Data Points to Scrape

| Source | Data Points | Frequency | Method |
|---|---|---|---|
| **FBRef** | Match results, scores, dates, venues, lineups | Weekly | `pandas.read_html` + `requests` |
| **Understat** | xG, xGA, xPts, shot data per match | Weekly | `requests` + JSON parsing |
| **API-Football** | Live fixtures, team stats, standings, injuries | Daily | REST API (`httpx`) |
| **Transfermarkt** (scrape) | Squad market values, transfer activity | Monthly | `requests` + `beautifulsoup4` |

### 2.2 Feature Transformation Pipeline

Each raw data point is transformed into model-ready features through a standardized pipeline:

#### A. Form Features (`form_features.py`)
| Raw Input | Engineered Feature | Transformation |
|---|---|---|
| Last N match results | `form_points_last5` | Rolling sum of points (W=3, D=1, L=0) over last 5 |
| Last N match results | `form_points_last10` | Rolling sum over last 10 |
| Win/Loss sequence | `current_streak_type` | Categorical: W_streak / D_streak / L_streak |
| Win/Loss sequence | `current_streak_length` | Integer count of consecutive same result |
| Goals scored (last 5) | `goals_scored_rolling5` | Rolling mean of goals scored |
| Goals conceded (last 5) | `goals_conceded_rolling5` | Rolling mean of goals conceded |
| Match result | `manager_tenure_days` | Days since current manager appointed |
| Manager change date | `is_new_manager` | Binary: 1 if < 42 days (6 weeks) in post |

#### B. Head-to-Head Features (`h2h_features.py`)
| Raw Input | Engineered Feature | Transformation |
|---|---|---|
| Historical H2H results | `h2h_win_pct_home` | Home team win % in last 10 H2H meetings |
| Historical H2H goals | `h2h_avg_goals` | Average total goals in last 10 H2H |
| H2H results at venue | `h2h_home_venue_win_pct` | Win % at this specific venue |
| H2H recency | `h2h_last_meeting_days` | Days since last H2H encounter |

#### C. Expected Goals Features (`xg_features.py`)
| Raw Input | Engineered Feature | Transformation |
|---|---|---|
| Per-match xG | `xg_rolling5` | Rolling mean xG over last 5 matches |
| Per-match xGA | `xga_rolling5` | Rolling mean xG Against over last 5 |
| Actual goals vs xG | `xg_overperformance` | `goals_scored_rolling5 - xg_rolling5` |
| xG, xGA | `xg_differential` | `xg_rolling5 - xga_rolling5` |
| Season xPts vs actual Pts | `xpts_luck_factor` | Actual points - expected points (season) |

#### D. Schedule & Context Features (`schedule_features.py`)
| Raw Input | Engineered Feature | Transformation |
|---|---|---|
| Match dates | `days_since_last_match` | Integer days since team's last game |
| Match dates | `matches_in_last_14d` | Count of matches in prior 14 days |
| Both teams' rest | `rest_advantage` | `home_rest_days - away_rest_days` |
| Season calendar | `is_boxing_day` | Binary flag for Dec 26 fixture |
| Gameweek number | `season_phase` | Categorical: early / mid / late / final_5 |
| Venue | `is_home` | Binary: 1 for home team |

#### E. Squad Strength Features (`squad_features.py`)
| Raw Input | Engineered Feature | Transformation |
|---|---|---|
| Injury list | `injured_players_count` | Count of unavailable first-team players |
| Injury list + minutes | `injured_minutes_share` | % of total team minutes lost to injury |
| Squad market value | `squad_value_ratio` | `home_squad_value / away_squad_value` |
| Transfer window | `recent_transfers` | Binary: major transfer activity in last 30d |

### 2.3 Feature Matrix Summary

Total engineered features: **~28 features per match** (14 per team, transformed into differentials or paired features).

All features are computed with **strict temporal ordering** — no data leakage. Rolling windows use only information available *before* kickoff of the match being predicted.

---

## 3. Risk Mitigation

### 3.1 API Rate Limits & Scraping Restrictions

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FBRef blocks scraper (Cloudflare) | **High** | Medium | Random delays (3-8s), rotate User-Agents, `diskcache` to avoid re-fetching. Fallback to manual CSV download if fully blocked. |
| API-Football free tier exhaustion (100 req/day) | Medium | Low | Cache all responses for 24h. Batch requests during off-peak hours. Only fetch delta updates. |
| Understat structure changes | Medium | High | Version-pinned CSS selectors with fallback parsing. Alerts on parse failures. |
| Source data disagreement (team name mismatches) | **High** | Medium | Centralized `team_mappings.json` with normalized team IDs. Fuzzy matching with `rapidfuzz` as fallback. |

**Caching Architecture:**
```python
# Every API/scrape call passes through the cache layer
@cached(ttl=86400, cache_dir="data/cache")
def fetch_match_data(season: str, matchweek: int) -> pd.DataFrame:
    ...
```

### 3.2 Missing Player Data

| Scenario | Frequency | Handling |
|---|---|---|
| Injury list not updated | Weekly | Fall back to last known status. Flag staleness > 7 days. |
| Player not found in squad data | Occasional | Graceful degradation: set `injured_minutes_share = 0` (assume full squad). Log warning. |
| Newly signed player (no PL history) | Per transfer window | Assign league-average stats until 5+ appearances accumulated. |
| International duty absences | 3-4x/season | Cross-reference FIFA calendar. Excluded from injury count. |

### 3.3 Model-Specific Risks

| Risk | Mitigation |
|---|---|
| Overfitting on small dataset (~380 matches/season) | Train on 3-5 seasons. Use 5-fold time-series cross-validation. L2 regularization. |
| Class imbalance (draws are ~25% of outcomes) | Use `class_weight='balanced'` in XGBoost. Monitor per-class F1 scores. |
| Concept drift (playing styles evolve) | Retrain monthly. Apply exponential time-decay to older training samples. |
| Promoted teams cold-start | Championship proxy features + rapid form-based recalibration (see Scratchpad §1). |

---

## 4. Implementation Phases

### Phase 1: Data Infrastructure (Week 1-2)
**Goal:** Build a reliable, cached data pipeline that populates `data/raw/` and `data/processed/`.

- [ ] Set up project scaffolding (`pyproject.toml`, `requirements.txt`, directory structure)
- [ ] Implement `base_scraper.py` with retry logic, rate limiting, and caching
- [ ] Build `fbref_scraper.py` — match results and schedules (2020-2026 seasons)
- [ ] Build `understat_scraper.py` — xG, xGA per match
- [ ] Build `api_football_client.py` — injuries, live fixtures, standings
- [ ] Create `team_mappings.json` for cross-source name normalization
- [ ] Write `orchestrator.py` to run full ingestion pipeline
- [ ] Unit tests for each scraper with mocked HTTP responses
- [ ] Integration test: full pipeline run producing a clean merged CSV

**Deliverable:** `data/processed/pl_matches_2020_2026.csv` — a single, clean dataset with all raw columns merged.

---

### Phase 2: Feature Engineering & EDA (Week 3-4)
**Goal:** Transform raw data into the 28-feature matrix. Validate through EDA.

- [ ] Implement all feature modules (`form_features.py`, `h2h_features.py`, etc.)
- [ ] Build `pipeline.py` — end-to-end feature generation with temporal integrity
- [ ] Exploratory Data Analysis notebook:
  - Feature distributions and correlation matrix
  - Target variable balance (Home/Draw/Away split)
  - Feature importance preview via mutual information
- [ ] Handle edge cases: promoted teams, manager changes, blank gameweeks
- [ ] Unit tests for feature calculations with known match fixtures
- [ ] Validate no data leakage via temporal ordering assertions

**Deliverable:** `data/processed/pl_features_matrix.csv` — model-ready dataset with all 28 features.

---

### Phase 3: Model Training & Evaluation (Week 5-6)
**Goal:** Train, evaluate, and select the best model configuration.

- [ ] Implement `trainer.py` — XGBoost + Random Forest + Logistic Regression baseline
- [ ] Time-series aware 5-fold cross-validation (no future data in training folds)
- [ ] Hyperparameter tuning via `Optuna` or `GridSearchCV`
- [ ] Implement `evaluator.py` — accuracy, log-loss, Brier score, per-class F1
- [ ] Build soft-voting ensemble and compare vs. individual models
- [ ] Calibration analysis (reliability diagram)
- [ ] Implement `registry.py` — save best model with metadata (params, scores, date)
- [ ] Unit tests for trainer/evaluator with synthetic data

**Deliverable:** Serialized best model in `model/saved/` with evaluation report.

**Target Metrics:**
| Metric | Target | Baseline (random) |
|---|---|---|
| Accuracy | > 55% | 33% |
| Log-loss | < 0.95 | 1.10 |
| Brier Score | < 0.22 | 0.33 |

---

### Phase 4: Streamlit UI & Deployment (Week 7-8)
**Goal:** Build a polished, interactive prediction dashboard.

- [ ] Design and implement `app.py` — main Streamlit layout with sidebar navigation
- [ ] Build `match_card.py` — display per-match predictions with probability bars
- [ ] Build `form_chart.py` — rolling form line charts per team (Plotly)
- [ ] Build `h2h_panel.py` — head-to-head history visualization
- [ ] Build `league_table.py` — current standings with prediction overlays
- [ ] Custom dark-mode theming via `style.css`
- [ ] Add "Predict Matchweek" CTA: select a gameweek, view all predictions
- [ ] Add "Deep Dive" mode: select any single match for detailed breakdown
- [ ] Manual testing across screen sizes
- [ ] Deploy to Streamlit Cloud

**Deliverable:** Live, accessible Streamlit application at a public URL.

**UI Wireframe Concept:**

```
┌─────────────────────────────────────────────────────┐
│  ⚽ PL PREDICTOR          [Matchweek ▼] [Season ▼] │
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│  NAV     │   MATCH CARDS                            │
│          │   ┌────────────────────────────────┐     │
│ 📊 This  │   │ Arsenal vs Chelsea             │     │
│   Week   │   │ ██████████░░░░  62% Home Win   │     │
│          │   │ ████░░░░░░░░░░  18% Draw       │     │
│ 📈 Form  │   │ ████░░░░░░░░░░  20% Away Win   │     │
│          │   │ Key: xG diff +0.4, Form ▲▲▲▼▲ │     │
│ 🏆 Table │   └────────────────────────────────┘     │
│          │   ┌────────────────────────────────┐     │
│ ⚔️ H2H   │   │ Man Utd vs Liverpool           │     │
│          │   │ ████████░░░░░░  38% Home Win   │     │
│ ⚙️ About │   │ ██████░░░░░░░░  27% Draw       │     │
│          │   │ ████████░░░░░░  35% Away Win   │     │
│          │   └────────────────────────────────┘     │
│          │                                          │
└──────────┴──────────────────────────────────────────┘
```

---

## 5. Coding Standards Enforcement

Per `skills.sh`, the following standards are enforced throughout all phases:

| Rule | Implementation |
|---|---|
| **Rule 1:** Modular, object-oriented Python | All scrapers inherit `BaseScraper`. All feature builders implement `BaseFeatureBuilder`. Trainer/Predictor are class-based. |
| **Rule 2:** Strict type hinting (PEP 484) | `mypy --strict` in CI. All function signatures fully typed. `pydantic` models for config/data validation. |
| **Rule 3:** Separate ingestion, model, UI directories | Directory structure enforces this (see §1.1). No cross-boundary imports except through clean interfaces. |
| **Rule 4:** Scikit-learn/Pandas best practices, no nested loops | All transformations use vectorized Pandas operations. Model interfaces follow scikit-learn's `fit/predict/predict_proba` API. |

---

## 6. Verification Plan

### Automated Tests
- **Unit tests:** `pytest tests/ -v --cov=ingestion,features,model --cov-report=term-missing`
  - Scraper tests use mocked HTTP responses (`pytest-httpserver` or `responses` library)
  - Feature tests use pre-computed fixtures with known correct outputs
  - Model tests validate predict interface and output shapes with synthetic data
- **Type checking:** `mypy ingestion/ features/ model/ ui/ --strict`
- **Linting:** `ruff check .` for style consistency

### Manual Verification
- After Phase 1: Manually inspect `data/processed/pl_matches_2020_2026.csv` — verify row counts match expected matches per season (~380), spot-check scores against known results
- After Phase 2: Review EDA notebook outputs — confirm feature distributions are reasonable and no future-looking features are present
- After Phase 3: Review model evaluation report — confirm accuracy exceeds random baseline (33%) and log-loss < 1.0
- After Phase 4: Interactive walkthrough of Streamlit UI — verify all pages render, predictions display correctly, responsive on different screen sizes
