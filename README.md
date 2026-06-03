# E-Commerce Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/github/license/Duvini/E-Commerce-Reccomendation-System?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![Deploy](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render&logoColor=white)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [Technical Architecture](#technical-architecture)
5. [ML Model Details](#ml-model-details)
6. [Data Requirements](#data-requirements)
7. [Project Structure](#project-structure)
8. [Dependencies & Requirements](#dependencies--requirements)
9. [Installation](#installation)
10. [Database Setup](#database-setup)
11. [Usage Guide](#usage-guide)
12. [API Reference](#api-reference)
13. [Deployment](#deployment)
14. [Contributing](#contributing)
15. [License](#license)

---

## Project Overview

The **E-Commerce Recommendation System** is a full-stack machine learning web application that delivers real-time, personalized product recommendations to shoppers. Built on Flask and powered by scikit-learn, it implements two complementary recommendation strategies—**content-based filtering** and **user-based collaborative filtering**—served through a responsive Bootstrap 5 UI.

The system ingests a pre-processed product and review dataset at startup, exposes recommendation endpoints via HTTP POST, and renders results dynamically using Jinja2 templates. It is deployable to cloud PaaS platforms (Render, Heroku) with a single `gunicorn` process.

**Business value**: Personalized product discovery directly correlates with increased conversion rates and average order value. By surfacing relevant items through two distinct ML signals (item similarity and peer behavior), the system covers both cold-start product discovery and returning-user personalization.

---

## Problem Statement

In a large e-commerce catalog, the majority of products receive negligible traffic because users cannot efficiently discover items relevant to their preferences or browsing context. Generic, popularity-ranked listings fail to surface the long tail of inventory and lead to poor engagement metrics.

This system addresses two specific gaps:

1. **Item-level discovery**: Given a product a user is currently viewing, surface semantically similar alternatives and complements using textual product metadata.
2. **User-level personalization**: Given a returning user's interaction history, predict products they are likely to engage with by modeling behavioral similarity to peer users.

---

## Key Features

| Feature | Mechanism | Endpoint |
|---|---|---|
| Trending products homepage | Rating-aggregated top-12 ranking | `GET /` |
| Content-based recommendations | TF-IDF + Euclidean distance on product tags | `POST /recommendations` |
| User-based collaborative filtering | Cosine similarity on user-item rating matrix | `POST /recommendations_by_user_id` |
| User registration | Password hashing via Werkzeug, MySQL persistence | `POST /signup` |
| User authentication | Hash comparison + session messaging | `POST /signin` |
| Configurable result count | `nbr` query parameter (default: 10, max set by caller) | `POST /recommendations` |

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Flask Application                     │
│                           app.py                            │
│                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │   Routes     │    │     Recommendation Engine        │   │
│  │              │    │                                  │   │
│  │  /           │───▶│  get_trending_products()         │   │
│  │  /recommend  │───▶│  content_based_recommendations() │   │
│  │  /forYou     │───▶│  collaborative_filtering_        │   │
│  │  /signup     │    │    recommendations()             │   │
│  │  /signin     │    │                                  │   │
│  └──────────────┘    └──────────────────────────────────┘   │
│                                    │                         │
│  ┌─────────────────┐               │                         │
│  │  Flask-SQLAlchemy│              ▼                         │
│  │  MySQL (ecom db) │   ┌─────────────────────┐             │
│  │  Signup model    │   │   models/ (CSV data) │             │
│  └─────────────────┘   │   clean_data.csv     │             │
│                         │   trending_products  │             │
│  ┌─────────────────┐   └─────────────────────┘             │
│  │  Jinja2 Templates│                                        │
│  │  index.html      │                                        │
│  │  main.html       │   ◀── Content-based results           │
│  │  forYou.html     │   ◀── Collaborative results           │
│  │  signup/signin   │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Request
  │
  ▼
Flask Route receives product name / user ID
  │
  ├─ Content-based path:
  │    TF-IDF vectorize product Tags column
  │    ──▶ Locate query item vector by partial name match
  │    ──▶ Compute Euclidean distances to all items
  │    ──▶ Return top-N nearest neighbors
  │
  └─ Collaborative path:
       Pivot train_data into (user × product) rating matrix
       ──▶ Compute cosine similarity between all user vectors
       ──▶ Rank similar users by similarity score
       ──▶ Collect products rated by similar users, unseen by target
       ──▶ Return up to top-N unique products
```

---

## ML Model Details

### 1. Content-Based Filtering

**Algorithm**: TF-IDF Vectorization + Euclidean Distance

| Parameter | Value | Rationale |
|---|---|---|
| `stop_words` | `'english'` | Removes high-frequency, low-signal tokens |
| `max_df` | `0.8` | Excludes terms appearing in >80% of documents (too generic) |
| `min_df` | `2` | Excludes terms appearing in <2 documents (too rare) |
| `ngram_range` | `(1, 2)` | Captures single words and two-word phrases for richer semantics |
| Distance metric | Euclidean | Measures absolute distance in TF-IDF space |
| `top_n` default | `40` candidates → `nbr` returned | Oversamples candidates for ranking flexibility |

**Process**:
1. All product `Tags` fields are vectorized into a sparse TF-IDF matrix at query time.
2. The query product is located by partial case-insensitive name match against the `Name` column.
3. Euclidean distances are computed between the query vector and all item vectors using `sklearn.metrics.pairwise.pairwise_distances`.
4. The `top_n` smallest-distance items (excluding the query item itself) are returned sorted by ascending distance (most similar first).

> **Note**: The TF-IDF matrix is recomputed on each request. For production scale, this should be precomputed and cached (see [Deployment Considerations](#deployment)).

---

### 2. User-Based Collaborative Filtering

**Algorithm**: User-Item Matrix + Cosine Similarity

| Parameter | Value | Notes |
|---|---|---|
| Matrix construction | `pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean')` | Mean-aggregates duplicate ratings |
| Fill strategy | `fillna(0)` | Absent ratings treated as zero interaction |
| Similarity metric | Cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) | Direction-based; robust to user rating scale differences |
| Candidate strategy | Items rated by similar users, not yet rated by target | Pure collaborative signal; no content overlap |
| Result cap | `top_n` (default: 10) | Configurable per-request |

**Process**:
1. A full user × product rating matrix is constructed from `clean_data.csv` at query time.
2. Cosine similarity is computed across all user vectors simultaneously.
3. Users are ranked by similarity score relative to the target user (descending).
4. For each similar user (in rank order), unrated products of the target user are added to the candidate set until `top_n` items are collected.
5. Candidates are matched back to product metadata and returned.

**Cold-start handling**: If the target `user_id` does not exist in the matrix, the endpoint returns an empty response with a user-facing message rather than raising an exception.

---

### 3. Trending Products (Baseline)

**Algorithm**: Rating mean aggregation

Products are grouped by `(Name, ReviewCount, Brand, Price, ImageURL)` and the mean `Rating` across all reviews is computed. The top 12 by mean rating populate the homepage. This serves as a non-personalized fallback and initial discovery surface.

---

## Data Requirements

### Primary Dataset: `models/clean_data.csv`

This file (~6.2 MB) is the pre-processed training dataset loaded at application startup. It must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `ID` | int | User identifier |
| `ProdID` | str/int | Product identifier |
| `Name` | str | Full product name (used for content-based lookup) |
| `Brand` | str | Product brand |
| `Price` | float | Product price |
| `Rating` | float | User rating (typically 1–5 scale) |
| `ReviewCount` | int | Total number of reviews |
| `ImageURL` | str | Absolute URL to product image |
| `Tags` | str | Space- or comma-separated descriptive keywords for TF-IDF vectorization |

### Secondary Dataset: `models/trending_products.csv`

| Column | Type | Description |
|---|---|---|
| `Name` | str | Product name |
| `Brand` | str | Brand |
| `Price` | float | Price |
| `Rating` | float | Pre-aggregated rating |
| `ImageURL` | str | Image URL |

> This file is loaded at startup but the homepage currently re-derives trending products live from `clean_data.csv`. The file may serve as a cached snapshot or for future offline aggregation jobs.

### Source Dataset

The repository ships with a raw Walmart.com product review dataset:  
`marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.tsv`  
(~3.2 MB, TSV format, covering H2 2020). The `recommendationSystem.ipynb` notebook documents the preprocessing pipeline that transforms this raw data into `clean_data.csv`.

---

## Project Structure

```
E-Commerce-Reccomendation-System/
│
├── app.py                          # Flask application: routes, ML inference, DB models
│
├── models/
│   ├── clean_data.csv              # Pre-processed training dataset (~6.2 MB)
│   └── trending_products.csv       # Pre-aggregated trending products snapshot
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Homepage: trending products, search modals
│   ├── main.html                   # Content-based recommendation results
│   ├── forYou.html                 # Collaborative filtering results (For You page)
│   ├── signup.html                 # User registration form
│   └── new.html                    # (Draft / unused template)
│
├── static/
│   ├── css/
│   │   ├── bootstrap.min.css       # Customized Bootstrap stylesheet
│   │   └── style.css               # Application-specific styles
│   ├── js/
│   │   └── main.js                 # Client-side JS (WOW.js init, Isotope grid)
│   ├── img/                        # Static images and favicon
│   └── lib/                        # Vendored frontend libraries
│       ├── animate/                # Animate.css
│       ├── owlcarousel/            # Owl Carousel
│       ├── lightbox/               # Lightbox2
│       └── wow/                    # WOW.js scroll animations
│
├── scss/
│   └── bootstrap/                  # Bootstrap SCSS source (for theme customization)
│
├── recommendationSystem.ipynb      # EDA, preprocessing, and model prototyping notebook
├── marketing_sample_for_walmart... # Raw Walmart review dataset (TSV, ~3.2 MB)
│
├── requirements.txt                # Pinned Python dependencies
├── Procfile                        # Heroku/Render process definition
├── render.yaml                     # Render.com deployment manifest
├── LICENSE                         # MIT License
└── README.md                       # This document
```

---

## Dependencies & Requirements

### Runtime Environment

- **Python**: 3.8+ (tested with 3.8–3.11)
- **Database**: MySQL 5.7+ (local) or any SQLAlchemy-compatible RDBMS

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| Flask | 3.0.3 | WSGI web framework |
| Flask-SQLAlchemy | 3.1.1 | ORM integration |
| SQLAlchemy | 2.0.34 | Database abstraction |
| mysqlclient | 2.2.4 | MySQL C driver bindings |
| pandas | 2.2.2 | Tabular data loading and manipulation |
| numpy | 1.26.4 | Numerical array operations |
| scikit-learn | 1.5.2 | TF-IDF, cosine similarity, pairwise distances |
| scipy | 1.14.1 | Sparse matrix operations (scikit-learn backend) |
| Werkzeug | 3.0.4 | Password hashing, WSGI utilities |
| gunicorn | 20.1.0 | Production WSGI server |
| Jinja2 | 3.1.4 | HTML templating |
| spacy | 3.7.6 | NLP library (used in notebook preprocessing) |
| en-core-web-sm | 3.7.1 | spaCy English language model |
| matplotlib | 3.9.2 | Visualization (notebook only) |
| seaborn | 0.13.2 | Statistical plots (notebook only) |

Full pinned dependency list: [`requirements.txt`](requirements.txt)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- MySQL 5.7+ (or MariaDB equivalent) running locally
- `pip` and `venv`
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Duvini/E-Commerce-Reccomendation-System.git
cd E-Commerce-Reccomendation-System
```

### 2. Create and Activate a Virtual Environment

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `mysqlclient` requires the MySQL C development headers. On Debian/Ubuntu: `sudo apt-get install libmysqlclient-dev`. On macOS with Homebrew: `brew install mysql-client`.  
> On Windows, install the pre-built binary: `pip install mysqlclient` works if MySQL Connector is installed.

### 4. Download the spaCy Language Model (if running the notebook)

```bash
python -m spacy download en_core_web_sm
```

---

## Database Setup

The application requires a MySQL database for user authentication. The ORM will create the `signup` table automatically on first run.

### 1. Create the Database

```sql
CREATE DATABASE ecom CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. Configure the Connection String

Open `app.py` and update line 17 with your MySQL credentials:

```python
# app.py, line 17
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"
#                                                 ^^^^  ^
#                                            username  password (empty by default)
```

For environments with a password:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:yourpassword@localhost/ecom"
```

> **Security**: Do not hardcode credentials in source. In production, inject via environment variables (see [Deployment](#deployment)).

### 3. Initialize the Schema

The `Signup` table is created automatically by SQLAlchemy on application startup via `db.create_all()`. No manual migration step is required for the initial schema.

---

## Usage Guide

### Starting the Development Server

```bash
python app.py
```

The application starts on `http://127.0.0.1:5000` by default with debug mode enabled.

### Starting with Gunicorn (Production-like)

```bash
gunicorn app:app --bind 0.0.0.0:8000 --workers 2
```

### Navigating the UI

| Page | URL | Description |
|---|---|---|
| Homepage | `http://localhost:5000/` | Trending products grid with search modals |
| Content Recommendations | Via modal on homepage | Enter product name and number of results |
| For You | `http://localhost:5000/forYou` | Enter a numeric user ID for personalized results |
| Sign Up | `http://localhost:5000/signup` | Create a new user account |
| Sign In | `http://localhost:5000/signin` | Authenticate an existing user |

### Getting Content-Based Recommendations (via form)

1. Click **Product** on the homepage hero section.
2. Enter a product name keyword (e.g., `"laptop"`, `"headphones"`).
3. Optionally enter the number of results (default: 10).
4. Click **Search** — results render on `main.html`.

### Getting User-Based Recommendations (via form)

1. Click **User ID** on the homepage hero section.
2. Enter a numeric user ID that exists in `clean_data.csv`.
3. Click **Search** — personalized results render on `forYou.html`.

### Programmatic Usage (curl examples)

**Content-based recommendations:**
```bash
curl -X POST http://localhost:5000/recommendations \
  -d "prod=bluetooth+headphones&nbr=5"
```

**User-based recommendations:**
```bash
curl -X POST http://localhost:5000/recommendations_by_user_id \
  -d "user_id=12345"
```

---

## API Reference

### `GET /`
Returns the homepage with the top-12 trending products (highest mean rating).

**Response**: HTML (`index.html`)

---

### `POST /recommendations`
Returns content-based product recommendations using TF-IDF similarity on product tags.

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prod` | string | Yes | — | Product name or keyword to match |
| `nbr` | integer | No | 10 | Number of recommendations to return |

**Response**: HTML (`main.html`) with `content_based_rec` DataFrame of matched products.  
**Fallback**: If no matching product is found, renders `main.html` with a `"No recommendations available"` message.

---

### `POST /recommendations_by_user_id`
Returns collaborative filtering recommendations for a given user ID.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `user_id` | integer (string form) | Yes | Numeric user ID from the training dataset |

**Response**: HTML (`forYou.html`) with `content_based_rec` DataFrame.  
**Fallback**: Empty DataFrame response if user is not in the matrix, or `"Invalid User ID format"` if non-numeric.

---

### `POST /signup`
Registers a new user.

| Field | Type | Required |
|---|---|---|
| `username` | string | Yes |
| `email` | string | Yes |
| `password` | string | Yes |

**Response**: Redirects to `index.html` with success flash message.

---

### `POST /signin`
Authenticates an existing user.

| Field | Type | Required |
|---|---|---|
| `signinUsername` | string | Yes |
| `signinPassword` | string | Yes |

**Response**: `index.html` on success; `signin.html` with error on failure.

---

## Deployment

### Render (Recommended)

The repository ships with a `render.yaml` manifest for zero-configuration deployment on [Render.com](https://render.com).

```yaml
# render.yaml
services:
  - type: web
    name: app
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
```

**Steps**:
1. Push the repository to GitHub.
2. Connect the repo to Render and create a new **Web Service**.
3. Render auto-detects `render.yaml` and configures the build/start commands.
4. Set the following **Environment Variables** in the Render dashboard:

| Variable | Description |
|---|---|
| `DATABASE_URL` | Full MySQL connection URI (update `app.py` to read `os.environ.get('DATABASE_URL')`) |
| `SECRET_KEY` | Flask session secret (replace the hardcoded value in `app.py`) |

### Heroku

The `Procfile` is compatible with Heroku:

```
web: gunicorn app:app
```

```bash
heroku create your-app-name
heroku addons:create jawsdb:kitefin   # MySQL add-on
heroku config:set SECRET_KEY=<secure-random-string>
git push heroku main
```

### Environment Variable Hardening (Required Before Production)

The following values are currently hardcoded in `app.py` and **must** be externalized before any public deployment:

```python
# Current (insecure for production)
app.secret_key = "your_secret_key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/ecom"

# Recommended pattern
import os
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-fallback")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    "DATABASE_URL", "mysql://root:@localhost/ecom"
)
```

### Scaling Notes

- **TF-IDF recomputation**: Currently executed on every `/recommendations` request. For datasets beyond ~50K rows, precompute the TF-IDF matrix at startup and cache the vectorizer + matrix as module-level globals.
- **User-item matrix**: Similarly reconstructed on each `/recommendations_by_user_id` call. Cache the pivot table and similarity matrix at startup or refresh on a schedule.
- **Workers**: gunicorn workers share no in-memory state with the current architecture. For cached matrices, use a single worker + threads, or an external cache (Redis).
- **Dataset storage**: `models/clean_data.csv` is committed to the repository. For larger datasets or frequent updates, migrate to a database query or object storage (S3).

---

## Contributing

Contributions are welcome. Please follow the workflow below.

### Development Setup

```bash
git clone https://github.com/Duvini/E-Commerce-Reccomendation-System.git
cd E-Commerce-Reccomendation-System
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Branching Strategy

| Branch pattern | Purpose |
|---|---|
| `main` | Production-ready code only |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `experiment/<name>` | ML experiments (may not merge) |

### Submitting Changes

1. Fork the repository and create your branch from `main`.
2. Ensure the application starts without errors (`python app.py`).
3. If changing recommendation logic, document expected behavior changes and, where possible, provide sample inputs/outputs.
4. Open a Pull Request with a clear description of the change and its motivation.

### Areas for Contribution

- **Performance**: Cache TF-IDF matrix and user similarity matrix at app startup.
- **Evaluation harness**: Add offline evaluation scripts (precision@k, recall@k, NDCG) against a held-out test split.
- **Hybrid mode**: Combine content-based and collaborative signals into a unified ranked list.
- **Session-based personalization**: Use authenticated session state to auto-fetch user-based recommendations.
- **Configuration**: Externalize `SECRET_KEY`, `DATABASE_URL`, and model hyperparameters to environment variables / config file.
- **Tests**: Add pytest unit tests for `content_based_recommendations()` and `collaborative_filtering_recommendations()`.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

---

## Attribution

- Dataset sourced from Walmart.com product reviews (H2 2020 sample).
- Frontend template based on Bootstrap 5 with WOW.js, Owl Carousel, and Lightbox2.
- NLP preprocessing via [spaCy](https://spacy.io/) (`en_core_web_sm`).
- ML primitives from [scikit-learn](https://scikit-learn.org/).
- Developed by **DataWizards** — Temple Street, Malmbe, Sri Lanka.

---

*© 2024 E-Commerce Recommendation System. All Rights Reserved.*
