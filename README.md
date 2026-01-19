# Sri Lanka Newspaper Bias Analysis

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A data-driven analysis framework for detecting media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project analyzes **8,365 articles** from **4 Sri Lankan newspapers** (Daily News, The Morning, Daily FT, The Island) covering November-December 2025 to identify:

- 📰 **Selection bias**: Which topics each source covers (or ignores)
- 🔍 **Coverage patterns**: How different sources cover the same events
- 🏷️ **Topic discovery**: Data-driven topic categorization using BERTopic
- 📊 **Event clustering**: Grouping articles about the same events across sources

## Key Findings

### Topics Discovered
- **232 topics** automatically discovered from 8,365 articles
- **77% coverage**: Successfully categorized 6,455 articles
- **Top topics**: Sri Lanka politics, flooding/disasters, sports, education, economy

### Event Clusters
- **1,717 event clusters** identified
- **87% multi-source coverage**: Most events covered by 2+ sources
- **Top event**: UN allocates $4.5M for Sri Lanka disaster relief (72 articles across 4 sources)

### Major Events (Nov-Dec 2025)
1. Cyclone Ditwah aftermath - 56 articles
2. Economic crisis response - 56 articles
3. Disaster relief fundraising - 47 articles
4. Weather warnings and flooding - multiple clusters

## Features

- 🧠 **Semantic embeddings**: 768-dimensional vectors using `all-mpnet-base-v2`
- 🎯 **Topic modeling**: BERTopic with UMAP + HDBSCAN clustering
- 🔗 **Event clustering**: Cosine similarity with time-window constraints
- 📈 **Interactive dashboard**: Streamlit-based visualization
- 🗄️ **Vector database**: PostgreSQL with pgvector extension

## Tech Stack

- **Python 3.11+**: Core language
- **PostgreSQL 16 + pgvector**: Database with vector similarity search
- **Sentence Transformers**: Local embedding generation (no API needed)
- **BERTopic**: Topic modeling with UMAP/HDBSCAN
- **Streamlit**: Interactive dashboard
- **pandas, numpy**: Data processing

## Quick Start

### Prerequisites

```bash
# Database
PostgreSQL 16 with pgvector extension

# Python
Python 3.11+
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure database**
   ```bash
   # Copy configuration template
   cp config.yaml.example config.yaml

   # Edit config.yaml with your database credentials
   nano config.yaml
   ```

4. **Set up database schema**
   ```bash
   psql -h localhost -U your_db_user -d your_database -f schema.sql
   ```

5. **Run the analysis pipeline**
   ```bash
   # Generate embeddings (takes ~30 min on CPU)
   python3 scripts/01_generate_embeddings.py

   # Discover topics (takes ~2-3 min)
   python3 scripts/02_discover_topics.py

   # Cluster events (takes ~10 min)
   python3 scripts/03_cluster_events.py
   ```

6. **Launch dashboard**
   ```bash
   streamlit run dashboard/app.py
   # Access at http://localhost:8501
   ```

## Project Structure

```
sl-newspaper-bias-analysis/
├── config.yaml.example     # Configuration template
├── schema.sql              # Database schema
├── requirements.txt        # Python dependencies
├── src/
│   ├── db.py              # Database operations
│   ├── embeddings.py      # Embedding generation
│   ├── topics.py          # Topic modeling
│   └── clustering.py      # Event clustering
├── scripts/
│   ├── 01_generate_embeddings.py
│   ├── 02_discover_topics.py
│   └── 03_cluster_events.py
└── dashboard/
    └── app.py             # Streamlit dashboard
```

## Dashboard Preview

The dashboard includes 4 interactive views:

1. **📊 Coverage Tab**: Article volume and timeline by source
2. **🏷️ Topics Tab**: Top topics and source-topic heatmap
3. **📰 Events Tab**: Browse event clusters and cross-source coverage
4. **⚖️ Source Comparison**: Topic focus and selection bias analysis

## Database Schema

### Original Data
- `news_articles` - Scraped newspaper articles (8,365 articles)

### Analysis Tables
- `embeddings` - Article embeddings (768-dim vectors)
- `topics` - Discovered topics (232 topics)
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters (1,717 clusters)
- `article_clusters` - Article-to-cluster mappings

## Research Methodology

Based on: **"The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale"** (University of Pennsylvania, 2025)

### Adaptations for Sri Lankan Context
- ✅ Topic hierarchy via data-driven discovery
- ✅ Event clustering for coverage comparison
- ✅ Selection bias analysis (topic coverage patterns)
- ❌ Political lean (Democrat/Republican) - not applicable
- ⏸️ Framing bias analysis - requires LLM API (future work)

## Future Enhancements

### With LLM API (Claude/OpenAI)
- **Tone analysis**: Score articles on sentiment scale
- **Article type classification**: news/opinion/analysis/editorial
- **Quote extraction**: Extract speaker information
- **Better topic labels**: LLM-generated descriptive names

### Other Improvements
- Hierarchical topic relationships
- Time-series topic trends
- Quantified selection bias metrics
- Framing analysis across sources
- Export functionality

## Configuration

All configuration is in `config.yaml`:

```yaml
database:
  host: localhost
  name: your_database
  schema: your_schema
  user: your_db_user

embeddings:
  provider: local  # Free, no API needed
  model: all-mpnet-base-v2

clustering:
  similarity_threshold: 0.8
  time_window_days: 7
```

## Performance Notes

- **Embedding generation**: ~30 minutes for 8,365 articles (CPU)
- **Topic discovery**: ~2-3 minutes
- **Event clustering**: ~10 minutes
- **Memory usage**: ~2GB RAM during embedding generation

## Troubleshooting

See [CLAUDE.md](CLAUDE.md) for detailed troubleshooting guides.

## License

MIT License - see LICENSE file for details

## Attribution

Based on the Media Bias Detector framework from University of Pennsylvania.
Adapted for Sri Lankan newspaper analysis (2025).

## Documentation

- **README.md** (this file) - Quick start and overview
- **CLAUDE.md** - Comprehensive documentation and setup guide
- **schema.sql** - Database schema with comments

## Contact

For questions or issues, please open a GitHub issue or refer to the documentation.
