# Sri Lanka Media Bias Detector

A dashboard for analyzing media bias in Sri Lankan English newspapers by examining coverage patterns, topic distribution, and event clustering.

## Overview

This project analyzes 8,365 articles from 4 Sri Lankan newspapers (Daily News, The Morning, Daily FT, The Island) covering November-December 2025 to identify:
- **Selection bias**: Which topics each source covers (or ignores)
- **Coverage patterns**: How different sources cover the same events
- **Topic discovery**: Data-driven topic categorization using BERTopic

## First-Time Setup

If you're setting up this project for the first time:

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create configuration file**
   ```bash
   # Copy the example configuration
   cp config.yaml.example config.yaml

   # Edit config.yaml and add your database credentials
   nano config.yaml  # or use your preferred editor
   ```

   Replace the placeholder values:
   - `your_database` → Your PostgreSQL database name
   - `your_schema` → Your database schema name
   - `your_db_user` → Your database username
   - `YOUR_DATABASE_PASSWORD_HERE` → Your database password

4. **Set up database**
   ```bash
   # Create schema and tables
   psql -h localhost -U your_db_user -d your_database -f schema.sql
   ```

5. **Load your news article data**

   The project expects a table `news_articles` in your schema with columns:
   - `article_id` (integer, primary key)
   - `title` (text)
   - `content` (text)
   - `source` (text) - newspaper name
   - `published_date` (date)
   - `url` (text)

   Import your scraped articles into this table.

6. **Run the analysis pipeline** (see "Running the Pipeline" section below)

### For Contributing to the Project

If you're contributing to an organization repository:

1. **Fork the repository** on GitHub (click "Fork" button)

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sl-newspaper-bias-analysis.git
   cd sl-newspaper-bias-analysis
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/ORG_NAME/sl-newspaper-bias-analysis.git
   git remote -v  # Verify remotes are set up correctly
   ```

4. **Keep your fork synced**
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

## Project Structure

```
database-analysis/
├── config.yaml              # Configuration (LLM, embeddings, clustering)
├── schema.sql              # Database schema for analysis tables
├── requirements.txt        # Python dependencies
├── src/
│   ├── db.py              # Database operations (PostgreSQL + pgvector)
│   ├── llm.py             # LLM client abstraction (Claude/OpenAI/local)
│   ├── embeddings.py      # Embedding generation (local/OpenAI)
│   ├── topics.py          # BERTopic hierarchical topic modeling
│   └── clustering.py      # Event clustering using embeddings
├── scripts/
│   ├── 01_generate_embeddings.py  # Generate embeddings for all articles
│   ├── 02_discover_topics.py      # Run BERTopic topic discovery
│   └── 03_cluster_events.py       # Cluster articles into events
├── dashboard/
│   └── app.py             # Streamlit dashboard
└── models/
    └── bertopic_model/    # Saved BERTopic model
```

## Database Schema

### Original Data
- `news_articles` - Scraped newspaper articles (8,365 articles)

### Analysis Tables
- `embeddings` - Article embeddings (768-dim vectors from all-mpnet-base-v2)
- `topics` - Discovered topics (232 topics via BERTopic)
- `article_analysis` - Article-topic assignments
- `event_clusters` - Event clusters (1,717 clusters)
- `article_clusters` - Article-to-cluster mappings

Note: All tables are created in the schema specified in your `config.yaml`.

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Embeddings | ✅ Complete | 8,365 articles embedded using local model |
| Topic Discovery | ✅ Complete | 232 topics discovered, 6,455 articles categorized |
| Event Clustering | ✅ Complete | 1,717 clusters, 5,399 articles clustered |
| Tone Analysis | ⏸️ Skipped | Requires LLM API (Claude/OpenAI) |
| Article Type | ⏸️ Skipped | Requires LLM API |
| Dashboard | ✅ Running | Streamlit app with 4 views |

## Key Findings

### Topics Discovered
- **Top topics**: Sri Lanka politics, flooding/disasters, sports (netball, cricket), education, economy
- **Coverage**: 77% of articles successfully categorized into topics
- **Outliers**: 1,910 articles (23%) don't fit into specific topics

### Event Clusters
- **Total clusters**: 1,717 events
- **Multi-source coverage**: 87% of clusters covered by 2+ sources
- **Average cluster size**: 4.4 articles
- **Top event**: UN allocates $4.5M for Sri Lanka disaster relief (72 articles, 4 sources)

### Major Events (Nov-Dec 2025)
1. Cyclone Ditwah aftermath - 56 articles across all sources
2. Economic crisis response - 56 articles
3. Disaster relief fundraising - 47 articles
4. Weather warnings and flooding - multiple clusters

## Running the Project

### Prerequisites
```bash
# Database
PostgreSQL 16 with pgvector extension

# Python environment
Python 3.11+
pip install -r requirements.txt
```

### Database Connection
Update `config.yaml` with your database credentials:
```yaml
database:
  host: localhost
  port: 5432
  name: your_database
  schema: your_schema
  user: your_db_user
  password: "YOUR_PASSWORD"
```

### Running the Pipeline

#### 1. Generate Embeddings
```bash
python3 scripts/01_generate_embeddings.py
```
- Uses local `all-mpnet-base-v2` model (free, no API needed)
- Takes ~30 minutes for 8,365 articles on CPU
- Stores 768-dimensional vectors in PostgreSQL

#### 2. Discover Topics
```bash
python3 scripts/02_discover_topics.py
```
- Uses BERTopic with UMAP + HDBSCAN clustering
- Discovers topics automatically from data
- Generates keyword-based topic labels
- Takes ~2-3 minutes

#### 3. Cluster Events
```bash
python3 scripts/03_cluster_events.py
```
- Groups similar articles using cosine similarity (threshold: 0.8)
- Applies 7-day time window constraint
- Takes ~10 minutes

#### 4. Run Dashboard
```bash
streamlit run dashboard/app.py
```
- Access at: http://localhost:8501
- Auto-refreshes when code changes

### Sharing the Dashboard

**Local access via SSH tunnel:**
```bash
# On your local machine
ssh -L 8501:localhost:8501 amanda@server-ip
# Then open http://localhost:8501
```

**Public access via Cloudflare Tunnel:**
```bash
./cloudflared-linux-amd64 tunnel --url http://localhost:8501
# Provides a public https:// URL
```

## Dashboard Features

### 📊 Coverage Tab
- Article volume by source (bar chart)
- Coverage timeline (daily article counts)

### 🏷️ Topics Tab
- Top 20 topics by article count
- Topic-source heatmap showing coverage distribution
- Identifies which sources focus on which topics

### 📰 Events Tab
- Browse top event clusters
- See which sources covered each event
- View all articles in an event cluster

### ⚖️ Source Comparison Tab
- Topic focus comparison across sources
- Selection bias indicators
- Coverage variance analysis

## Configuration

### config.yaml

```yaml
# LLM Configuration (for future tone/type analysis)
llm:
  provider: claude  # claude | openai | local
  model: claude-sonnet-4-20250514
  temperature: 0.0

# Embeddings
embeddings:
  provider: local  # local (free) | openai
  model: all-mpnet-base-v2
  dimensions: 768

# Topic Modeling
topics:
  min_topic_size: 10  # Minimum articles per topic
  diversity: 0.5      # Keyword diversity

# Clustering
clustering:
  similarity_threshold: 0.8   # Cosine similarity threshold
  time_window_days: 7         # Only cluster articles within 7 days
  min_cluster_size: 2         # Minimum articles per cluster
```

## Future Enhancements

### With LLM API (Claude/OpenAI)
1. **Tone Analysis**: Score articles on -5 to +5 scale
2. **Article Type Classification**: news/opinion/analysis/editorial
3. **Sentence-level Analysis**: fact/opinion/quote classification
4. **Quote Extraction**: Extract speaker information
5. **Better Topic Labels**: Use LLM to generate descriptive topic names

### Other Improvements
1. **Hierarchical Topics**: Parent-child topic relationships
2. **Time-series Analysis**: Topic trends over time
3. **Source Comparison Metrics**: Quantify selection bias
4. **Framing Analysis**: Compare how sources frame the same events
5. **Export Functionality**: Download analysis results

## Troubleshooting

### Dashboard not loading
```bash
# Check if Streamlit is running
pgrep -f streamlit

# Restart dashboard
pkill -f streamlit
streamlit run dashboard/app.py
```

### Database connection issues
```bash
# Test connection
PGPASSWORD='your_password' psql -h localhost -U your_db_user -d your_database -c "SELECT COUNT(*) FROM your_schema.news_articles;"

# Check pgvector extension
psql -d your_database -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### Embedding generation fails
```bash
# Check available memory (embeddings use ~2GB RAM)
free -h

# Process in smaller batches (edit src/embeddings.py)
# Change batch_size from 100 to 50
```

## Research Methodology

Based on: "The Media Bias Detector: A Framework for Annotating and Analyzing the News at Scale" (UPenn, 2025)

### Adapted for Sri Lankan Context
- ❌ **Skipped**: Political lean (Democrat/Republican) - not applicable to SL politics
- ✅ **Kept**: Topic hierarchy via data-driven discovery
- ✅ **Kept**: Event clustering for coverage comparison
- ✅ **Kept**: Selection bias analysis (topic coverage patterns)
- ⏸️ **Future**: Framing bias analysis (requires tone scoring via LLM)

## License & Attribution

Based on the Media Bias Detector framework from University of Pennsylvania.
Adapted for Sri Lankan newspaper analysis (2025).

## Contact

For questions or issues, refer to this documentation or check the code comments.
