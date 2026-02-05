# Database Migrations

## Model Storage Migration

### What Changed

BERTopic models are now stored in the PostgreSQL database for team collaboration, rather than only on local filesystems.

**Benefits:**
- ✅ Team members can view visualizations from any machine
- ✅ Models automatically shared via the database
- ✅ No manual file copying needed
- ✅ Backward compatible with existing filesystem models

### Running the Migration

**Step 1: Update the database schema**

```bash
# Replace with your actual credentials
psql -h localhost -U your_db_user -d your_database -f migrations/add_model_storage.sql
```

**Step 2: Verify the column was added**

```bash
psql -h localhost -U your_db_user -d your_database -c "\d media_bias.result_versions"
```

You should see a `model_data` column of type `bytea`.

**Step 3: Test with a new pipeline run**

```bash
# Create a test version
python3 -c "
from src.versions import create_version, get_default_topic_config
version_id = create_version('test-db-storage', 'Testing database model storage', get_default_topic_config(), analysis_type='topics')
print(f'Version ID: {version_id}')
"

# Run the pipeline (will save to both filesystem AND database)
python3 scripts/topics/01_generate_embeddings.py --version-id <version-id>
python3 scripts/topics/02_discover_topics.py --version-id <version-id>
```

**Step 4: Verify model saved to database**

```sql
SELECT id, name,
       CASE WHEN model_data IS NULL THEN 'No model in DB'
            ELSE pg_size_pretty(length(model_data)::bigint)
       END as model_size
FROM media_bias.result_versions
WHERE name = 'test-db-storage';
```

You should see something like "7.5 MB" in the model_size column.

**Step 5: Test dashboard on a different machine**

1. On a different machine (or delete local `models/` directory)
2. Start the dashboard: `streamlit run dashboard/app.py`
3. Go to Topics tab → Select the test version
4. Scroll to "Topic Model Visualizations"
5. Visualizations should load successfully (from database)

### How It Works

#### Pipeline Behavior

When you run `scripts/topics/02_discover_topics.py`:

1. BERTopic model is trained
2. Model saved to `models/bertopic_model_{version_id[:8]}/` (filesystem)
3. Model also compressed as tar.gz and saved to database
4. Both saves happen automatically - no flags needed

#### Dashboard Behavior

When you view the Topics tab:

1. Dashboard tries to load model from database first
2. If not in database, falls back to filesystem
3. If neither location has the model, shows a message
4. Streamlit caches the loaded model in memory

#### Backward Compatibility

- **Old versions** (created before migration): Work as before, load from filesystem
- **New versions** (created after migration): Automatically stored in database
- **No breaking changes**: Everything still works if you don't run the migration

### Troubleshooting

**Q: Migration fails with "column already exists"**

A: The migration script uses `IF NOT EXISTS`, so it's safe to run multiple times. If it still fails, the column might already exist from a previous attempt.

**Q: Model saves to filesystem but not database**

A: Check the pipeline output for warnings. Common causes:
- Database connection issues
- Permissions issues on the `result_versions` table
- Model directory doesn't exist when save_model_to_version() is called

**Q: Dashboard shows "model not found" even after pipeline runs**

A: Check if the model was actually saved:
```sql
SELECT name, model_data IS NOT NULL as has_model
FROM media_bias.result_versions
WHERE id = '<version-id>';
```

If `has_model` is false, the database save failed. Check pipeline logs.

**Q: Database getting too large**

A: Each model is ~6-8 MB compressed. With 20 versions, that's ~140 MB. To clean up:

```sql
-- Delete old versions (this cascades to model_data)
DELETE FROM media_bias.result_versions
WHERE created_at < NOW() - INTERVAL '30 days'
AND name NOT IN ('baseline', 'production');  -- Keep important versions

-- Or just clear model_data but keep the version
UPDATE media_bias.result_versions
SET model_data = NULL
WHERE created_at < NOW() - INTERVAL '30 days';
```

### Migration Rollback

If you need to rollback:

```sql
ALTER TABLE media_bias.result_versions
DROP COLUMN IF EXISTS model_data;
```

Dashboard will automatically fall back to filesystem-only loading.

---

## DITWAH Claims Two-Step Process Migration

### What Changed

DITWAH claims analysis now uses a two-step process:
1. **Step 1:** Generate individual claims (one per article)
2. **Step 2:** Cluster similar individual claims into general claims (max ~40)

This enables better claim quality and control over the final number of claims.

**Benefits:**
- ✅ More accurate claims (LLM focuses on one article at a time)
- ✅ Control over number of general claims (~40 max)
- ✅ Better clustering (group similar claims automatically)
- ✅ Traceability (see which articles contribute to each general claim)

### Running the Migration

**Step 1: Update the database schema**

```bash
# Using your credentials from config.yaml
PGPASSWORD='AFYpwE%0sZNg@W' psql -h localhost -U ai_agent -d taf_media -f migrations/001_add_article_claims_table.sql
```

**Step 2: Verify the migration**

```sql
-- Check if new table exists
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'media_bias'
  AND table_name = 'ditwah_article_claims';

-- Check if columns were added to ditwah_claims
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'media_bias'
  AND table_name = 'ditwah_claims'
  AND column_name IN ('representative_article_id', 'individual_claims_count');

-- Check the new view
SELECT * FROM media_bias.ditwah_claims_hierarchy LIMIT 5;
```

### New Database Objects

1. **Table: `ditwah_article_claims`**
   - Stores individual claims (one per DITWAH article)
   - Links to general claims via `general_claim_id`

2. **Columns added to `ditwah_claims`:**
   - `representative_article_id` - best example article for this claim
   - `individual_claims_count` - how many individual claims map here

3. **View: `ditwah_claims_hierarchy`**
   - Shows individual → general claim mappings
   - Useful for debugging and analysis

### Pipeline Changes

**Old workflow (deprecated):**
```bash
python3 scripts/ditwah_claims/02_generate_claims.py --version-id <id>
```

**New workflow:**
```bash
# Step 1: Generate individual claims (one per article)
python3 scripts/ditwah_claims/02_generate_individual_claims.py --version-id <id>

# Step 2: Cluster into general claims (max ~40)
python3 scripts/ditwah_claims/03_cluster_claims.py --version-id <id>

# Step 3: Analyze sentiment and stance
python3 scripts/ditwah_claims/04_analyze_sentiment_stance.py --version-id <id>
```

### Migration Rollback

If you need to rollback:

```sql
-- Drop the view
DROP VIEW IF EXISTS media_bias.ditwah_claims_hierarchy;

-- Remove columns from ditwah_claims
ALTER TABLE media_bias.ditwah_claims
DROP COLUMN IF EXISTS representative_article_id,
DROP COLUMN IF EXISTS individual_claims_count;

-- Drop the table
DROP TABLE IF EXISTS media_bias.ditwah_article_claims CASCADE;
```

### Backward Compatibility

- Old versions using the previous approach will continue to work
- New versions will use the two-step process
- Dashboard supports both old and new claim formats
