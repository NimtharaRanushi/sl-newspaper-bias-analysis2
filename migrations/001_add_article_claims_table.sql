-- Migration: Add support for two-step DITWAH claims (individual → general)
-- Run with: psql -h localhost -U your_db_user -d your_database -f migrations/001_add_article_claims_table.sql

-- Step 1: Create table for individual article claims
CREATE TABLE IF NOT EXISTS media_bias.ditwah_article_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES media_bias.news_articles(id) ON DELETE CASCADE,
    result_version_id UUID NOT NULL REFERENCES media_bias.result_versions(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    general_claim_id UUID REFERENCES media_bias.ditwah_claims(id) ON DELETE SET NULL,
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id, result_version_id)
);

COMMENT ON TABLE media_bias.ditwah_article_claims IS
  'Individual claims generated for each DITWAH article (step 1 of two-step process)';

COMMENT ON COLUMN media_bias.ditwah_article_claims.general_claim_id IS
  'Links to the general claim this individual claim was clustered into (step 2)';

-- Step 2: Add new columns to ditwah_claims table (general claims)
ALTER TABLE media_bias.ditwah_claims
ADD COLUMN IF NOT EXISTS representative_article_id UUID REFERENCES media_bias.news_articles(id),
ADD COLUMN IF NOT EXISTS individual_claims_count INTEGER DEFAULT 0;

COMMENT ON COLUMN media_bias.ditwah_claims.representative_article_id IS
  'Article that best represents this general claim';

COMMENT ON COLUMN media_bias.ditwah_claims.individual_claims_count IS
  'Number of individual article claims clustered into this general claim';

-- Step 3: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_article_claims_article ON media_bias.ditwah_article_claims(article_id);
CREATE INDEX IF NOT EXISTS idx_article_claims_version ON media_bias.ditwah_article_claims(result_version_id);
CREATE INDEX IF NOT EXISTS idx_article_claims_general ON media_bias.ditwah_article_claims(general_claim_id);
CREATE INDEX IF NOT EXISTS idx_ditwah_claims_representative ON media_bias.ditwah_claims(representative_article_id);

-- Step 4: Create view to easily see individual → general claim mappings
CREATE OR REPLACE VIEW media_bias.ditwah_claims_hierarchy AS
SELECT
    ac.id as individual_claim_id,
    ac.article_id,
    ac.claim_text as individual_claim,
    gc.id as general_claim_id,
    gc.claim_text as general_claim,
    gc.claim_category,
    n.title as article_title,
    n.source_id,
    n.date_posted,
    ac.result_version_id
FROM media_bias.ditwah_article_claims ac
LEFT JOIN media_bias.ditwah_claims gc ON ac.general_claim_id = gc.id
LEFT JOIN media_bias.news_articles n ON ac.article_id = n.id
ORDER BY gc.claim_order, ac.created_at;

COMMENT ON VIEW media_bias.ditwah_claims_hierarchy IS
  'Shows the mapping from individual article claims to general claims';
