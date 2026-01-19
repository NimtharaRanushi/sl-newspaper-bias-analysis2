-- Media Bias Analysis Schema Extensions
-- Run with: psql -h localhost -U your_db_user -d your_database -f schema.sql
-- Note: Replace 'your_schema' below with your actual schema name

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Topics discovered via BERTopic
CREATE TABLE IF NOT EXISTS your_schema.topics (
    id SERIAL PRIMARY KEY,
    topic_id INTEGER UNIQUE NOT NULL,
    parent_topic_id INTEGER REFERENCES your_schema.topics(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    keywords TEXT[],
    article_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Article embeddings
CREATE TABLE IF NOT EXISTS your_schema.embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES your_schema.news_articles(id),
    embedding VECTOR(3072),
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id)
);

-- Article-level analysis results
CREATE TABLE IF NOT EXISTS your_schema.article_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES your_schema.news_articles(id),
    primary_topic_id INTEGER REFERENCES your_schema.topics(id),
    topic_confidence FLOAT,
    article_type VARCHAR(50),
    article_type_confidence FLOAT,
    overall_tone FLOAT,
    headline_tone FLOAT,
    tone_reasoning TEXT,
    llm_provider VARCHAR(50),
    llm_model VARCHAR(100),
    processed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(article_id)
);

-- Event clusters
CREATE TABLE IF NOT EXISTS your_schema.event_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_name VARCHAR(255),
    cluster_description TEXT,
    representative_article_id UUID REFERENCES your_schema.news_articles(id),
    article_count INTEGER DEFAULT 0,
    sources_count INTEGER DEFAULT 0,
    date_start DATE,
    date_end DATE,
    primary_topic_id INTEGER REFERENCES your_schema.topics(id),
    centroid_embedding VECTOR(3072),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Article to cluster mapping
CREATE TABLE IF NOT EXISTS your_schema.article_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES your_schema.news_articles(id),
    cluster_id UUID NOT NULL REFERENCES your_schema.event_clusters(id),
    similarity_score FLOAT,
    UNIQUE(article_id, cluster_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_article ON your_schema.embeddings(article_id);
CREATE INDEX IF NOT EXISTS idx_article_analysis_article ON your_schema.article_analysis(article_id);
CREATE INDEX IF NOT EXISTS idx_article_analysis_topic ON your_schema.article_analysis(primary_topic_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_article ON your_schema.article_clusters(article_id);
CREATE INDEX IF NOT EXISTS idx_article_clusters_cluster ON your_schema.article_clusters(cluster_id);

-- HNSW index for similarity search (if pgvector supports it)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON your_schema.embeddings
--     USING hnsw (embedding vector_cosine_ops);
