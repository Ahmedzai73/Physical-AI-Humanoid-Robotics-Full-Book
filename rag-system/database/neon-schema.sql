-- Database schema for Physical AI & Humanoid Robotics Book RAG System
-- Designed for Neon Serverless Postgres

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Table for storing document chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    chapter_id VARCHAR(255) NOT NULL,
    chapter_title TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536), -- For OpenAI text-embedding-3-large
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for chapter_id for faster filtering
CREATE INDEX IF NOT EXISTS idx_document_chunks_chapter_id
ON document_chunks (chapter_id);

-- Index for metadata queries
CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
ON document_chunks
USING GIN (metadata);

-- Table for chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Index for user_id for faster user session queries
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
ON chat_sessions (user_id);

-- Table for chat messages
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    grounding_score DECIMAL(3,2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for session_id for faster message retrieval
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
ON chat_messages (session_id);

-- Index for timestamp for chronological queries
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp
ON chat_messages (timestamp);

-- Table for users
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

-- Index for username and email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_username
ON users (username);

CREATE INDEX IF NOT EXISTS idx_users_email
ON users (email);

-- Table for book metadata
CREATE TABLE IF NOT EXISTS book_metadata (
    id SERIAL PRIMARY KEY,
    chapter_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    module VARCHAR(50) NOT NULL,
    word_count INTEGER DEFAULT 0,
    estimated_reading_time INTEGER DEFAULT 0, -- in minutes
    difficulty VARCHAR(20) DEFAULT 'intermediate' CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    prerequisites JSONB DEFAULT '[]',
    learning_objectives JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for module and chapter_id for faster queries
CREATE INDEX IF NOT EXISTS idx_book_metadata_module
ON book_metadata (module);

CREATE INDEX IF NOT EXISTS idx_book_metadata_chapter_id
ON book_metadata (chapter_id);

-- Table for user interactions and analytics
CREATE TABLE IF NOT EXISTS user_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES chat_sessions(id),
    interaction_type VARCHAR(50) NOT NULL, -- 'query', 'click', 'feedback', etc.
    content TEXT,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id
ON user_interactions (user_id);

CREATE INDEX IF NOT EXISTS idx_user_interactions_session_id
ON user_interactions (session_id);

CREATE INDEX IF NOT EXISTS idx_user_interactions_type
ON user_interactions (interaction_type);

CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp
ON user_interactions (timestamp);

-- Table for feedback and ratings
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES chat_sessions(id),
    message_id UUID REFERENCES chat_messages(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5), -- 1-5 star rating
    comment TEXT,
    helpful BOOLEAN, -- Whether the response was helpful
    feedback_type VARCHAR(50) DEFAULT 'general' CHECK (feedback_type IN ('accuracy', 'relevance', 'completeness', 'other')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient querying
CREATE INDEX IF NOT EXISTS idx_feedback_user_id
ON feedback (user_id);

CREATE INDEX IF NOT EXISTS idx_feedback_session_id
ON feedback (session_id);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update the updated_at timestamp
CREATE TRIGGER update_document_chunks_updated_at
    BEFORE UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_book_metadata_updated_at
    BEFORE UPDATE ON book_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data insertion for initial setup
-- This would typically be done during the ingestion process

-- Insert sample book metadata
INSERT INTO book_metadata (chapter_id, title, module, word_count, estimated_reading_time, difficulty, prerequisites, learning_objectives)
VALUES
    ('module-1-ros-intro', 'Introduction to ROS 2', 'Module 1', 1200, 5, 'beginner', '[]', '["Understand what ROS 2 is", "Learn basic ROS 2 concepts"]'),
    ('module-1-ros-architecture', 'ROS 2 Architecture & DDS', 'Module 1', 1500, 6, 'intermediate', '["Basic programming knowledge"]', '["Explain DDS in ROS 2", "Understand ROS 2 architecture"]'),
    ('module-2-gazebo-intro', 'Introduction to Digital Twins', 'Module 2', 1300, 5, 'intermediate', '["Module 1 knowledge"]', '["Understand digital twins", "Learn Gazebo basics"]')
ON CONFLICT (chapter_id) DO NOTHING;

-- Create a view for easy access to chapter information with metadata
CREATE OR REPLACE VIEW chapter_info AS
SELECT
    d.chapter_id,
    d.chapter_title,
    b.module,
    b.word_count,
    b.estimated_reading_time,
    b.difficulty,
    b.prerequisites,
    b.learning_objectives,
    COUNT(m.id) as message_count
FROM document_chunks d
JOIN book_metadata b ON d.chapter_id = b.chapter_id
LEFT JOIN chat_messages m ON m.content LIKE '%' || d.chapter_title || '%'
GROUP BY d.chapter_id, d.chapter_title, b.module, b.word_count, b.estimated_reading_time, b.difficulty, b.prerequisites, b.learning_objectives;

-- Grant permissions (adjust as needed for your Neon setup)
-- In Neon, permissions are typically managed through roles and connection pooling
-- GRANT USAGE ON SCHEMA public TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- Enable Row Level Security (RLS) if needed for multi-tenancy
-- ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Add comments to tables and columns for documentation
COMMENT ON TABLE document_chunks IS 'Stores document chunks for the RAG system with embeddings for similarity search';
COMMENT ON COLUMN document_chunks.embedding IS 'Vector embedding using OpenAI text-embedding-3-large (1536 dimensions)';
COMMENT ON TABLE chat_sessions IS 'Stores chat session information for conversation history';
COMMENT ON TABLE chat_messages IS 'Stores individual messages within chat sessions';
COMMENT ON TABLE book_metadata IS 'Stores metadata about book chapters including difficulty, prerequisites, etc.';