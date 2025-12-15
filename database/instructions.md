To create the DB, I compose up'd the docker_compose.yaml.

docker compuse up -d


Then I connected to the database on pgadmin and ran these on the Query Editor.

CREATE TABLE rag_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    case_name TEXT,
    case_type TEXT,
    tags JSONB,
    summary_llm TEXT,
    page_number INTEGER,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536)
	
);

CREATE INDEX ON rag_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX ON rag_chunks (document_id);

CREATE INDEX ON rag_chunks (case_type);

CREATE INDEX tags_idx ON rag_chunks USING GIN (tags);




You may use local_connect.py to check if locally run ollama works well. Simple and quick. 