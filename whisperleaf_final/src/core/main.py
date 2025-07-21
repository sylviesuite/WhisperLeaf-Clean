"""
Main FastAPI application for Sovereign AI system.
"""

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import os
import tempfile
from pathlib import Path

from database import get_db, init_database
from models import Document, User
from vault import VaultManager
from vector_store import VectorStore
from document_processor import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(
    title="Sovereign AI API",
    description="API for the Sovereign AI system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vault_manager = VaultManager()
vector_store = VectorStore()
document_processor = DocumentProcessor()

@app.on_event("startup")
async def startup_event():
    """Initialize database and components on startup."""
    init_database()
    print("Sovereign AI API server started successfully")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Sovereign AI API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "components": ["vault", "vector_store", "database"]}

# Vault Management Endpoints

@app.post("/api/vault/documents")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    tags: Optional[str] = Form(None),
    user_id: str = Form("default_user"),  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Upload a document to the vault."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process tags
            tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
            
            # Add document to vault
            document = vault_manager.add_document(
                file_path=temp_file_path,
                title=title,
                user_id=user_id,
                db=db,
                tags=tag_list,
                metadata={"original_filename": file.filename}
            )
            
            # Process document content for vector store
            processed_doc = document_processor.process_document(document.file_path)
            
            if processed_doc["processing_status"] == "success" and processed_doc["text"]:
                # Add to vector store
                vector_store.add_document(
                    document_id=document.id,
                    content=processed_doc["text"],
                    metadata={
                        "title": document.title,
                        "content_type": document.content_type,
                        "tags": document.tags,
                        "word_count": processed_doc.get("word_count", 0)
                    }
                )
            
            return {
                "id": document.id,
                "title": document.title,
                "content_type": document.content_type,
                "file_size": document.file_size,
                "processing_status": processed_doc["processing_status"],
                "word_count": processed_doc.get("word_count", 0)
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/vault/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    tags: Optional[str] = None,
    content_type: Optional[str] = None,
    user_id: str = "default_user",  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """List documents in the vault."""
    try:
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        documents = vault_manager.list_documents(
            user_id=user_id,
            db=db,
            skip=skip,
            limit=limit,
            tags=tag_list,
            content_type=content_type
        )
        
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content_type": doc.content_type,
                "file_size": doc.file_size,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "tags": doc.tags,
                "metadata": doc.metadata_json
            }
            for doc in documents
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.get("/api/vault/documents/{document_id}")
async def get_document(
    document_id: str,
    user_id: str = "default_user",  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Get document details."""
    try:
        document = vault_manager.get_document(document_id, user_id, db)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "title": document.title,
            "content_type": document.content_type,
            "file_size": document.file_size,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "source_url": document.source_url,
            "source_type": document.source_type,
            "tags": document.tags,
            "metadata": document.metadata_json,
            "content_hash": document.content_hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")

@app.delete("/api/vault/documents/{document_id}")
async def delete_document(
    document_id: str,
    user_id: str = "default_user",  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Delete a document from the vault."""
    try:
        # Remove from vector store first
        vector_store.remove_document(document_id)
        
        # Remove from vault
        success = vault_manager.delete_document(document_id, user_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Search Endpoints

@app.post("/api/vault/search")
async def search_documents(
    query: str = Form(...),
    n_results: int = Form(10),
    document_ids: Optional[str] = Form(None),
    user_id: str = Form("default_user")  # TODO: Get from authentication
):
    """Search documents using semantic similarity."""
    try:
        doc_ids = [doc_id.strip() for doc_id in document_ids.split(",")] if document_ids else None
        
        results = vector_store.search(
            query=query,
            n_results=n_results,
            document_ids=doc_ids
        )
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# Statistics Endpoints

@app.get("/api/vault/statistics")
async def get_vault_statistics(
    user_id: str = "default_user",  # TODO: Get from authentication
    db: Session = Depends(get_db)
):
    """Get vault statistics."""
    try:
        vault_stats = vault_manager.get_vault_statistics(user_id, db)
        vector_stats = vector_store.get_collection_stats()
        
        return {
            "vault": vault_stats,
            "vector_store": vector_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# Document Processing Endpoints

@app.get("/api/processing/supported-types")
async def get_supported_types():
    """Get supported document types."""
    return {
        "supported_types": document_processor.get_supported_types(),
        "vault_allowed_extensions": list(vault_manager.allowed_extensions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

