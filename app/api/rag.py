import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.document import DocumentUploadResponse, QueryRequest, QueryResponse
from app.models.conversation import ConversationRequest, ConversationResponse
from app.models.user import User
from app.services.auth import get_current_user
from app.services.rag import AIProviderError, rag_service
from app.utils.pdf import extract_text_from_pdf
from app.database import get_db
from app.core.config import settings
import uuid

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
        file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a PDF document"""
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        max_upload_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if file_size > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF files must be {settings.MAX_UPLOAD_MB} MB or smaller"
            )

        text = extract_text_from_pdf(file.file, max_pages=settings.MAX_PDF_PAGES)

        if not text.strip():
            raise HTTPException(status_code=400, detail="PDF contains no extractable text")

        result = rag_service.add_document(text, file.filename, current_user.email)

        return DocumentUploadResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunks_created=result["chunks_created"],
            message="Document uploaded successfully"
        )
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF or PDF exceeds the configured page limit"
        )
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Unable to process document")


@router.get("/documents")
async def list_documents(current_user: User = Depends(get_current_user)):
    """List all user documents"""
    try:
        docs = rag_service.get_user_documents(current_user.email)
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail="Unable to fetch documents")


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, current_user: User = Depends(get_current_user)):
    """Delete a document"""
    try:
        success = rag_service.delete_document(doc_id, current_user.email)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document deleted successfully", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail="Unable to delete document")


@router.post("/query", response_model=QueryResponse)
async def query_documents(
        query: QueryRequest,
        current_user: User = Depends(get_current_user)
):
    """Query without conversation history"""
    try:
        result = rag_service.query(query.question, query.top_k, current_user.email)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except AIProviderError:
        raise HTTPException(status_code=502, detail="AI provider unavailable")
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail="Unable to process query")


@router.post("/chat", response_model=ConversationResponse)
async def chat_with_documents(
        request: ConversationRequest,
        current_user: User = Depends(get_current_user)
):
    """Chat with conversation memory"""
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            # Create conversation in DB
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (id, user_email) VALUES (?, ?)",
                    (conversation_id, current_user.email)
                )
                conn.commit()
        else:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT 1 FROM conversations WHERE id = ? AND user_email = ?",
                    (conversation_id, current_user.email)
                )
                if not cursor.fetchone():
                    raise HTTPException(status_code=404, detail="Conversation not found")

        result = rag_service.query_with_conversation(
            request.question,
            conversation_id,
            current_user.email,
            request.top_k
        )

        return ConversationResponse(
            answer=result["answer"],
            sources=result["sources"],
            conversation_id=conversation_id
        )
    except AIProviderError:
        raise HTTPException(status_code=502, detail="AI provider unavailable")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG chat failed")
        raise HTTPException(status_code=500, detail="Unable to process chat")


@router.get("/conversations")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """List all user conversations"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT c.id, c.created_at, 
                   (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at ASC LIMIT 1) as first_message
                   FROM conversations c 
                   WHERE c.user_email = ? 
                   ORDER BY c.created_at DESC""",
                (current_user.email,)
            )
            rows = cursor.fetchall()
            conversations = [
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "preview": row["first_message"][:100] + "..." if row["first_message"] else "Empty conversation"
                }
                for row in rows
            ]
            return {"conversations": conversations, "count": len(conversations)}
    except Exception as e:
        logger.exception("Conversation listing failed")
        raise HTTPException(status_code=500, detail="Unable to fetch conversations")


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
        conversation_id: str,
        current_user: User = Depends(get_current_user)
):
    """Get all messages in a conversation"""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            # Verify ownership
            cursor.execute(
                "SELECT user_email FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if not row or row["user_email"] != current_user.email:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Get messages
            cursor.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,)
            )
            messages = [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"]
                }
                for row in cursor.fetchall()
            ]
            return {"messages": messages, "count": len(messages)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Conversation messages fetch failed")
        raise HTTPException(status_code=500, detail="Unable to fetch conversation messages")
