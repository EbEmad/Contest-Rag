"""
AI Topic Detector
Automatically detects which curriculum topics a book covers.
"""

from typing import List
from motor.motor_asyncio import AsyncIOMotorDatabase
from AI.llm.LLMInterface import LLMInterface
from bson.objectid import ObjectId
import logging

logger = logging.getLogger(__name__)

TOPIC_DETECTION_PROMPT = """
You are an educational content analyzer.
Available topics in our curriculum:
{available_topics}
Book content sample:
---
{book_sample}
---
Task: Identify which topics from our curriculum this book covers.
Rules:
1. Only return topics that are EXPLICITLY covered in the sample
2. Return topic names EXACTLY as they appear in the available topics list
3. Return as a JSON array of topic names
Example response:
["Linear Equations", "Quadratic Equations"]
Your response (JSON array only):
"""

class TopicDetector:
    """AI agent for detecting topics in educational content"""
    def __init__(self,llm_client:LLMInterface,db:AsyncIOMotorDatabase):
        self.llm=llm_client
        self.db=db
    async def detect_topics(
        self,
        book_content: str,
        grade: int,
        subject: str,
        sample_size: int = 3000
    ) -> List[ObjectId]:
        """
        Detect which topics a book covers using AI.
        
        Args:
            book_content: Full text of the book
            grade: Grade level (e.g., 10)
            subject: Subject name (e.g., "Mathematics")
            sample_size: Number of characters to sample from book
            
        Returns:
            List of topic ObjectIds
        """
        # Get available topics for this subject
        subject_doc=await self.db.subjects.find_one({
            "name": subject,
            "grade": grade
        })

        if not subject_doc:
            logger.warning(f"Subject '{subject}' Grade {grade} not found")
            return []
        
        # Get all topics for this subject
        topics = await self.db.topics.find({
            "subject_id": subject_doc["_id"]
        }).to_list(length=100)

        if not topics:
            logger.warning(f"No topics found for {subject}")
            return []
        
        # Format available topics
        topic_names=[t["name"] for t in topics]

        available_topics_str="\n".join([f"- {name}" for name in topic_names])

        # Sample book content (first N characters)
        book_sample=book_content[:sample_size]

        # Ask LLM to detect topics
        prompt=TOPIC_DETECTION_PROMPT.format(
            available_topics=available_topics_str,
            book_sample=book_sample
        )

        logger.info(f"Detecting topics for {subject} Grade {grade}...")

        try:
            response

        

