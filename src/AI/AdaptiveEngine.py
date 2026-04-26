from typing import List,Dict
from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from .PromptTemplates import DIFFICULTY_DECISION_PROMPT,ROADMAP_GENERATION_PROMPT
from AI.llm.LLMInterface import LLMInterface
from models.db_schemes.quiz import DifficultyLevel
from models.db_schemes.performance import LearningRoadmap
import logging
logger = logging.getLogger(__name__)

class AdaptiveEngine:
    """AI agent for adaptive learning decisions"""

    def __init__(self,llm_client:LLMInterface,db:AsyncIOMotorDatabase):
        self.llm=llm_client
        self.db=db
    
    async def determine_next_difficulty(self,student_id: ObjectId,topic_id: ObjectId)-> DifficultyLevel:
        """Uses LLM to decide next quiz difficulty based on performance"""

        cursor=self.db.quiz_attempts.find({
            "student_id": student_id,
            "topic_id": topic_id
        }).sort("completed_at", -1).limit(5)

        attempts=await cursor.to_list(length=5)

        if not attempts:
            return DifficultyLevel.EASY
        

        # Calculate stats
        accuracies=[
            a["score"] / a["max_score"] 
            for a in attempts 
            if a.get("score") is not None
        ]

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

        # Determine trend
        trend = "IMPROVING" if len(accuracies) >= 2 and accuracies[0] > accuracies[-1] else "STABLE"

        # Get topic name
        topic = await self.db.topics.find_one({"_id": topic_id})

        # Ask LLM
        prompt = DIFFICULTY_DECISION_PROMPT.format(
            topic_name=topic["name"],
            attempts=[f"{a.get('score', 0)}/{a.get('max_score', 0)}" for a in attempts],
            avg_accuracy=f"{avg_accuracy*100:.1f}",
            trend=trend
        )
        
        response = await self.llm.generate_text(prompt)

        # Parse response
        decision = response.strip().upper()
        for level in ["EASY", "MEDIUM", "HARD"]:
            if level in decision:
                return DifficultyLevel(level)
        
        return DifficultyLevel.MEDIUM  # Default fallback
    async def generate_roadmap(self, student_id: ObjectId) -> LearningRoadmap:
        """Generates personalized learning roadmap"""
        # Get student profile
        profile = await self.db.student_profiles.find_one({"user_id": student_id})

        # Get weak topics (simplified - you'd implement full detection)
        performances=await self.db.topic_performances.find({
            "student_id": student_id,
            "accuracy": {"$lt": 0.6}
        }).to_list(length=10)

        weak_topics = []
        for perf in performances:
            topic=await self.db.topics.find_one({"_id":perf["topic_id"]})
            weak_topics.append(f"- {topic['name']}: {perf['accuracy']*100:.0f}% accuracy")

        # Generate roadmap
        prompt = ROADMAP_GENERATION_PROMPT.format(
            grade=profile.get("grade", "N/A"),
            level=profile.get("current_level", "BEGINNER"),
            weak_topics="\n".join(weak_topics) if weak_topics else "No weak topics identified"
        )

        roadmap_text = await self.llm.generate_text(prompt)
        
        # Save roadmap
        roadmap = LearningRoadmap(
            student_id=student_id,
            weak_topic_ids=[p["topic_id"] for p in performances],
            llm_explanation=roadmap_text
        )
        
        result = await self.db.learning_roadmaps.insert_one(roadmap.dict(by_alias=True))
        roadmap.id = result.inserted_id
        
        return roadmap