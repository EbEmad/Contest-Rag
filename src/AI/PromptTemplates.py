DIFFICULTY_DECISION_PROMPT = """
You are an educational AI analyzing student performance.
Student's recent performance on topic "{topic_name}":
- Last 5 attempts: {attempts}
- Average accuracy: {avg_accuracy}%
- Trend: {trend}
Based on this data, what difficulty level should the next quiz be?
Respond with ONLY one word: EASY, MEDIUM, or HARD.
Rules:
- If accuracy > 80% → MEDIUM or HARD
- If accuracy < 50% → EASY
- If improving trend → increase difficulty
- If declining trend → decrease difficulty
"""


QUESTION_GENERATION_PROMPT = """
You are an educational content creator.
Based on the following curriculum content:
---
{content}
---
Generate {count} {difficulty} level {question_type} questions for grade {grade} students.
Requirements:
1. Questions MUST be answerable from the provided content
2. For MCQ: provide exactly 4 options with only 1 correct answer
3. Include the correct answer
4. Keep appropriate for grade level
Return as JSON array:
[
  {{
    "question_text": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "...",
    "explanation": "..."
  }}
]
"""

ROADMAP_GENERATION_PROMPT = """
You are an expert educational advisor creating a personalized 2-week study plan.
Student Profile:
- Grade: {grade}
- Current Level: {level}
Weak Topics Identified:
{weak_topics}
Create a detailed study plan with:
1. Top 3 weakest topics prioritized
2. Daily activities (15-30 min each)
3. Time estimates
4. Encouragement and motivation
Format as markdown.
"""