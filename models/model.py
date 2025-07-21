from dataclasses import dataclass



@dataclass
class EmotionStatus:
    """Data class for student emotional state"""
    stamina: int = 100
    knowledge: int = 50
    stress: int = 50
    happy: int = 50
    sleep: int = 100
    social: int = 50


@dataclass
class AcademicScore:
    """Data class for academic performance"""
    week: int
    topic: str
    score: float
    max_score: float
    correct_answers: int
    total_questions: int