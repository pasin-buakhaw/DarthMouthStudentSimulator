"""
Student Life Pipeline - Main pipeline class
"""
import os
import sys
import re
import glob
import pandas as pd
import csv
import json
from typing import Dict, List, Tuple


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import LLMClient
from scripts.tools import BigFiveAnalyzer, AcademicEvaluator, StudentAgent
from models.model import EmotionStatus, AcademicScore


class StudentLifePipeline:
    """Main pipeline class that orchestrates the entire analysis"""
    
    def __init__(self, llm_client: LLMClient, config: Dict):
        self.llm_client = llm_client
        self.config = config
        self.big_five_analyzer = BigFiveAnalyzer()
        self.academic_evaluator = AcademicEvaluator(llm_client)
        
        self._load_data()
    
    def _load_data(self):
        """Load all necessary data files"""
        self.trait_df = self.big_five_analyzer.compute_bigfive_scores(self.config["big_five_path"])
        
        self.class_df = self._load_class_data(self.config["class_csv_path"])
        with open(self.config["class_info_path"], 'r') as f:
            self.class_info = json.load(f)
        
   
        self.exam_df = pd.read_csv(self.config["exam_path"])
        

        self.deadline_df = pd.read_csv(self.config["deadline_path"])
    
    def _load_class_data(self, csv_path: str) -> pd.DataFrame:
        """Load and process class data"""
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:
                    rows.append(row)
        
        max_cols = max(len(r) for r in rows)
        colnames = ["uid"] + [f"c{i}" for i in range(1, max_cols)]
        rows_padded = [r + [None] * (max_cols - len(r)) for r in rows]
        
        return pd.DataFrame(rows_padded, columns=colnames)
    
    def _get_user_class_map(self, uid: str) -> Dict:
        """Get class information for a specific user"""
        user_row = self.class_df[self.class_df["uid"] == uid]
        if user_row.empty:
            return {}
        
      
        if len(user_row.columns) < 2:
            return {}
        
        class_list = user_row.iloc[0, 1:].dropna().tolist()
        return {
            cls: self.class_info.get(cls, {"location": "unknown", "schedule": "TBA", "instructor": "Unknown"})
            for cls in class_list
        }
    
    def _get_deadline_dates_and_counts(self, uid: str) -> List[Tuple[str, int]]:
        """Get deadline information for a specific user"""
        row = self.deadline_df[self.deadline_df['uid'] == uid]
        if row.empty:
            return []
        
   
        if len(row.columns) < 2:
            return []
        
        date_series = row.drop(columns=['uid']).iloc[0]
        valid_dates = date_series[date_series > 0]
        return list(valid_dates.items())
    
    def _format_deadline(self, deadline_data: List[Tuple[str, int]]) -> List[str]:
        """Format deadline data for display"""
        formatted = []
        for date_str, count in deadline_data:
            dt = pd.to_datetime(date_str + ' 23:59')
            formatted.append(f"{dt.strftime('%a %H:%M')} (count: {int(count)})")
        return formatted
    
    def process_student_week(self, uid: str, week_num: int, week_data_path: str, 
                           class_data_path: str = None) -> Dict:
        """Process a single week of student data"""
        print(f"\n=== 🗂 Processing Week {week_num} | File: {os.path.basename(week_data_path)} ===")
        
        try:
    
            week_df = pd.read_csv(week_data_path)
            print(f"📊 Loaded week data with {len(week_df)} rows")
            
          
            big_five = self.big_five_analyzer.get_bigfive_dict(self.trait_df, uid=uid, type_='pre')
            print(f"👤 Loaded Big Five data for {uid}")
            
            class_info = self._get_user_class_map(uid)
            print(f"📚 Found {len(class_info)} classes for {uid}")
            
            deadline_data = self._get_deadline_dates_and_counts(uid)
            deadline_text = self._format_deadline(deadline_data)
            print(f"📅 Found {len(deadline_data)} deadlines for {uid}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
        
       
        student = StudentAgent(big_five, class_info, self.llm_client)
        
        emotion_history_path = f"{uid}_emotion_status_history.jsonl"
        if os.path.exists(emotion_history_path):
            with open(emotion_history_path, "r") as f:
                lines = f.readlines()
                if lines:
                    latest_status = json.loads(lines[-1])["emotion"]
                    student.emotion_status = EmotionStatus(**latest_status)
        
   
        student.load_real_week_data(week_df)
        
     
        if class_data_path and os.path.exists(class_data_path):
            class_df = pd.read_csv(class_data_path)
            student.set_weekly_class_experience(class_df)
        else:
            print(f"⚠️ No class experience data for week {week_num}. Filling with blank.")
            student.weekly_data['class_experience'] = ["(No class experience recorded this week.)"]
        

        journal = student.generate_journal_entry(deadline_text)

        
        emotion_status, reasoning = student.analyze_emotion(journal)

        try:
            academic_score = self.academic_evaluator.evaluate_weekly_exam(
                self.exam_df, week_num, emotion_status, big_five
            )


            print(f"📝 Academic evaluation completed for week {week_num}")
        except Exception as e:
            print(f"❌ Error in academic evaluation: {e}")
       
            academic_score = AcademicScore(
                week=week_num,
                topic="Error in evaluation",
                score=0,
                max_score=0,
                correct_answers=0,
                total_questions=0
            )

        if week_num == 10:
            idea_submission = student.generate_project_submission()
            project_result = self.academic_evaluator.evaluate_project_idea(submission=idea_submission)
            result = {
                "week": week_num,
                "emotion": emotion_status.__dict__,
                "lab_assessment": {
                    "score": academic_score.score,
                    "max_score": academic_score.max_score,
                    "topic": academic_score.topic,
                    "correct_answers": academic_score.correct_answers,
                    "total_questions": academic_score.total_questions,
                    "week":academic_score.week,
                },
                "project": {
                    "score": project_result["score"],
                    "full_text_response": project_result["feedback"],
                },
                "weekly_desc": journal,
                "judge_reasoning": reasoning,
            }
            

        else:
            result = {
                "week": week_num,
                "emotion": emotion_status.__dict__,
                "lab_assessment": {
                    "score": academic_score.score,
                    "max_score": academic_score.max_score,
                    "topic": academic_score.topic,
                    "correct_answers": academic_score.correct_answers,
                    "total_questions": academic_score.total_questions,
                    "week":academic_score.week,
                },
                "weekly_desc": journal,
                "judge_reasoning": reasoning,
            }
        
        # Save results
        with open(emotion_history_path, "a") as f:
            json.dump(result, f)
            f.write("\n")
        
        with open(f"{uid}_emotion_status.json", "w") as f:
            json.dump(emotion_status.__dict__, f)
        
        return result
    
    def run_full_pipeline(self, uid: str, weeks_range: range = range(1, 11)) -> List[Dict]:
        """Run the complete pipeline for multiple weeks"""
        results = []
        

        week_paths = [f"./student_info/{uid}/data_per_week{i}.csv" for i in range(1,11)]#sorted(glob.glob(f"{uid}_test/data_per_week*.csv"))
        class_paths = [f"./student_info/{uid}/class_1_week{i}.csv" for i in range(1,11)]#sorted(glob.glob(f"{uid}_test/class_1_week*.csv"))
        
   
        week_to_class = {}
        for path in class_paths:
            match = re.search(r'class_1_week(\d+)', path)
            if match:
                week_to_class[int(match.group(1))] = path
        
        for week_path in week_paths:
            match = re.search(r'data_per_week(\d+)', week_path)
            if match:
                week_num = int(match.group(1))
                if week_num in weeks_range:
                    class_data_path = week_to_class.get(week_num)
                    result = self.process_student_week(uid, week_num, week_path, class_data_path)
                    results.append(result)
        
        return results
    
    def evaluate_project_submission(self, submission: str) -> Dict:
        """Evaluate a project idea submission"""
        return self.academic_evaluator.evaluate_project_idea(submission)