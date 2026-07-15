from llm import LLMProvider


class Parser:
    def __init__(self):
        pass
    
    def parse_posting(self, posting_text):
        pass
    
    def split_posting_into_lines(self, posting_text):
        # Split the posting text into batches of lines (e.g., every x \n and .)
        pass
    
    def parse_posting_line(self, posting_line):
        # expected return
        # SkillExtraction = {
        #     "posting_line": str,
        #     "extracted_raw_terms": [str],  # terms found in the line
        #     "matched_skills": [
        #         {
        #             "raw_term": str,
        #             "canonical_name": str,
        #             "match_type": "exact" | "alias" | "related",
        #             "confidence": 0.0–1.0,
        #             "relevance_score": 1–5,  # how central to the role
        #             "evidence": str,  # why this skill was selected
        #         }
        #     ],
        # }
        pass
    
    def validate_parsed_line(self, extracted_skills):
        # Validate that the extracted skills are valid and relevant, there are no duplicates or miscategorized skills
        pass