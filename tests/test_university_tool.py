import unittest
from tools.university_search_tool import _execute_university_search, classify_query

class TestUniversityTool(unittest.TestCase):
    def test_fee_query_classification(self):
        query = "How much is the B.Tech tuition fee per semester?"
        category = classify_query(query)
        self.assertEqual(category, "fees")
        
        result = _execute_university_search(query)
        self.assertEqual(result["category_detected"], "fees")
        self.assertGreater(len(result["answer_chunks"]), 0)
        self.assertIn("tuition", result["answer_chunks"][0]["text"].lower())

    def test_exam_query_classification(self):
        query = "When is the Artificial Intelligence theory exam?"
        category = classify_query(query)
        self.assertEqual(category, "exams")
        
        result = _execute_university_search(query)
        self.assertEqual(result["category_detected"], "exams")
        self.assertGreater(len(result["answer_chunks"]), 0)
        self.assertIn("exam", result["answer_chunks"][0]["text"].lower())

    def test_library_query_classification(self):
        query = "Can I borrow reference books from the library?"
        category = classify_query(query)
        self.assertEqual(category, "library")
        
        result = _execute_university_search(query)
        self.assertEqual(result["category_detected"], "library")
        self.assertGreater(len(result["answer_chunks"]), 0)
        self.assertIn("library", result["answer_chunks"][0]["text"].lower())

    def test_hostel_query_classification(self):
        query = "What is the curfew time for hostel residents?"
        category = classify_query(query)
        self.assertEqual(category, "hostel")
        
        result = _execute_university_search(query)
        self.assertEqual(result["category_detected"], "hostel")
        self.assertGreater(len(result["answer_chunks"]), 0)
        self.assertIn("hostel", result["answer_chunks"][0]["text"].lower())

    def test_calendar_query_classification(self):
        query = "When does the Diwali vacation start?"
        category = classify_query(query)
        self.assertEqual(category, "academic-calendar")
        
        result = _execute_university_search(query)
        self.assertEqual(result["category_detected"], "academic-calendar")
        self.assertGreater(len(result["answer_chunks"]), 0)
        self.assertIn("vacation", result["answer_chunks"][0]["text"].lower())

if __name__ == "__main__":
    unittest.main()
