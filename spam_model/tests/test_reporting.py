import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spam_model.reports.report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ReportGenerator()
        
    def test_risk_scoring(self):
        self.assertEqual(self.generator.calculate_risk_level(0.1), "Low")
        self.assertEqual(self.generator.calculate_risk_level(0.4), "Medium")
        self.assertEqual(self.generator.calculate_risk_level(0.9), "High")
        
    def test_info_detection_otp(self):
        text = "sure, my otp is 123456"
        info = self.generator.detect_shared_info(text)
        self.assertTrue(info['otp'])
        
    def test_info_detection_none(self):
        text = "I will not share that information"
        info = self.generator.detect_shared_info(text)
        self.assertFalse(any(info.values()))
        
    def test_full_report_generation(self):
        data, text = self.generator.generate_report(
            "unittest_call",
            0.95,
            "scam agent text",
            "victim text with pin 1234",
            ["Urgency"]
        )
        self.assertEqual(data['risk_level'], "High")
        self.assertEqual(data['caller_classification'], "Likely Scam")
        self.assertTrue(data['information_shared']['pin'])
        self.assertIn("Customer revealed PIN", data['risk_indicators'])
        
if __name__ == '__main__':
    unittest.main()
