"""
Unit tests for Clinical Biomarkers and Lifestyle Risk Evaluation.
Tests Gender, HbA1c, Physical Activity, and Smoking Status evaluation logic.
"""
import unittest
from server import evaluate_clinical_biomarkers

class TestClinicalEvaluation(unittest.TestCase):

    def test_gender_evaluation_female(self):
        evals = evaluate_clinical_biomarkers({'Gender': 'Female'})
        gender_eval = next(e for e in evals if e['name'] == 'Biological Sex / Gender')
        self.assertEqual(gender_eval['value'], 'Female')
        self.assertEqual(gender_eval['status'], 'Gestational Profile')

    def test_gender_evaluation_male(self):
        evals = evaluate_clinical_biomarkers({'Gender': 'Male'})
        gender_eval = next(e for e in evals if e['name'] == 'Biological Sex / Gender')
        self.assertEqual(gender_eval['value'], 'Male')
        self.assertEqual(gender_eval['status'], 'Visceral Profile')

    def test_physical_activity_evaluation(self):
        evals_sedentary = evaluate_clinical_biomarkers({'PhysicalActivity': 'Sedentary'})
        act_eval = next(e for e in evals_sedentary if e['name'] == 'Physical Activity & Exercise')
        self.assertEqual(act_eval['status'], 'Sedentary Lifestyle')
        self.assertEqual(act_eval['badge'], 'danger')

        evals_mod = evaluate_clinical_biomarkers({'PhysicalActivity': 'Moderate'})
        act_eval_mod = next(e for e in evals_mod if e['name'] == 'Physical Activity & Exercise')
        self.assertEqual(act_eval_mod['status'], 'Meets ADA Target')
        self.assertEqual(act_eval_mod['badge'], 'success')

    def test_smoking_status_evaluation(self):
        evals_current = evaluate_clinical_biomarkers({'SmokingStatus': 'Current'})
        smk_eval = next(e for e in evals_current if e['name'] == 'Smoking & Tobacco Status')
        self.assertEqual(smk_eval['status'], 'Active Nicotine Risk')
        self.assertEqual(smk_eval['badge'], 'danger')

        evals_never = evaluate_clinical_biomarkers({'SmokingStatus': 'Never'})
        smk_eval_never = next(e for e in evals_never if e['name'] == 'Smoking & Tobacco Status')
        self.assertEqual(smk_eval_never['status'], 'Optimal Profile')
        self.assertEqual(smk_eval_never['badge'], 'success')

    def test_hba1c_glucose_correlation(self):
        evals = evaluate_clinical_biomarkers({'Glucose': 120})
        a1c_eval = next(e for e in evals if 'HbA1c' in e['name'])
        self.assertIn('%', a1c_eval['value'])
        self.assertEqual(a1c_eval['status'], 'Pre-diabetic')

if __name__ == '__main__':
    unittest.main()
