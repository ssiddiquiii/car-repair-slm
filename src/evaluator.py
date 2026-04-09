from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig

def run_deepeval(results, gemini_api_key):
    print("Running DeepEval...")
    gemini_model = GeminiModel(model="gemini-2.0-flash", api_key=gemini_api_key)

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is factually correct and covers the key points mentioned in the 'expected output' for car repair advice.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=gemini_model
    )

    test_cases = [
        LLMTestCase(
            input=r["question"],
            actual_output=r["actual"],
            expected_output=r["expected"]
        ) for r in results
    ]

    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric],
        async_config=AsyncConfig(run_async=False)
    )
    print("\n--- BASELINE EVALUATION COMPLETE ---")