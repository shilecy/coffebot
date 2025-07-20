import pytest
from chatbot_app.chatbot_part4 import run_chatbot_logic

# === Success Cases ===
success_cases = [
    # Calculator
    ("What is 25 * 4 + 100?", ["200", "Answer"]),
    
    # Product RAG
    ("Which bottles are BPA-free?", ["BPA-free", "All Day Cup", "ZUS"]),
    
    # Outlet SQL
    ("List all outlets in Selangor", ["Selangor", "ZUS outlet", "address"]),
]

# === Failure Cases ===
failure_cases = [
    # Calculator
    ("Calculate apple + 5", ["not a number", "only calculate numbers"]),
    
    # Product RAG
    ("Any ZUS products made of gold?", ["does not", "no", "none of the", "no products", "not found", "sorry"]),
    
    # Outlet SQL
    ("Are there any ZUS outlets in Antarctica?", ["no", "there are no", "there is no", "no outlets match", "couldn’t find", "sorry"]),
]

# === Success Tests ===
@pytest.mark.parametrize("user_input, expected_keywords", success_cases)
def test_success_cases(user_input, expected_keywords):
    result = run_chatbot_logic(user_input)
    print(f"\n[✅ DEBUG] Input: {user_input}\nOutput: {result}")
    assert any(k.lower() in result.lower() for k in expected_keywords)

# === Failure Tests ===
@pytest.mark.parametrize("user_input, expected_keywords", failure_cases)
def test_failure_cases(user_input, expected_keywords):
    result = run_chatbot_logic(user_input)
    print(f"\n[❌ FAILURE] Input: {user_input}\nOutput: {result}")
    assert any(k.lower() in result.lower() for k in expected_keywords)
