"""
Test suite for OpenAI provider.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llm import get_llm_provider


def test_openai_text():
    """Test OpenAI text generation."""
    print("Testing OpenAI text generation...")
    
    llm = get_llm_provider("openai", model="gpt-4o")
    
    response = llm.call(
        prompt="What is Python?",
        system_prompt="You are a helpful assistant. Keep your response to 1-2 sentences.",
    )
    
    print(f"Response: {response}\n")
    assert len(response) > 0, "Response should not be empty"
    print("✓ OpenAI text generation test passed\n")


def test_openai_json():
    """Test OpenAI JSON mode."""
    print("Testing OpenAI JSON mode...")
    
    llm = get_llm_provider("openai", model="gpt-4o")
    
    prompt = """Extract programming languages from this text:
    "I'm proficient in Python, JavaScript, and C++."
    
    Return as JSON with format: {"languages": [...]}"""
    
    result = llm.call_json(
        prompt=prompt,
        system_prompt="You are a helpful assistant. Return valid JSON only.",
    )
    
    print(f"Result: {result}\n")
    assert isinstance(result, dict), "Result should be a dict"
    assert "languages" in result, "Result should have 'languages' key"
    print("✓ OpenAI JSON mode test passed\n")


def test_openai_skill_extraction():
    """Test skill extraction like the parser will do."""
    print("Testing skill extraction with OpenAI...")
    
    llm = get_llm_provider("openai", model="gpt-4o")
    
    # Example cache
    skills_cache = [
        {"name": "python", "aliases": ["py"], "related": ["scripting"]},
        {"name": "pytorch", "aliases": ["torch"], "related": ["deep learning"]},
    ]
    
    line = "Strong Python skills required; experience with PyTorch or similar ML frameworks."
    
    prompt = f"""Given the following line, extract skills and match them to the cache.
    
Line: "{line}"

Skills Cache:
{skills_cache}

Return JSON with format:
{{
    "extracted_raw_terms": ["skill1", "skill2"],
    "matched_skills": [
        {{
            "raw_term": "...",
            "canonical_name": "...",
            "match_type": "exact|alias|related",
            "confidence": 0.0-1.0,
            "evidence": "..."
        }}
    ]
}}"""
    
    result = llm.call_json(
        prompt=prompt,
        system_prompt="Extract skills and return valid JSON only.",
    )
    
    print(f"Result: {result}\n")
    assert "matched_skills" in result, "Result should have 'matched_skills' key"
    print("✓ Skill extraction test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("OpenAI Provider Tests")
    print("=" * 60 + "\n")
    
    try:
        test_openai_text()
        test_openai_json()
        test_openai_skill_extraction()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
