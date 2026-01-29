# insurance_ai/services/nlp/ollama_comparator.py
"""
Insurance Policy Comparison Service using local LLM (Ollama).

Compares two insurance policies side-by-side, extracting key comparison
points in a structured format suitable for table display.
"""

import requests
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .extractive_summarizer import extract_key_information


@dataclass
class ComparatorConfig:
    """Configuration for the comparator."""
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.2  # Low for factual extraction
    num_ctx: int = 8192
    num_predict: int = 3000


_config = ComparatorConfig()


def configure_comparator(
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    temperature: Optional[float] = None
):
    """Configure the comparator settings."""
    global _config
    if model:
        _config.model = model
    if ollama_url:
        _config.ollama_url = ollama_url
    if temperature is not None:
        _config.temperature = temperature


def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Make a request to Ollama API."""
    payload = {
        "model": _config.model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": _config.temperature,
            "num_ctx": _config.num_ctx,
            "num_predict": _config.num_predict,
        }
    }

    response = requests.post(
        f"{_config.ollama_url}/api/generate",
        json=payload,
        timeout=180
    )
    response.raise_for_status()
    return response.json().get("response", "")


def check_ollama_available() -> Tuple[bool, str]:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{_config.ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding"

        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        target_base = _config.model.split(":")[0]
        if not any(target_base in name for name in model_names):
            return False, f"Model '{_config.model}' not found"

        return True, "OK"
    except requests.exceptions.ConnectionError:
        return False, "Ollama not running"
    except Exception as e:
        return False, str(e)


# Comparison categories - these will be the rows in the comparison table
COMPARISON_CATEGORIES = [
    "Policy Type",
    "Coverage Scope",
    "Premium Amount",
    "Deductible",
    "Co-payment/Co-insurance",
    "Out-of-Pocket Maximum",
    "Coverage Limit (Per Incident)",
    "Coverage Limit (Annual/Lifetime)",
    "In-Network Benefits",
    "Out-of-Network Benefits",
    "Waiting Period",
    "Pre-existing Conditions",
    "Key Exclusions",
    "Claim Filing Process",
    "Claim Deadline",
    "Cancellation Policy",
    "Renewal Terms",
    "Grace Period",
    "Key Warnings/Red Flags",
    "Special Benefits/Riders"
]


SYSTEM_PROMPT = """You are an expert insurance policy analyst. Your task is to extract specific information from insurance documents for comparison purposes.

CRITICAL RULES:
1. Extract ONLY information that is explicitly stated in the document
2. If information is not found, respond with "Not specified" for that field
3. Be precise - include exact numbers, percentages, dates, and conditions
4. Keep each response concise but complete (1-3 sentences max per field)
5. Do NOT make assumptions or infer information not explicitly stated"""


EXTRACTION_PROMPT = """Analyze this insurance policy document and extract information for each of the following categories.

DOCUMENT TEXT:
{document_text}

---

For each category below, extract the relevant information from the document. If the information is not found, write "Not specified".

Categories to extract:
1. Policy Type - What type of insurance is this? (health, auto, life, home, etc.)
2. Coverage Scope - Brief description of what is covered
3. Premium Amount - Monthly/annual premium cost
4. Deductible - Deductible amount and type (per-incident or annual)
5. Co-payment/Co-insurance - Co-pay amounts or co-insurance percentages
6. Out-of-Pocket Maximum - Maximum annual out-of-pocket expense
7. Coverage Limit (Per Incident) - Maximum payout per claim/incident
8. Coverage Limit (Annual/Lifetime) - Annual or lifetime maximum coverage
9. In-Network Benefits - Coverage for in-network providers
10. Out-of-Network Benefits - Coverage for out-of-network providers
11. Waiting Period - Any waiting periods before coverage begins
12. Pre-existing Conditions - How pre-existing conditions are handled
13. Key Exclusions - Main things NOT covered (list top 3-5)
14. Claim Filing Process - How to file a claim (brief)
15. Claim Deadline - Deadline for filing claims
16. Cancellation Policy - How policy can be cancelled, any penalties
17. Renewal Terms - How/when policy renews
18. Grace Period - Grace period for late payments
19. Key Warnings/Red Flags - Important limitations or conditions that could cause claim denial
20. Special Benefits/Riders - Any additional benefits or special coverages included

Respond in this EXACT JSON format:
{{
    "Policy Type": "extracted value or Not specified",
    "Coverage Scope": "extracted value or Not specified",
    "Premium Amount": "extracted value or Not specified",
    "Deductible": "extracted value or Not specified",
    "Co-payment/Co-insurance": "extracted value or Not specified",
    "Out-of-Pocket Maximum": "extracted value or Not specified",
    "Coverage Limit (Per Incident)": "extracted value or Not specified",
    "Coverage Limit (Annual/Lifetime)": "extracted value or Not specified",
    "In-Network Benefits": "extracted value or Not specified",
    "Out-of-Network Benefits": "extracted value or Not specified",
    "Waiting Period": "extracted value or Not specified",
    "Pre-existing Conditions": "extracted value or Not specified",
    "Key Exclusions": "extracted value or Not specified",
    "Claim Filing Process": "extracted value or Not specified",
    "Claim Deadline": "extracted value or Not specified",
    "Cancellation Policy": "extracted value or Not specified",
    "Renewal Terms": "extracted value or Not specified",
    "Grace Period": "extracted value or Not specified",
    "Key Warnings/Red Flags": "extracted value or Not specified",
    "Special Benefits/Riders": "extracted value or Not specified"
}}

IMPORTANT: Respond with ONLY the JSON object, no other text."""


def _extract_document_info(text: str) -> Dict[str, str]:
    """Extract comparison information from a single document."""
    # First, extract key sentences to reduce document size
    key_sentences = extract_key_information(text, num_sentences=100)
    condensed_text = "\n\n".join(key_sentences)

    # If still too long, truncate
    max_chars = 12000
    if len(condensed_text) > max_chars:
        condensed_text = condensed_text[:max_chars] + "\n\n[Document truncated...]"

    prompt = EXTRACTION_PROMPT.format(document_text=condensed_text)

    result = _call_ollama(prompt, SYSTEM_PROMPT)

    # Parse JSON from response
    try:
        # Find JSON in response (handle cases where model adds extra text)
        json_start = result.find('{')
        json_end = result.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = result[json_start:json_end]
            extracted = json.loads(json_str)

            # Ensure all categories are present
            for category in COMPARISON_CATEGORIES:
                if category not in extracted:
                    extracted[category] = "Not specified"

            return extracted
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {result[:500]}...")

    # If JSON parsing fails, return empty dict with all categories
    return {category: "Extraction failed" for category in COMPARISON_CATEGORIES}


VERDICT_PROMPT = """Based on the comparison data below, provide a brief verdict on which policy might be better and why.

POLICY 1 DATA:
{policy1_data}

POLICY 2 DATA:
{policy2_data}

Provide a brief analysis (3-5 sentences) covering:
1. Which policy offers better value overall
2. Key advantages of Policy 1
3. Key advantages of Policy 2
4. Which policy is better for different situations (e.g., "Policy 1 is better for those who want lower premiums, while Policy 2 is better for comprehensive coverage")

Keep the analysis balanced and factual based on the extracted data."""


def compare_policies(
    text1: str,
    text2: str,
    hash1: str = "policy1",
    hash2: str = "policy2",
    include_verdict: bool = True
) -> Dict:
    """
    Compare two insurance policies side-by-side.

    Args:
        text1: Normalized text of first policy
        text2: Normalized text of second policy
        hash1: Identifier for first policy
        hash2: Identifier for second policy
        include_verdict: Whether to include an AI verdict/recommendation

    Returns:
        Dictionary with comparison data structured for table display:
        {
            "categories": ["Category1", "Category2", ...],
            "policy1": {
                "hash": "hash1",
                "values": ["value1", "value2", ...]
            },
            "policy2": {
                "hash": "hash2",
                "values": ["value1", "value2", ...]
            },
            "verdict": "AI analysis..." (optional)
        }
    """
    available, message = check_ollama_available()
    if not available:
        raise RuntimeError(f"Ollama not available: {message}")

    print(f"Extracting comparison data from policy 1 ({hash1})...")
    info1 = _extract_document_info(text1)

    print(f"Extracting comparison data from policy 2 ({hash2})...")
    info2 = _extract_document_info(text2)

    # Build parallel arrays for table display
    categories = COMPARISON_CATEGORIES
    values1 = [info1.get(cat, "Not specified") for cat in categories]
    values2 = [info2.get(cat, "Not specified") for cat in categories]

    result = {
        "categories": categories,
        "policy1": {
            "hash": hash1,
            "values": values1
        },
        "policy2": {
            "hash": hash2,
            "values": values2
        },
        "model": _config.model
    }

    # Generate verdict if requested
    if include_verdict:
        print("Generating comparison verdict...")
        try:
            policy1_summary = "\n".join([f"- {cat}: {val}" for cat, val in zip(categories, values1)])
            policy2_summary = "\n".join([f"- {cat}: {val}" for cat, val in zip(categories, values2)])

            verdict_prompt = VERDICT_PROMPT.format(
                policy1_data=policy1_summary,
                policy2_data=policy2_summary
            )

            verdict = _call_ollama(verdict_prompt, SYSTEM_PROMPT)
            result["verdict"] = verdict.strip()
        except Exception as e:
            print(f"Warning: Failed to generate verdict: {e}")
            result["verdict"] = "Unable to generate verdict."

    # Also provide a "highlights" section for quick differences
    result["highlights"] = _identify_key_differences(info1, info2)

    return result


def _identify_key_differences(info1: Dict, info2: Dict) -> List[Dict]:
    """Identify key differences between two policies."""
    highlights = []

    # Categories to highlight differences
    key_categories = [
        ("Premium Amount", "cost", "lower is typically better"),
        ("Deductible", "cost", "lower deductible means less out-of-pocket per claim"),
        ("Out-of-Pocket Maximum", "cost", "lower maximum limits your risk"),
        ("Coverage Limit (Annual/Lifetime)", "coverage", "higher limit provides more protection"),
        ("Key Exclusions", "coverage", "fewer exclusions means broader coverage"),
        ("Waiting Period", "coverage", "shorter waiting period is better"),
        ("Key Warnings/Red Flags", "risk", "important limitations to consider"),
    ]

    for category, category_type, note in key_categories:
        val1 = info1.get(category, "Not specified")
        val2 = info2.get(category, "Not specified")

        if val1 != val2 and val1 != "Not specified" and val2 != "Not specified":
            highlights.append({
                "category": category,
                "type": category_type,
                "policy1": val1,
                "policy2": val2,
                "note": note
            })

    return highlights


def quick_compare(text1: str, text2: str, hash1: str = "policy1", hash2: str = "policy2") -> Dict:
    """
    Quick comparison focusing only on the most important differences.
    Faster than full comparison.
    """
    available, message = check_ollama_available()
    if not available:
        raise RuntimeError(f"Ollama not available: {message}")

    # Extract fewer sentences for speed
    key1 = extract_key_information(text1, num_sentences=50)
    key2 = extract_key_information(text2, num_sentences=50)

    condensed1 = "\n".join(key1[:30])
    condensed2 = "\n".join(key2[:30])

    prompt = f"""Compare these two insurance policies and identify the TOP 10 most important differences.

POLICY 1:
{condensed1}

---

POLICY 2:
{condensed2}

---

List the 10 most important differences in this format:
1. [Category]: Policy 1: [value] | Policy 2: [value]
2. [Category]: Policy 1: [value] | Policy 2: [value]
...

Focus on: premiums, deductibles, coverage limits, exclusions, and any red flags.
If a value is not found, write "Not specified"."""

    result = _call_ollama(prompt, SYSTEM_PROMPT)

    # Parse the differences
    differences = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Try to parse the line
            if "Policy 1:" in line and "Policy 2:" in line:
                try:
                    # Extract category and values
                    parts = line.split(":", 1)
                    if len(parts) >= 2:
                        rest = parts[1]
                        if "Policy 1:" in rest and "|" in rest:
                            category = parts[0].strip().lstrip("0123456789.-) ")
                            policy_parts = rest.split("|")
                            val1 = policy_parts[0].replace("Policy 1:", "").strip()
                            val2 = policy_parts[1].replace("Policy 2:", "").strip() if len(policy_parts) > 1 else "Not specified"
                            differences.append({
                                "category": category,
                                "policy1": val1,
                                "policy2": val2
                            })
                except:
                    continue

    return {
        "differences": differences[:10],
        "policy1_hash": hash1,
        "policy2_hash": hash2,
        "model": _config.model,
        "type": "quick_comparison"
    }
