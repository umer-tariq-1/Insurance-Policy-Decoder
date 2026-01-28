# insurance_ai/services/nlp/ollama_summarizer.py
"""
Production-ready insurance policy summarizer using local LLM via Ollama.

Recommended models for 4GB VRAM (Quadro T1200):
- llama3.2:3b (best balance of quality and speed)
- phi3:mini (Microsoft's efficient model)
- qwen2.5:3b (good for structured output)

Install Ollama: https://ollama.ai
Pull model: ollama pull llama3.2:3b
"""

import os
import requests
import json
from typing import Optional
from dataclasses import dataclass

from .extractive_summarizer import extract_key_information


@dataclass
class SummaryConfig:
    """Configuration for the summarizer."""
    model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"
    max_chunk_tokens: int = 2500  # Token limit per chunk
    temperature: float = 0.3  # Lower = more focused/factual
    num_ctx: int = 8192  # Context window - increased for longer output
    num_predict: int = 4096  # Max tokens to generate - allows longer responses


# Global config instance
_config = SummaryConfig()


def configure_summarizer(
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    temperature: Optional[float] = None
):
    """Configure the summarizer settings."""
    global _config
    if model:
        _config.model = model
    if ollama_url:
        _config.ollama_url = ollama_url
    if temperature is not None:
        _config.temperature = temperature


def check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get(f"{_config.ollama_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, "Ollama server not responding"

        models = response.json().get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        # Check if our model (or base name) is available
        target_base = _config.model.split(":")[0]
        if not any(target_base in name for name in model_names):
            available = ", ".join(model_names[:5]) if model_names else "none"
            return False, f"Model '{_config.model}' not found. Available: {available}. Run: ollama pull {_config.model}"

        return True, "OK"
    except requests.exceptions.ConnectionError:
        return False, "Ollama not running. Start with: ollama serve"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"


def _call_ollama(prompt: str, system_prompt: str = "", max_tokens: Optional[int] = None) -> str:
    """Make a request to Ollama API."""
    payload = {
        "model": _config.model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": _config.temperature,
            "num_ctx": _config.num_ctx,
            "num_predict": max_tokens or _config.num_predict,
        }
    }

    response = requests.post(
        f"{_config.ollama_url}/api/generate",
        json=payload,
        timeout=300  # 5 min timeout for longer generation
    )
    response.raise_for_status()

    return response.json().get("response", "")


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (avg 4 chars per token for English)."""
    return len(text) // 4


def _chunk_text_for_llm(text: str, max_tokens: int = 2000) -> list[str]:
    """Split text into chunks that fit within token limit."""
    max_chars = max_tokens * 4  # Convert tokens to approximate chars

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single paragraph is too long, split by sentences
            if len(para) > max_chars:
                sentences = para.replace(". ", ".\n").split("\n")
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 < max_chars:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "
            else:
                current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Insurance-specific system prompt
INSURANCE_SYSTEM_PROMPT = """You are an expert insurance policy analyst with years of experience helping consumers understand complex insurance documents. Your task is to create DETAILED, THOROUGH summaries that leave no important information behind.

CRITICAL GUIDELINES:
- Be COMPREHENSIVE - include ALL relevant details, not just highlights
- Be accurate and factual - only state what is explicitly in the document
- Use specific numbers, dates, percentages, and amounts whenever available
- Explain insurance jargon in plain language
- Highlight information that directly affects the policyholder's rights, obligations, and finances
- Structure your response with clear sections and sub-sections
- Include context and explanations, not just bullet points
- Do NOT make assumptions or add information not in the document
- When in doubt, include more detail rather than less"""


SECTION_EXTRACT_PROMPT = """Analyze this section of an insurance policy and extract ALL important information in detail.

For each piece of information found, provide:
- The specific detail (numbers, dates, percentages, conditions)
- Any related conditions or requirements
- Any exceptions or special cases mentioned

Focus on extracting:
1. COVERAGE DETAILS: What's covered, coverage limits, sub-limits, per-incident limits, annual maximums
2. EXCLUSIONS: What's specifically NOT covered, circumstances that void coverage
3. FINANCIAL TERMS: Premiums, deductibles, co-pays, co-insurance percentages, out-of-pocket maximums
4. CONDITIONS: Pre-authorization requirements, network restrictions, waiting periods
5. PROCEDURES: How to file claims, appeal processes, required documentation
6. DEFINITIONS: Important terms defined in this section
7. TIME LIMITS: Deadlines, grace periods, notification requirements

TEXT TO ANALYZE:
{text}

Provide a DETAILED extraction with all specifics. Include exact amounts, percentages, and conditions. If the section doesn't contain relevant insurance details, respond with "No key insurance information in this section."
"""


FINAL_SYNTHESIS_PROMPT = """Based on the following extracted information from an insurance policy, create a COMPREHENSIVE and DETAILED summary that a policyholder can use as a complete reference guide.

EXTRACTED INFORMATION:
{extracted_info}

Create a thorough, well-organized summary with ALL of the following sections. For each section, provide DETAILED explanations with specific numbers, dates, and conditions. Do not just list bullet points - explain what each item means for the policyholder.

---

## POLICY OVERVIEW
Provide a detailed description of:
- Type of insurance policy and its primary purpose
- Who is covered (insured parties, dependents, beneficiaries)
- Policy period and effective dates
- General scope of coverage

## COVERAGE DETAILS & BENEFITS

### What's Covered
List and explain each type of coverage included, with specific details about:
- Coverage categories and what falls under each
- Specific services, items, or situations covered
- Any special benefits or riders included

### Coverage Limits & Amounts
Detail all financial limits:
- Per-incident/per-occurrence limits
- Annual or lifetime maximums
- Sub-limits for specific categories
- Any caps on specific services

### Network & Provider Information
If applicable:
- In-network vs out-of-network coverage differences
- How to find approved providers
- Referral requirements

## COSTS & FINANCIAL OBLIGATIONS

### Premium Information
- Premium amount and payment frequency
- Payment methods accepted
- Consequences of missed payments

### Deductibles
- Amount and type (per-incident vs annual)
- What counts toward the deductible
- Family vs individual deductibles if applicable

### Co-payments & Co-insurance
- Co-pay amounts for different services
- Co-insurance percentages
- When each applies

### Out-of-Pocket Maximum
- Maximum amount you'll pay annually
- What counts toward this maximum
- What happens after reaching the maximum

## EXCLUSIONS & LIMITATIONS

### What's NOT Covered
List all exclusions with explanations:
- Specific services or situations excluded
- Pre-existing condition limitations
- Experimental or investigational exclusions

### Coverage Restrictions
- Geographic limitations
- Time-based restrictions
- Quantity or frequency limits

### Conditions That Could Void Coverage
- Actions that could result in claim denial
- Material misrepresentation consequences
- Policy cancellation triggers

## CLAIM PROCEDURES & REQUIREMENTS

### How to File a Claim
Step-by-step process:
- Required documentation
- Where and how to submit
- Timeline for submission

### Claim Processing
- Expected processing time
- How you'll be notified of decisions
- Payment methods

### Appeals Process
- How to appeal a denied claim
- Deadlines for appeals
- Required information for appeals

## POLICYHOLDER RESPONSIBILITIES

### Required Actions
- Notification requirements
- Premium payment obligations
- Cooperation requirements

### Pre-authorization Requirements
- Services requiring prior approval
- How to obtain authorization
- Consequences of not getting authorization

## IMPORTANT DATES & DEADLINES

List all time-sensitive information:
- Policy effective and expiration dates
- Open enrollment periods
- Claim filing deadlines
- Cancellation notice requirements
- Waiting periods for specific coverages

## RENEWAL & CANCELLATION

### Renewal Terms
- Automatic vs manual renewal
- Rate change notifications
- How to make changes at renewal

### Cancellation Policy
- How to cancel the policy
- Refund policy for cancellations
- When the insurer can cancel

## CRITICAL WARNINGS & RED FLAGS

Highlight the most important things the policyholder MUST know:
- Common reasons for claim denial
- Easy-to-miss requirements
- Situations that could leave you unprotected
- Time-sensitive requirements that if missed could cost you

---

Be thorough and specific. Use actual numbers, dates, and percentages from the document. Explain technical terms in plain language. This summary should serve as a complete reference that eliminates the need to read the full policy for most questions."""


def _consolidate_extractions(extractions: list[str], max_tokens: int) -> str:
    """
    If extractions are too long, consolidate them in stages.
    This preserves more information than simple truncation.
    """
    combined = "\n\n---\n\n".join(extractions)

    if _estimate_tokens(combined) <= max_tokens:
        return combined

    # Split extractions into groups and summarize each group
    print(f"Consolidating {len(extractions)} extractions (too long for single pass)...")

    group_size = max(2, len(extractions) // 3)
    groups = [extractions[i:i + group_size] for i in range(0, len(extractions), group_size)]

    consolidated_parts = []
    consolidation_prompt = """Consolidate and merge these extracted points into a comprehensive but more concise format.
Keep ALL important details including specific numbers, dates, conditions, and requirements.
Remove only redundant or duplicate information.

EXTRACTIONS TO CONSOLIDATE:
{text}

Provide a consolidated summary that preserves all key information:"""

    for group in groups:
        group_text = "\n\n".join(group)
        if _estimate_tokens(group_text) > 100:
            try:
                consolidated = _call_ollama(
                    consolidation_prompt.format(text=group_text),
                    INSURANCE_SYSTEM_PROMPT,
                    max_tokens=1500
                )
                consolidated_parts.append(consolidated)
            except Exception as e:
                print(f"Warning: Consolidation failed, using original: {e}")
                consolidated_parts.append(group_text[:max_tokens * 2])  # Rough truncation as fallback

    return "\n\n---\n\n".join(consolidated_parts)


def generate_summary_with_ollama(
    text: str,
    use_extractive_prefilter: bool = True,
    detailed: bool = False
) -> dict:
    """
    Generate a comprehensive insurance policy summary using local LLM.

    Args:
        text: The normalized policy text
        use_extractive_prefilter: If True, first extract key sentences to reduce noise
        detailed: If True, process more content for a more detailed summary

    Returns:
        dict with 'summary', 'model', and 'sections_processed'
    """
    # Check Ollama availability first
    available, message = check_ollama_available()
    if not available:
        raise RuntimeError(f"Ollama not available: {message}")

    # Step 1: Optionally prefilter with extractive summarization
    # Use MORE sentences for detailed summaries
    if use_extractive_prefilter:
        num_sentences = 150 if detailed else 80
        key_sentences = extract_key_information(text, num_sentences=num_sentences)
        working_text = "\n\n".join(key_sentences)
        print(f"Extracted {len(key_sentences)} key sentences for processing")
    else:
        working_text = text

    # Step 2: Chunk the text for processing
    chunks = _chunk_text_for_llm(working_text, _config.max_chunk_tokens)
    print(f"Processing {len(chunks)} text chunks...")

    # Step 3: Extract key information from each chunk
    extracted_parts = []

    for i, chunk in enumerate(chunks):
        if _estimate_tokens(chunk) < 50:  # Skip very short chunks
            continue

        prompt = SECTION_EXTRACT_PROMPT.format(text=chunk)
        print(f"  Processing chunk {i+1}/{len(chunks)}...")

        try:
            extraction = _call_ollama(prompt, INSURANCE_SYSTEM_PROMPT, max_tokens=1500)

            # Skip non-informative extractions
            if "no key insurance information" not in extraction.lower():
                extracted_parts.append(extraction)
        except Exception as e:
            print(f"Warning: Failed to process chunk {i+1}: {e}")
            continue

    if not extracted_parts:
        return {
            "summary": "Unable to extract meaningful information from the document.",
            "model": _config.model,
            "sections_processed": 0
        }

    print(f"Extracted information from {len(extracted_parts)} chunks")

    # Step 4: Consolidate extractions if too long (instead of truncating)
    max_synthesis_tokens = _config.num_ctx // 2  # Leave room for prompt and output
    combined_extractions = _consolidate_extractions(extracted_parts, max_synthesis_tokens)

    # Step 5: Generate final comprehensive summary
    print("Generating final comprehensive summary...")
    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(extracted_info=combined_extractions)

    # Allow longer output for the final summary
    final_summary = _call_ollama(
        synthesis_prompt,
        INSURANCE_SYSTEM_PROMPT,
        max_tokens=_config.num_predict  # Use full allowed output length
    )

    return {
        "summary": final_summary.strip(),
        "model": _config.model,
        "sections_processed": len(extracted_parts),
        "sentences_analyzed": len(key_sentences) if use_extractive_prefilter else "N/A"
    }


def generate_quick_summary(text: str, max_points: int = 20) -> dict:
    """
    Generate a quick bullet-point summary (faster, less comprehensive).
    Good for initial overview or when speed is important.
    """
    available, message = check_ollama_available()
    if not available:
        raise RuntimeError(f"Ollama not available: {message}")

    # Extract key sentences first
    key_sentences = extract_key_information(text, num_sentences=40)
    key_text = "\n".join(f"- {s}" for s in key_sentences)

    prompt = f"""From these key sentences from an insurance policy, identify the {max_points} MOST important points a policyholder must know.

KEY SENTENCES:
{key_text}

Provide exactly {max_points} detailed bullet points organized by category. Each point should include specific numbers, dates, or conditions when available.

Categories to use: [COVERAGE], [BENEFIT], [COST], [DEDUCTIBLE], [EXCLUSION], [LIMIT], [DEADLINE], [REQUIREMENT], [CLAIM], [WARNING]

Format each point as:
[CATEGORY] Detailed explanation including specific amounts, conditions, or requirements"""

    result = _call_ollama(prompt, INSURANCE_SYSTEM_PROMPT, max_tokens=2000)

    # Parse into structured format
    points = []
    for line in result.strip().split("\n"):
        line = line.strip()
        if line.startswith("[") and "]" in line:
            bracket_end = line.index("]") + 1
            category = line[1:bracket_end-1].upper()
            content = line[bracket_end:].strip()
            if content:
                points.append({"category": category, "content": content})
        elif line.startswith("-") or line.startswith("*"):
            points.append({"category": "INFO", "content": line[1:].strip()})

    return {
        "points": points,
        "model": _config.model,
        "type": "quick_summary"
    }


# Convenience function for direct use
def summarize_insurance_policy(
    text: str,
    mode: str = "comprehensive"
) -> dict:
    """
    Main entry point for summarizing insurance policies.

    Args:
        text: Normalized policy text
        mode: "comprehensive" (detailed), "standard" (balanced), or "quick" (fast)

    Returns:
        Summary dict with appropriate fields based on mode
    """
    if mode == "quick":
        return generate_quick_summary(text)
    elif mode == "comprehensive":
        return generate_summary_with_ollama(text, detailed=True)
    else:  # standard
        return generate_summary_with_ollama(text, detailed=False)
