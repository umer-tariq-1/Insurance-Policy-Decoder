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
    max_chunk_tokens: int = 2000  # Safe limit for 3B models
    temperature: float = 0.3  # Lower = more focused/factual
    num_ctx: int = 4096  # Context window


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
        }
    }

    response = requests.post(
        f"{_config.ollama_url}/api/generate",
        json=payload,
        timeout=120  # 2 min timeout for generation
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
INSURANCE_SYSTEM_PROMPT = """You are an expert insurance policy analyst. Your task is to extract and summarize the most important information from insurance policy documents.

IMPORTANT GUIDELINES:
- Be accurate and factual - only state what is explicitly in the document
- Use specific numbers, dates, and amounts when available
- Highlight information that directly affects the policyholder
- Use clear, simple language that anyone can understand
- Structure your response clearly with sections
- Do NOT make assumptions or add information not in the document"""


SECTION_EXTRACT_PROMPT = """Analyze this section of an insurance policy and extract key information.
Focus on:
- Coverage details (what's covered, limits, amounts)
- Exclusions (what's NOT covered)
- Costs (premiums, deductibles, co-pays)
- Important conditions or requirements
- Deadlines and waiting periods

TEXT TO ANALYZE:
{text}

Provide a bullet-point summary of the key information found. If the section doesn't contain relevant insurance details, respond with "No key insurance information in this section."
"""


FINAL_SYNTHESIS_PROMPT = """Based on the following extracted information from an insurance policy, create a comprehensive summary.

EXTRACTED INFORMATION:
{extracted_info}

Create a well-organized summary with these sections (only include sections that have relevant information):

**POLICY OVERVIEW**
Brief description of what this policy covers

**COVERAGE & BENEFITS**
- What's covered
- Coverage limits and amounts

**COSTS & PAYMENTS**
- Premium information
- Deductibles and co-pays
- Out-of-pocket maximums

**EXCLUSIONS & LIMITATIONS**
- What's NOT covered
- Important restrictions

**KEY REQUIREMENTS**
- Policyholder responsibilities
- Claim procedures
- Important deadlines

**IMPORTANT WARNINGS**
- Conditions that could void coverage
- Critical terms to be aware of

Be specific with numbers and dates. Keep each point concise but informative."""


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
    if use_extractive_prefilter:
        num_sentences = 50 if detailed else 30
        key_sentences = extract_key_information(text, num_sentences=num_sentences)
        working_text = "\n\n".join(key_sentences)
    else:
        working_text = text

    # Step 2: Chunk the text for processing
    chunks = _chunk_text_for_llm(working_text, _config.max_chunk_tokens)

    # Step 3: Extract key information from each chunk
    extracted_parts = []

    for i, chunk in enumerate(chunks):
        if _estimate_tokens(chunk) < 50:  # Skip very short chunks
            continue

        prompt = SECTION_EXTRACT_PROMPT.format(text=chunk)

        try:
            extraction = _call_ollama(prompt, INSURANCE_SYSTEM_PROMPT)

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

    # Step 4: Synthesize final summary
    combined_extractions = "\n\n---\n\n".join(extracted_parts)

    # If combined extractions are too long, we need to summarize in stages
    if _estimate_tokens(combined_extractions) > _config.max_chunk_tokens:
        # Truncate to fit (keep first and last parts which often have key info)
        max_chars = _config.max_chunk_tokens * 4
        half = max_chars // 2
        combined_extractions = combined_extractions[:half] + "\n\n[...]\n\n" + combined_extractions[-half:]

    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(extracted_info=combined_extractions)
    final_summary = _call_ollama(synthesis_prompt, INSURANCE_SYSTEM_PROMPT)

    return {
        "summary": final_summary.strip(),
        "model": _config.model,
        "sections_processed": len(extracted_parts)
    }


def generate_quick_summary(text: str, max_points: int = 10) -> dict:
    """
    Generate a quick bullet-point summary (faster, less comprehensive).
    Good for initial overview or when speed is important.
    """
    available, message = check_ollama_available()
    if not available:
        raise RuntimeError(f"Ollama not available: {message}")

    # Extract key sentences first
    key_sentences = extract_key_information(text, num_sentences=20)
    key_text = "\n".join(f"- {s}" for s in key_sentences)

    prompt = f"""From these key sentences from an insurance policy, identify the {max_points} MOST important points a policyholder must know.

KEY SENTENCES:
{key_text}

Provide exactly {max_points} bullet points, each starting with a category in brackets like [COVERAGE], [COST], [EXCLUSION], [DEADLINE], [REQUIREMENT], or [WARNING].

Format:
[CATEGORY] Concise point about that topic"""

    result = _call_ollama(prompt, INSURANCE_SYSTEM_PROMPT)

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
