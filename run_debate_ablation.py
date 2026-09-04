import json
from pathlib import Path

# Adjust imports from the project structure
from causalyst.src.agents import (
    build_live_evidence_for_ticker, 
    _call_claude, 
    run_debate
)

BASELINE_PROMPT = """You are a sole equity research analyst. 
You will be given multiple sources of evidence (fundamentals, sentiment, macro, institutional flow).
Produce a final judgment on the stock direction.
Respond ONLY with JSON: {"final_stance": "...", "confidence": 0.0, "plain_english_explanation": "..."}"""

HOMOGENEOUS_PROMPT = """You are a general equity research analyst.
You will be given multiple sources of evidence.
Produce a judgment on the stock direction.
Respond ONLY with JSON: {"stance": "...", "confidence": 0.0, "reasoning": "..."}"""

HOMOGENEOUS_SYNTHESIZER_PROMPT = """You are the Synthesizer. You will be given three independent generic analyst opinions.
Produce a final judgment. 
Respond ONLY with JSON: {"final_stance": "...", "confidence": 0.0, "plain_english_explanation": "...", "key_disagreement": "..."}"""

def format_evidence(evidence):
    lines = []
    for k, v in evidence.items():
        if isinstance(v, dict) and "evidence" in v:
            lines.append(f"[{k.upper()}]: " + "; ".join(v["evidence"]))
    return "\n".join(lines)

def run_baseline(evidence_text):
    return _call_claude(BASELINE_PROMPT, f"Evidence:\n{evidence_text}")

def run_homogeneous(evidence_text):
    opinions = {}
    for i in range(3):
        opinions[f"Analyst_{i+1}"] = _call_claude(HOMOGENEOUS_PROMPT, f"Evidence:\n{evidence_text}")
    
    synth_input = "Analyst Opinions:\n" + "\n".join(f"{k}: {v}" for k, v in opinions.items())
    final = _call_claude(HOMOGENEOUS_SYNTHESIZER_PROMPT, synth_input)
    return {"analysts": opinions, "synthesis": final}

def main():
    target_ticker = "RELIANCE"
    print(f"Fetching evidence for {target_ticker}...")
    evidence = build_live_evidence_for_ticker(target_ticker)
    
    evidence_text = format_evidence(evidence)
    print("\n--- EVIDENCE GATHERED ---")
    print(evidence_text)
    
    results = {}
    print("\n--- 1. Running Single-Agent Baseline ---")
    try:
        results["baseline"] = run_baseline(evidence_text)
        print("Done.")
    except Exception as e:
        print(f"Failed: {e}")
        
    print("\n--- 2. Running Homogeneous Debate (3 generic agents + 1 synthesizer) ---")
    try:
        results["homogeneous"] = run_homogeneous(evidence_text)
        print("Done.")
    except Exception as e:
        print(f"Failed: {e}")
        
    print("\n--- 3. Running Structured Debate (4 domain agents + 1 contrarian + 1 synthesizer) ---")
    try:
        # Reformat evidence dict to match what run_debate expects
        structured_evidence = {}
        for role in ["fundamentals", "sentiment", "macro", "institutional_flow"]:
            if role in evidence and isinstance(evidence[role], dict) and "evidence" in evidence[role]:
                structured_evidence[role] = evidence[role]["evidence"]
        
        results["structured"] = run_debate(structured_evidence)
        print("Done.")
    except Exception as e:
        print(f"Failed: {e}")
        
    out_file = Path("ablation_results.json")
    with out_file.open("w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {out_file.absolute()}")
    print("NOTE: Ensure ANTHROPIC_API_KEY is set in your environment to successfully run the LLM calls.")

if __name__ == "__main__":
    main()
