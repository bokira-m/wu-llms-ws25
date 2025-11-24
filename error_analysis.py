"""
Error Analysis for Baseline Evaluation
Analyzes where and why the model fails
"""

import json
import os
from collections import defaultdict


def load_results(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_errors(results: dict) -> dict:
    """Perform comprehensive error analysis."""

    data = results['results']

    analysis = {
        "total_questions": len(data),
        "error_categories": defaultdict(list),
        "worst_performers": [],
        "best_performers": [],
        "category_analysis": defaultdict(lambda: {"total": 0, "sum_f1": 0, "errors": []}),
        "difficulty_analysis": defaultdict(lambda: {"total": 0, "sum_f1": 0}),
        "error_patterns": defaultdict(int),
        "sample_errors": []
    }

    # Sort by F1 score
    sorted_by_f1 = sorted(data, key=lambda x: x['metrics']['f1_score'])

    # Worst 10 - with FULL answers
    analysis['worst_performers'] = [
        {
            "id": r['id'],
            "question": r['question'],
            "f1_score": r['metrics']['f1_score'],
            "cosine_sim": r['metrics']['cosine_similarity'],
            "category": r['category'],
            "difficulty": r['difficulty'],
            "expected_answer": r['expected_answer'],  # FULL answer
            "generated_answer": r['generated_answer']  # FULL answer
        }
        for r in sorted_by_f1[:10]
    ]

    # Best 10 - with FULL answers
    analysis['best_performers'] = [
        {
            "id": r['id'],
            "question": r['question'],
            "f1_score": r['metrics']['f1_score'],
            "cosine_sim": r['metrics']['cosine_similarity'],
            "category": r['category'],
            "difficulty": r['difficulty'],
            "expected_answer": r['expected_answer'],  # FULL answer
            "generated_answer": r['generated_answer']  # FULL answer
        }
        for r in sorted_by_f1[-10:][::-1]
    ]

    # ALL results with full answers for manual analysis
    analysis['all_results'] = [
        {
            "id": r['id'],
            "question": r['question'],
            "category": r['category'],
            "difficulty": r['difficulty'],
            "expected_answer": r['expected_answer'],
            "generated_answer": r['generated_answer'],
            "metrics": r['metrics']
        }
        for r in sorted_by_f1
    ]

    # Analyze each result
    for r in data:
        f1 = r['metrics']['f1_score']
        cat = r['category']
        diff = r['difficulty']

        # Category analysis
        analysis['category_analysis'][cat]['total'] += 1
        analysis['category_analysis'][cat]['sum_f1'] += f1

        # Difficulty analysis
        analysis['difficulty_analysis'][diff]['total'] += 1
        analysis['difficulty_analysis'][diff]['sum_f1'] += f1

        # Error pattern detection
        generated = r['generated_answer'].lower()
        expected = r['expected_answer'].lower()

        # Pattern 1: Model says "I don't know" or similar
        if any(phrase in generated for phrase in ['i don\'t', 'ich weiß nicht', 'i\'m not', 'i cannot', 'nicht bekannt']):
            analysis['error_patterns']['no_knowledge'] += 1

        # Pattern 2: Model gives generic HR answer (not Rexx-specific)
        if 'rexx' not in generated and 'rexx' in expected:
            analysis['error_patterns']['not_rexx_specific'] += 1

        # Pattern 3: Answer is in wrong language
        if 'the ' in generated[:50] or ' is ' in generated[:50]:  # English in German context
            analysis['error_patterns']['wrong_language'] += 1

        # Pattern 4: Answer is too short
        if len(generated) < 50:
            analysis['error_patterns']['too_short'] += 1

        # Pattern 5: Answer is too long/verbose
        if len(generated) > 500:
            analysis['error_patterns']['too_verbose'] += 1

        # Pattern 6: Hallucination - mentions things not in expected
        if f1 < 0.1 and len(generated) > 100:
            analysis['error_patterns']['possible_hallucination'] += 1

        # Categorize error severity
        if f1 == 0:
            analysis['error_categories']['complete_miss'].append(r['id'])
        elif f1 < 0.1:
            analysis['error_categories']['severe_error'].append(r['id'])
        elif f1 < 0.2:
            analysis['error_categories']['significant_error'].append(r['id'])
        elif f1 < 0.3:
            analysis['error_categories']['moderate_error'].append(r['id'])
        else:
            analysis['error_categories']['acceptable'].append(r['id'])

    # Calculate averages
    for cat in analysis['category_analysis']:
        total = analysis['category_analysis'][cat]['total']
        analysis['category_analysis'][cat]['avg_f1'] = round(
            analysis['category_analysis'][cat]['sum_f1'] / total, 4
        ) if total > 0 else 0

    for diff in analysis['difficulty_analysis']:
        total = analysis['difficulty_analysis'][diff]['total']
        analysis['difficulty_analysis'][diff]['avg_f1'] = round(
            analysis['difficulty_analysis'][diff]['sum_f1'] / total, 4
        ) if total > 0 else 0

    # Convert defaultdicts to regular dicts for JSON
    analysis['error_categories'] = dict(analysis['error_categories'])
    analysis['category_analysis'] = dict(analysis['category_analysis'])
    analysis['difficulty_analysis'] = dict(analysis['difficulty_analysis'])
    analysis['error_patterns'] = dict(analysis['error_patterns'])

    return analysis


def print_error_report(analysis: dict):
    """Print formatted error analysis report."""

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS REPORT - BASELINE (llama2)")
    print("=" * 70)

    print(f"\nTotal Questions Analyzed: {analysis['total_questions']}")

    # Error Distribution
    print("\n" + "-" * 40)
    print("ERROR SEVERITY DISTRIBUTION")
    print("-" * 40)
    for severity, ids in analysis['error_categories'].items():
        pct = len(ids) / analysis['total_questions'] * 100
        bar = "█" * int(pct / 2)
        print(f"{severity:20s}: {len(ids):3d} ({pct:5.1f}%) {bar}")

    # Error Patterns
    print("\n" + "-" * 40)
    print("ERROR PATTERNS DETECTED")
    print("-" * 40)
    for pattern, count in sorted(analysis['error_patterns'].items(), key=lambda x: -x[1]):
        pct = count / analysis['total_questions'] * 100
        print(f"{pattern:25s}: {count:3d} ({pct:5.1f}%)")

    # Category Performance
    print("\n" + "-" * 40)
    print("PERFORMANCE BY CATEGORY")
    print("-" * 40)
    sorted_cats = sorted(
        analysis['category_analysis'].items(),
        key=lambda x: x[1]['avg_f1']
    )
    for cat, stats in sorted_cats:
        avg = stats['avg_f1']
        bar = "█" * int(avg * 20)
        print(f"{cat:20s}: F1={avg:.3f} (n={stats['total']:2d}) {bar}")

    # Difficulty Performance
    print("\n" + "-" * 40)
    print("PERFORMANCE BY DIFFICULTY")
    print("-" * 40)
    for diff, stats in analysis['difficulty_analysis'].items():
        avg = stats['avg_f1']
        bar = "█" * int(avg * 20)
        print(f"{diff:10s}: F1={avg:.3f} (n={stats['total']:2d}) {bar}")

    # Worst Performers
    print("\n" + "-" * 40)
    print("WORST 10 QUESTIONS (Lowest F1)")
    print("-" * 40)
    for i, item in enumerate(analysis['worst_performers'], 1):
        print(f"\n{i}. [{item['category']}] F1={item['f1_score']:.3f}")
        print(f"   Q: {item['question']}")
        print(f"   Expected: {item['expected_answer'][:100]}...")
        print(f"   Got: {item['generated_answer'][:100]}...")

    # Best Performers
    print("\n" + "-" * 40)
    print("BEST 10 QUESTIONS (Highest F1)")
    print("-" * 40)
    for i, item in enumerate(analysis['best_performers'], 1):
        print(f"{i}. [{item['category']}] F1={item['f1_score']:.3f} - {item['question'][:50]}...")

    # Summary & Recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    total_errors = sum(len(ids) for sev, ids in analysis['error_categories'].items()
                       if sev != 'acceptable')
    error_rate = total_errors / analysis['total_questions'] * 100

    print(f"\n• Error Rate: {error_rate:.1f}% of answers have F1 < 0.3")
    print(f"• Most Common Pattern: {max(analysis['error_patterns'], key=analysis['error_patterns'].get)}")

    weakest_cat = min(analysis['category_analysis'].items(), key=lambda x: x[1]['avg_f1'])
    print(f"• Weakest Category: {weakest_cat[0]} (F1={weakest_cat[1]['avg_f1']:.3f})")

    print("\nKey Issues Identified:")
    if analysis['error_patterns'].get('not_rexx_specific', 0) > 10:
        print("  1. Model lacks Rexx-specific knowledge → Fine-tuning needed")
    if analysis['error_patterns'].get('wrong_language', 0) > 5:
        print("  2. Model sometimes responds in English → German prompting needed")
    if analysis['error_patterns'].get('possible_hallucination', 0) > 10:
        print("  3. Model hallucinates information → RAG context would help")
    if analysis['error_patterns'].get('too_verbose', 0) > 10:
        print("  4. Answers too verbose → Need more concise training data")

    print("\nRecommendations for Improvement:")
    print("  • Fine-tune on Rexx-specific Q&A pairs")
    print("  • Use RAG to provide document context")
    print("  • Add German-specific prompting")
    print("  • Train for concise, focused answers")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "baseline_results.json")

    print("Loading baseline results...")
    results = load_results(results_path)

    print("Performing error analysis...")
    analysis = analyze_errors(results)

    # Save analysis
    output_path = os.path.join(script_dir, "error_analysis_baseline.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Analysis saved to: {output_path}")

    # Print report
    print_error_report(analysis)


if __name__ == "__main__":
    main()
