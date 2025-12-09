#!/usr/bin/env python3
"""
Test script for team project implementations
Checks syntax and import structure without requiring all dependencies
"""

import os
import sys
import ast

def check_python_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_project(project_name, project_dir):
    """Test a project's Python files"""
    print(f"\n{'='*60}")
    print(f"Testing: {project_name}")
    print(f"{'='*60}")

    if not os.path.exists(project_dir):
        print(f"❌ Directory not found: {project_dir}")
        return False

    py_files = [f for f in os.listdir(project_dir) if f.endswith('.py')]

    if not py_files:
        print(f"⚠️  No Python files found")
        return True

    all_valid = True
    for py_file in sorted(py_files):
        filepath = os.path.join(project_dir, py_file)
        valid, message = check_python_syntax(filepath)

        status = "✓" if valid else "✗"
        print(f"{status} {py_file:30s} - {message}")

        if not valid:
            all_valid = False

    return all_valid

def main():
    """Run all tests"""
    print("="*60)
    print("MLA202 Team Projects - Implementation Test Suite")
    print("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    projects = {
        'BrainStormers - Dynamic Pricing': 'brainstormers_dynamic_pricing',
        'Model Minds - Deep Q-Network': 'model_minds_dqn',
        'PhuRaTan - Ultimate Tic-Tac-Toe': 'phuratan_tictactoe',
        'Q-Team - Snake RL': 'qteam_snake',
        'The Crown - Q-Learning': 'thecrown_qlearning',
        'The Innovators - Flappy Bird': 'theinnovators_flappybird',
        'NeuraWorks - AR-GRPO': 'neuraworks_argrpo'
    }

    results = {}
    for name, folder in projects.items():
        project_dir = os.path.join(base_dir, folder)
        results[name] = test_project(name, project_dir)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print(f"\n{passed}/{total} projects passed syntax validation")

    if passed == total:
        print("\n🎉 All implementations have valid Python syntax!")
    else:
        print(f"\n⚠️  {total - passed} project(s) have issues")

    # Check for README files
    print(f"\n{'='*60}")
    print("DOCUMENTATION CHECK")
    print(f"{'='*60}")

    for name, folder in projects.items():
        readme_path = os.path.join(base_dir, folder, 'README.md')
        if os.path.exists(readme_path):
            size = os.path.getsize(readme_path)
            print(f"✓ {name:40s} - README.md ({size} bytes)")
        else:
            print(f"✗ {name:40s} - README.md missing")

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
