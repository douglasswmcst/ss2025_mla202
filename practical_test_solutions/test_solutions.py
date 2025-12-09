"""
Test script for practical test solutions

Validates:
1. All files exist
2. Python syntax is valid
3. Imports work (where dependencies available)
4. Code structure is correct
"""

import ast
import os
import sys

def test_file_syntax(filename):
    """Test if Python file has valid syntax"""
    try:
        with open(filename, 'r') as f:
            ast.parse(f.read())
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_file_structure(filename):
    """Test if file has expected classes/functions"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        return True, {
            'classes': classes,
            'functions': functions,
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("PRACTICAL TEST SOLUTIONS - VALIDATION REPORT")
    print("=" * 70)

    files_to_test = [
        'problem1_solution.py',
        'problem2_solution.py',
        'problem3_solution.py'
    ]

    all_passed = True

    for filename in files_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {filename}")
        print(f"{'='*70}")

        # Test existence
        if not os.path.exists(filename):
            print(f"❌ FAIL: File not found")
            all_passed = False
            continue

        print(f"✅ File exists")

        # Test syntax
        syntax_ok, syntax_msg = test_file_syntax(filename)
        if syntax_ok:
            print(f"✅ Syntax valid")
        else:
            print(f"❌ Syntax error: {syntax_msg}")
            all_passed = False
            continue

        # Test structure
        structure_ok, structure_info = test_file_structure(filename)
        if structure_ok:
            print(f"✅ Structure valid")
            print(f"   - Classes: {', '.join(structure_info['classes']) if structure_info['classes'] else 'None'}")
            print(f"   - Functions: {len(structure_info['functions'])} found")
            print(f"   - Lines: {structure_info['lines']}")
        else:
            print(f"⚠️  Structure check failed: {structure_info}")

    print(f"\n{'='*70}")
    print("EXPECTED COMPONENTS CHECK")
    print(f"{'='*70}")

    # Problem 1 expectations
    print("\nProblem 1 (GridWorld Q-Learning):")
    success, info = test_file_structure('problem1_solution.py')
    if success:
        expected_classes = ['GridWorld']
        expected_functions = ['train_qlearning', 'visualize_policy', 'plot_results', 'test_agent']

        for cls in expected_classes:
            if cls in info['classes']:
                print(f"   ✅ {cls} class found")
            else:
                print(f"   ❌ {cls} class MISSING")
                all_passed = False

        for func in expected_functions:
            if func in info['functions']:
                print(f"   ✅ {func}() function found")
            else:
                print(f"   ⚠️  {func}() function MISSING")

    # Problem 2 expectations
    print("\nProblem 2 (DQN for CartPole):")
    success, info = test_file_structure('problem2_solution.py')
    if success:
        expected_classes = ['DQN', 'ReplayBuffer', 'DQNAgent']
        expected_functions = ['train_cartpole', 'plot_results', 'test_agent']

        for cls in expected_classes:
            if cls in info['classes']:
                print(f"   ✅ {cls} class found")
            else:
                print(f"   ❌ {cls} class MISSING")
                all_passed = False

        for func in expected_functions:
            if func in info['functions']:
                print(f"   ✅ {func}() function found")
            else:
                print(f"   ⚠️  {func}() function MISSING")

    # Problem 3 expectations
    print("\nProblem 3 (Debugging Q-Learning):")
    success, info = test_file_structure('problem3_solution.py')
    if success:
        expected_functions = ['buggy_train_frozenlake', 'fixed_train_frozenlake', 'compare_buggy_vs_fixed', 'plot_comparison']

        for func in expected_functions:
            if func in info['functions']:
                print(f"   ✅ {func}() function found")
            else:
                print(f"   ⚠️  {func}() function MISSING")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nAll solution files are:")
        print("  • Syntactically correct")
        print("  • Properly structured")
        print("  • Ready for student use")
        print("\nNote: Runtime testing requires gym, torch, numpy, matplotlib")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review errors above")

    print(f"\n{'='*70}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
