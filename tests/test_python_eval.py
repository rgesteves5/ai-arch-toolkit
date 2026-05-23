"""Tests for the safe Python REPL with output capture."""

from __future__ import annotations

from ai_arch_toolkit.toolkit.tools._python import python_repl

# ---------------------------------------------------------------------------
# Output capture — REPL-style: print + last expression
# ---------------------------------------------------------------------------


class TestPythonReplOutputCapture:
    def test_single_expression(self):
        assert python_repl("2 + 2") == "4"

    def test_sorted_list(self):
        assert python_repl("sorted(['c','a','b'])") == "['a', 'b', 'c']"

    def test_assignment_then_expression(self):
        assert python_repl("x = 2 + 2\nx") == "4"

    def test_assignment_then_sum(self):
        assert python_repl("x = [1,2,3]\nsum(x)") == "6"

    def test_print_captures_output(self):
        assert python_repl("x = 2 + 2\nprint(x)") == "4"

    def test_print_in_loop(self):
        assert python_repl("for x in [1,2,3]: print(x)") == "1\n2\n3"

    def test_prints_plus_last_expr(self):
        assert python_repl('print("a")\nprint("b")\n2+2') == "a\nb\n\n4"

    def test_assignment_only_returns_none(self):
        assert python_repl("x = 5") == "None"

    def test_print_with_sep(self):
        assert python_repl('print(1, 2, 3, sep=", ")') == "1, 2, 3"

    def test_print_with_end(self):
        assert python_repl('print("hello", end="!")\nprint(" world")') == "hello! world"

    def test_multiple_prints(self):
        assert python_repl('print("line1")\nprint("line2")') == "line1\nline2"


# ---------------------------------------------------------------------------
# Arithmetic, bitwise, boolean operators
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_addition(self):
        assert python_repl("3 + 4") == "7"

    def test_subtraction(self):
        assert python_repl("10 - 3") == "7"

    def test_multiplication(self):
        assert python_repl("3 * 4") == "12"

    def test_true_division(self):
        assert python_repl("10 / 4") == "2.5"

    def test_floor_division(self):
        assert python_repl("10 // 4") == "2"

    def test_modulo(self):
        assert python_repl("10 % 3") == "1"

    def test_power(self):
        assert python_repl("2 ** 10") == "1024"

    def test_unary_minus(self):
        assert python_repl("-(2 + 3)") == "-5"

    def test_unary_not(self):
        assert python_repl("not True") == "False"


class TestBitwise:
    def test_and(self):
        assert python_repl("0b1100 & 0b1010") == "8"

    def test_or(self):
        assert python_repl("0b1100 | 0b1010") == "14"

    def test_xor(self):
        assert python_repl("0b1100 ^ 0b1010") == "6"

    def test_left_shift(self):
        assert python_repl("1 << 4") == "16"

    def test_right_shift(self):
        assert python_repl("16 >> 2") == "4"


class TestBoolean:
    def test_and_short_circuit(self):
        assert python_repl("True and False") == "False"

    def test_or_short_circuit(self):
        assert python_repl("False or 42") == "42"

    def test_chained(self):
        assert python_repl("True and 1 < 2 and 'a' in 'abc'") == "True"


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_equal(self):
        assert python_repl("1 == 1") == "True"

    def test_not_equal(self):
        assert python_repl("1 != 2") == "True"

    def test_less_than(self):
        assert python_repl("1 < 2") == "True"

    def test_less_equal(self):
        assert python_repl("2 <= 2") == "True"

    def test_greater_than(self):
        assert python_repl("3 > 2") == "True"

    def test_greater_equal(self):
        assert python_repl("3 >= 3") == "True"

    def test_in(self):
        assert python_repl("2 in [1, 2, 3]") == "True"

    def test_not_in(self):
        assert python_repl("4 not in [1, 2, 3]") == "True"

    def test_is(self):
        assert python_repl("None is None") == "True"

    def test_is_not(self):
        assert python_repl("1 is not None") == "True"

    def test_chained_comparison(self):
        assert python_repl("1 < 2 < 3") == "True"


# ---------------------------------------------------------------------------
# Collection literals + indexing/slicing
# ---------------------------------------------------------------------------


class TestCollections:
    def test_list_literal(self):
        assert python_repl("[1, 2, 3]") == "[1, 2, 3]"

    def test_tuple_literal(self):
        assert python_repl("(1, 2, 3)") == "(1, 2, 3)"

    def test_dict_literal(self):
        assert python_repl("{'a': 1, 'b': 2}") == "{'a': 1, 'b': 2}"

    def test_set_literal(self):
        # Sets stringify with deterministic order for small int sets in CPython 3.13+
        assert python_repl("sorted(list({3, 1, 2}))") == "[1, 2, 3]"

    def test_nested_list(self):
        assert python_repl("[[1, 2], [3, 4]]") == "[[1, 2], [3, 4]]"


class TestIndexingAndSlicing:
    def test_positive_index(self):
        assert python_repl("[10, 20, 30][1]") == "20"

    def test_negative_index(self):
        assert python_repl("[10, 20, 30][-1]") == "30"

    def test_slice_start_stop(self):
        assert python_repl("[1, 2, 3, 4, 5][1:3]") == "[2, 3]"

    def test_slice_start_only(self):
        assert python_repl("[1, 2, 3, 4, 5][2:]") == "[3, 4, 5]"

    def test_slice_stop_only(self):
        assert python_repl("[1, 2, 3, 4, 5][:3]") == "[1, 2, 3]"

    def test_slice_with_step(self):
        assert python_repl("[1, 2, 3, 4, 5][::2]") == "[1, 3, 5]"

    def test_dict_subscript(self):
        assert python_repl("{'a': 1, 'b': 2}['b']") == "2"

    def test_string_index(self):
        # python_repl returns str(last_value), so strings come back un-quoted
        assert python_repl("'hello'[1]") == "e"

    def test_string_slice(self):
        assert python_repl("'hello'[1:4]") == "ell"


# ---------------------------------------------------------------------------
# Comprehensions and ternary
# ---------------------------------------------------------------------------


class TestComprehensions:
    def test_list_comp_basic(self):
        assert python_repl("[x * 2 for x in [1, 2, 3]]") == "[2, 4, 6]"

    def test_list_comp_filter(self):
        assert python_repl("[x for x in range(10) if x % 2 == 0]") == "[0, 2, 4, 6, 8]"

    def test_dict_comp(self):
        assert python_repl("{x: x*x for x in [1, 2, 3]}") == "{1: 1, 2: 4, 3: 9}"

    def test_dict_comp_filter(self):
        result = python_repl("{x: x*x for x in range(5) if x % 2 == 0}")
        assert result == "{0: 0, 2: 4, 4: 16}"

    def test_set_comp(self):
        assert python_repl("sorted({x % 3 for x in range(10)})") == "[0, 1, 2]"

    def test_nested_comprehension(self):
        assert python_repl("[x for row in [[1, 2], [3, 4]] for x in row]") == "[1, 2, 3, 4]"


class TestTernary:
    def test_true_branch(self):
        assert python_repl("'big' if 5 > 3 else 'small'") == "big"

    def test_false_branch(self):
        assert python_repl("'big' if 2 > 3 else 'small'") == "small"


# ---------------------------------------------------------------------------
# Statements: assignment, loops, control flow
# ---------------------------------------------------------------------------


class TestAssignment:
    def test_simple_assign(self):
        assert python_repl("x = 5\nx") == "5"

    def test_chained_assign(self):
        assert python_repl("x = y = 7\n(x, y)") == "(7, 7)"

    def test_tuple_unpacking(self):
        assert python_repl("a, b = 1, 2\n(a, b)") == "(1, 2)"

    def test_swap(self):
        assert python_repl("a = 1\nb = 2\na, b = b, a\n(a, b)") == "(2, 1)"

    def test_list_unpacking(self):
        assert python_repl("[a, b, c] = [10, 20, 30]\n(a, b, c)") == "(10, 20, 30)"

    def test_subscript_assignment(self):
        assert python_repl("d = {}\nd['k'] = 1\nd") == "{'k': 1}"

    def test_list_item_assignment(self):
        assert python_repl("xs = [1, 2, 3]\nxs[0] = 99\nxs") == "[99, 2, 3]"


class TestAugmentedAssignment:
    def test_iadd(self):
        assert python_repl("x = 5\nx += 3\nx") == "8"

    def test_isub(self):
        assert python_repl("x = 5\nx -= 3\nx") == "2"

    def test_imul(self):
        assert python_repl("x = 5\nx *= 3\nx") == "15"

    def test_ifloordiv(self):
        assert python_repl("x = 10\nx //= 3\nx") == "3"

    def test_imod(self):
        assert python_repl("x = 10\nx %= 3\nx") == "1"

    def test_iadd_subscript(self):
        assert python_repl("d = {'k': 1}\nd['k'] += 10\nd") == "{'k': 11}"


class TestControlFlow:
    def test_if(self):
        assert python_repl("x = 5\nif x > 0: result = 'pos'\nresult") == "pos"

    def test_if_else(self):
        assert python_repl("x = -1\nif x > 0:\n  r = 'pos'\nelse:\n  r = 'neg'\nr") == "neg"

    def test_if_elif_else(self):
        code = "x = 0\nif x > 0:\n  r = 'pos'\nelif x < 0:\n  r = 'neg'\nelse:\n  r = 'zero'\nr"
        assert python_repl(code) == "zero"

    def test_for_loop(self):
        code = "total = 0\nfor x in range(5):\n  total += x\ntotal"
        assert python_repl(code) == "10"

    def test_for_loop_append(self):
        code = "evens = []\nfor x in range(6):\n  if x % 2 == 0:\n    evens.append(x)\nevens"
        assert python_repl(code) == "[0, 2, 4]"

    def test_break_blocked(self):
        # break/continue are not in the whitelist
        assert "Error:" in python_repl("for x in [1, 2, 3]:\n  break")

    def test_continue_blocked(self):
        assert "Error:" in python_repl("for x in [1, 2, 3]:\n  continue")


# ---------------------------------------------------------------------------
# Method calls (whitelisted)
# ---------------------------------------------------------------------------


class TestStringMethods:
    def test_split(self):
        assert python_repl("'a,b,c'.split(',')") == "['a', 'b', 'c']"

    def test_join(self):
        assert python_repl("'-'.join(['a', 'b', 'c'])") == "a-b-c"

    def test_replace(self):
        assert python_repl("'hello'.replace('l', 'L')") == "heLLo"

    def test_upper_lower(self):
        assert python_repl("'Hello'.upper().lower()") == "hello"

    def test_strip(self):
        assert python_repl("'  hi  '.strip()") == "hi"

    def test_startswith_endswith(self):
        assert python_repl("'hello'.startswith('he')") == "True"
        assert python_repl("'hello'.endswith('lo')") == "True"

    def test_isdigit_isalpha(self):
        assert python_repl("'123'.isdigit()") == "True"
        assert python_repl("'abc'.isalpha()") == "True"

    def test_zfill(self):
        assert python_repl("'42'.zfill(5)") == "00042"


class TestListMethods:
    def test_append(self):
        assert python_repl("xs = [1, 2]\nxs.append(3)\nxs") == "[1, 2, 3]"

    def test_extend(self):
        assert python_repl("xs = [1]\nxs.extend([2, 3])\nxs") == "[1, 2, 3]"

    def test_insert(self):
        assert python_repl("xs = [1, 3]\nxs.insert(1, 2)\nxs") == "[1, 2, 3]"

    def test_pop(self):
        assert python_repl("xs = [1, 2, 3]\nxs.pop()") == "3"

    def test_remove(self):
        assert python_repl("xs = [1, 2, 3]\nxs.remove(2)\nxs") == "[1, 3]"

    def test_sort(self):
        assert python_repl("xs = [3, 1, 2]\nxs.sort()\nxs") == "[1, 2, 3]"

    def test_reverse(self):
        assert python_repl("xs = [1, 2, 3]\nxs.reverse()\nxs") == "[3, 2, 1]"


class TestDictMethods:
    def test_keys(self):
        assert python_repl("sorted({'a': 1, 'b': 2}.keys())") == "['a', 'b']"

    def test_values(self):
        assert python_repl("sorted({'a': 1, 'b': 2}.values())") == "[1, 2]"

    def test_get(self):
        assert python_repl("{'a': 1}.get('a')") == "1"

    def test_get_default(self):
        assert python_repl("{'a': 1}.get('missing', 42)") == "42"

    def test_update(self):
        assert python_repl("d = {'a': 1}\nd.update({'b': 2})\nd") == "{'a': 1, 'b': 2}"

    def test_pop_dict(self):
        assert python_repl("d = {'a': 1, 'b': 2}\nd.pop('a')") == "1"


# ---------------------------------------------------------------------------
# Whitelisted builtins
# ---------------------------------------------------------------------------


class TestBuiltins:
    def test_len(self):
        assert python_repl("len([1, 2, 3])") == "3"

    def test_sorted(self):
        assert python_repl("sorted([3, 1, 2])") == "[1, 2, 3]"

    def test_sum(self):
        assert python_repl("sum([1, 2, 3])") == "6"

    def test_min_max(self):
        assert python_repl("min([3, 1, 2])") == "1"
        assert python_repl("max([3, 1, 2])") == "3"

    def test_abs(self):
        assert python_repl("abs(-7)") == "7"

    def test_round(self):
        assert python_repl("round(3.7)") == "4"

    def test_range(self):
        assert python_repl("list(range(5))") == "[0, 1, 2, 3, 4]"

    def test_enumerate(self):
        assert python_repl("list(enumerate(['a', 'b']))") == "[(0, 'a'), (1, 'b')]"

    def test_zip(self):
        assert python_repl("list(zip([1, 2], ['a', 'b']))") == "[(1, 'a'), (2, 'b')]"

    def test_all_any(self):
        assert python_repl("all([True, True, False])") == "False"
        assert python_repl("any([False, False, True])") == "True"

    def test_type_conversions(self):
        assert python_repl("int('42')") == "42"
        assert python_repl("float('3.14')") == "3.14"
        assert python_repl("str(42)") == "42"
        assert python_repl("bool(0)") == "False"

    def test_isinstance(self):
        assert python_repl("isinstance(1, int)") == "True"

    def test_ord_chr(self):
        assert python_repl("ord('A')") == "65"
        assert python_repl("chr(65)") == "A"

    def test_filter_via_listcomp(self):
        # lambdas are rejected (see security tests) — use a list comprehension instead.
        assert python_repl("[x for x in [1, 2, 3] if x > 1]") == "[2, 3]"


class TestMathFunctions:
    """Math helpers are exposed as bare names (gcd, factorial, ...), not as ``math.X``."""

    def test_gcd(self):
        assert python_repl("gcd(12, 18)") == "6"

    def test_factorial(self):
        assert python_repl("factorial(5)") == "120"

    def test_isqrt(self):
        assert python_repl("isqrt(50)") == "7"

    def test_comb(self):
        assert python_repl("comb(5, 2)") == "10"

    def test_perm(self):
        assert python_repl("perm(5, 2)") == "20"


# ---------------------------------------------------------------------------
# Error / security boundary
# ---------------------------------------------------------------------------


class TestPythonReplErrorCapture:
    def test_error_with_partial_output(self):
        result = python_repl('print("start")\n1/0')
        assert result == "start\n\nError: division by zero"

    def test_error_without_output(self):
        result = python_repl("1/0")
        assert result == "Error: division by zero"

    def test_syntax_error(self):
        result = python_repl("def foo():")
        assert result.startswith("Error:")

    def test_blocked_function(self):
        result = python_repl("eval('1')")
        assert "Error:" in result


class TestSecurityBoundary:
    """Each of these must NOT execute — they should return an error string."""

    def test_import_blocked(self):
        assert "Error:" in python_repl("import os")

    def test_from_import_blocked(self):
        assert "Error:" in python_repl("from os import system")

    def test_dunder_class_blocked(self):
        assert "Error:" in python_repl("().__class__")

    def test_dunder_dict_blocked(self):
        result = python_repl("().__dict__")
        assert "Error:" in result

    def test_eval_blocked(self):
        assert "Error:" in python_repl("eval('2+2')")

    def test_exec_blocked(self):
        assert "Error:" in python_repl("exec('x=1')")

    def test_compile_blocked(self):
        assert "Error:" in python_repl("compile('1', '<>', 'eval')")

    def test_open_blocked(self):
        assert "Error:" in python_repl("open('/etc/passwd')")

    def test_dunder_import_blocked(self):
        assert "Error:" in python_repl("__import__('os')")

    def test_globals_blocked(self):
        assert "Error:" in python_repl("globals()")

    def test_locals_blocked(self):
        assert "Error:" in python_repl("locals()")

    def test_getattr_blocked(self):
        assert "Error:" in python_repl("getattr([], 'append')")

    def test_setattr_blocked(self):
        assert "Error:" in python_repl("setattr([], 'x', 1)")

    def test_fstring_blocked(self):
        assert "Error:" in python_repl('x = 1\nf"{x}"')

    def test_walrus_blocked(self):
        assert "Error:" in python_repl("(x := 5)")

    def test_while_loop_blocked(self):
        assert "Error:" in python_repl("while True: pass")

    def test_function_def_blocked(self):
        assert "Error:" in python_repl("def f(): return 1")

    def test_class_def_blocked(self):
        assert "Error:" in python_repl("class C: pass")

    def test_lambda_blocked(self):
        # Lambda is rejected (function definition)
        assert "Error:" in python_repl("(lambda x: x)(1)")

    def test_star_args_blocked(self):
        assert "Error:" in python_repl("f(*[1, 2, 3])")

    def test_unknown_builtin_blocked(self):
        # print is allowed; vars() is not
        assert "Error:" in python_repl("vars()")

    def test_attribute_chain_to_unsafe(self):
        # Try to reach through a safe attribute to an unsafe one
        result = python_repl("''.__class__.__mro__")
        assert "Error:" in result
