import re
from sympy import sympify, simplify, N

DATASET = ["SQA", "ECQA", "ARC", "GSM8K", "MATH", "MATH500", "MMLUProPhys", "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol"]

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv_1(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def normalize_fraction_notation(expr):
    """
    Normalizes different fraction notations (\\frac, \frac, rac) into a consistent format.

    :param expr: A string containing the expression with fraction notations.
    :return: A string with the normalized fraction format.
    """
    # Regular expression to find different fraction notations
    frac_pattern = r"(\\\\frac|\\frac|rac|\\dfrac)\{([^}]+)\}\{([^}]+)\}"

    # Function to replace the fraction notations with (numerator)/(denominator)
    def replace_frac(match):
        _, num, den = match.groups()
        return f"({num})/({den})"

    # Replace all fraction expressions in the input string
    return re.sub(frac_pattern, replace_frac, expr)

def is_equiv_2(expr1, expr2):
    """
    Determines whether two mathematical expressions, possibly containing different fraction notations,
    are equivalent.

    :param expr1: A string representing the first mathematical expression.
    :param expr2: A string representing the second mathematical expression.
    :return: True if the expressions are equivalent, False otherwise.
    """
    try:
        # Normalize fraction notations
        expr1_sympy = normalize_fraction_notation(expr1)
        expr2_sympy = normalize_fraction_notation(expr2)

        # Convert the string expressions into sympy expressions
        sympy_expr1 = sympify(expr1_sympy)
        sympy_expr2 = sympify(expr2_sympy)

        # Simplify both expressions and check for equality
        try:
            ret1=bool(simplify(sympy_expr1 - sympy_expr2) == 0)
        except:
            ret1=False
        try:
            ret2=bool(abs(N(sympy_expr1) - N(sympy_expr2)) <= 0.01)
        except:
            ret2=False
        return ret1 or ret2
    except:
        pass
        # print(e)
        # If the expression cannot be parsed, return False
        return False

def is_equiv(str1, str2):
    ret1=is_equiv_1(str1, str2)
    ret2=is_equiv_2(str1, str2)
    return ret1 or ret2
    
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    if isinstance(s, str):
        # left = 
        if s.startswith("\\boxed{") and s.endswith('}'):
            return s[len("\\boxed{"):-1]
        elif s.startswith("\\fbox") and s.endswith('}'):
            return s[len("\\fbox"):-1]
        else:
            return None
    else:
        return None

def extract_from_boxed(text):
    pattern = r"\$(.*?)(?=\$|###|```|\n|$)"
    match = re.findall(pattern, text)
    ret=None
    if match:
        ret=remove_boxed(last_boxed_only_string((match[0]).strip()))
        if not ret:
            ret=(match[0]).strip()
        ret='$'+ret+'$'
    return ret

def extract_from_boxed_fallback(text):
    pattern = r"\$(.*?)(?=\$|###|```|\n|$)"
    match = re.findall(pattern, text)
    ret=None
    if match:
        ret=remove_boxed(last_boxed_only_string((match[-1]).strip()))
        if not ret:
            ret=(match[-1]).strip()
        ret='$'+ret+'$'
    return ret

def parse_answer(dataset, text):
    assert dataset in DATASET
    if dataset == "SQA":
        matches = re.findall(r'yes|no', text, re.IGNORECASE)
        return matches[0].lower() if matches else None
    elif dataset == "ECQA":
        matches = re.findall(r'([1-5])', text)
        return matches[0] if matches else None
    elif dataset == "ARC":
        matches = re.findall(r'([A|B|C|D])', text)
        return matches[0] if matches else None
    elif dataset in ["MMLUProPhys", "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol"]:
        matches = re.findall(r'([A|B|C|D|E|F|G|H|I|J])', text)
        return matches[0] if matches else None
    elif dataset == "GSM8K":
        # match = re.findall(r"answer: \$?([0-9]+\.?[0-9]*)", text)
        # if match:
        #     return float(match[0])
        # else:
        match = re.findall(r"\$?([0-9]+\.?[0-9]*)", text)
        if match:
            return float(match[0])
        else:
            return None
    elif dataset == "MATH":
        return extract_from_boxed(text)

def parse_answer_fallback(dataset, text):
    assert dataset in DATASET
    if dataset == "SQA":
        matches = re.findall(r'yes|no', text, re.IGNORECASE)
        return matches[-1].lower() if matches else None
    elif dataset == "ECQA":
        matches = re.findall(r'([1-5])', text)
        return matches[-1] if matches else None
    elif dataset == "ARC":
        matches = re.findall(r'([A|B|C|D])', text)
        return matches[-1] if matches else None
    elif dataset in ["MMLUProPhys", "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol"]:
        matches = re.findall(r'([A|B|C|D|E|F|G|H|I|J])', text)
        return matches[-1] if matches else None
    elif dataset == "GSM8K":
        match = re.findall(r"\$?([0-9]+\.?[0-9]*)", text)
        if match:
            return float(match[-1])
        else:
            return None
    elif dataset == "MATH":
        return extract_from_boxed_fallback(text)

def calc_acc(dataset, test_pred, test_samples):
    assert dataset in DATASET
    num_correct, num_samples = 0, len(test_pred)
    if dataset in ["SQA", "ECQA", "ARC", "MMLUProPhys", "MMLUProComp", "MMLUProChem", "MMLUProMath", "MMLUProBiol", "MMLUProHist"]:
        for e, t in enumerate(test_pred):
            if t == test_samples[e]['answer']:
                num_correct += 1
    elif dataset == "GSM8K":
        for e, t in enumerate(test_pred):
            if t == float(test_samples[e]['answer']):
                num_correct += 1
    elif dataset == "MATH":
        for e, t in enumerate(test_pred):
            if is_equiv(str(t).strip('$'), test_samples[e]['answer']):
                num_correct += 1
    return round(num_correct/num_samples, 4)
