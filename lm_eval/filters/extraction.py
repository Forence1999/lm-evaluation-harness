import re
import sys
import unicodedata
import pandas as pd

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


@register_filter("regex")
class RegexFilter(Filter):
    """ """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # ----------- djw test, multiple answers (3)
        def fix_fracs(string):
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
                        except AssertionError:
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

        def fix_a_slash_b(string):
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
            except (AssertionError, ValueError) as e:
                print(type(e).__name__, string)
                return string

        def remove_right_units(string):
            # "\\text{ " only ever occurs (at least in the val set) when describing units
            if "\\text{ " in string:
                splits = string.split("\\text{ ")
                assert len(splits) == 2
                return splits[0]
            else:
                return string

        def fix_sqrt(string):
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

        def strip_string(string):
            # linebreaks
            string = string.replace("\n", "")

            # remove inverse spaces
            string = string.replace("\\!", "")

            # replace \\ with \
            string = string.replace("\\\\", "\\")

            # replace tfrac and dfrac with frac
            string = string.replace("tfrac", "frac")
            string = string.replace("dfrac", "frac")

            # remove \left and \right
            string = string.replace("\\left", "")
            string = string.replace("\\right", "")

            # Remove circ (degrees)
            string = string.replace("^{\\circ}", "")
            string = string.replace("^\\circ", "")

            # remove dollar signs
            string = string.replace("\\$", "")

            # remove units (on the right)
            string = remove_right_units(string)

            # remove percentage
            string = string.replace("\\%", "")
            string = string.replace("\%", "")  # noqa: W605

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
            string = fix_sqrt(string)

            # remove spaces
            string = string.replace(" ", "")

            # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
            string = fix_fracs(string)

            # manually change 0.5 --> \frac{1}{2}
            if string == "0.5":
                string = "\\frac{1}{2}"

            # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
            string = fix_a_slash_b(string)

            return string

        def normalization(inst):
            normalized = []
            # inst 只包含一个问题的所有答案，resps = 单个答案的str
            for resps in inst:
                indices = [pos for pos, char in enumerate(resps) if char == "$"]
                if len(indices) <= 1:
                    answer = strip_string(resps)
                elif len(indices) == 2 or len(indices) == 3:
                    answer = strip_string(resps[indices[0] + 1 : indices[-1]])
                else:
                    print("len($)=", len(indices), "    resps: ", resps)
                    answer = strip_string(resps)
                    # 两个答案的怎么办？
                normalized.append(answer)
            return normalized

        # ----------- djw test, multiple answers (3)

        # filtered_resps = list(map(lambda x: filter_set(x), resps))
        # return filtered_resps

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        normalized_resps = list(map(lambda x: normalization(x), filtered_resps))
        return normalized_resps

        # save = 0
        # if save:
        #    # Rachel Begin
        #    x = [list(filtered_resps[i]) for i in range(len(filtered_resps))]
        #    beam_width = 8
        #    if len(x[0]) == beam_width:
        #        #print("x ", x)
        #        generated_answers = pd.DataFrame(x, dtype="string")
        #        #print("generated_answers: ", generated_answers)
        #        generated_answers.to_csv('/workspace/temp/extract_answers_consis.csv', index=False, header=None)
        #    # Rachel End

        # filtered_resps [num_questions, beam_width]提取出来的答案，字符串


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """ """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """
    A filter used to extract a model's answer on multiple choice questions with
    letter answers. assumes each document has a "choices" field
    containing the list of answer choices and that the answer label symbols
    are of the form (A), (B), (C), ... or A, B, C.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                f":[\s]*({without_paren_fallback_regex})"
            )

            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps
