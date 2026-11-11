"""Seed reviewer prompts: the AI Scientist reviewer's prompts, with the overall score on ICLR 2026's scale.

The strings are read verbatim from third_party/AI-Scientist/ai_scientist/perform_review.py
(SakanaAI/AI-Scientist @ 1de1dbc1). We parse them instead of importing the module, which pulls in PDF
libraries we don't use. The only edits are the two ICLR-scale replacements below and filling in the
reflection prompt's round counter, which the original never formats.

ICLR 2026 rating labels are taken from papers quoting the 2026 scale, and consistent with ICLR 2026 meta-reviews in our data.
"""
import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "third_party/AI-Scientist/ai_scientist/perform_review.py"
_NAMES = ["reviewer_system_prompt_base", "reviewer_system_prompt_neg", "template_instructions",
          "neurips_form", "reviewer_reflection_prompt"]


def _load():
    nodes = {n.targets[0].id: n.value for n in ast.parse(SOURCE.read_text()).body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in _NAMES}
    ns = {}
    for name in _NAMES:  # in dependency order: later strings concatenate earlier ones
        ns[name] = eval(compile(ast.Expression(nodes[name]), str(SOURCE), "eval"), {}, ns)
    return ns


def _replace(text, old, new):
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


_ais = _load()

OVERALL_FIELD_OLD = '- "Overall": A rating from 1 to 10 (very strong reject to award quality).'
OVERALL_FIELD_NEW = '- "Overall": A rating that must be one of 0, 2, 4, 6, 8, 10 (strong reject to strong accept).'

OVERALL_SCALE_OLD = _ais["neurips_form"][_ais["neurips_form"].index("8. Overall:"):_ais["neurips_form"].index("9. Confidence:")]
OVERALL_SCALE_NEW = """8. Overall: Please provide an "overall score" for this submission. Choices:
  10: strong accept, should be highlighted at the conference
  8: accept, good paper
  6: marginally above the acceptance threshold
  4: marginally below the acceptance threshold
  2: reject, not good enough
  0: strong reject

"""

SYSTEM_PROMPT = _ais["reviewer_system_prompt_neg"]  # perform_review's default system prompt
REVIEW_FORM = _replace(_replace(_ais["neurips_form"], OVERALL_SCALE_OLD, OVERALL_SCALE_NEW),
                       OVERALL_FIELD_OLD, OVERALL_FIELD_NEW)
REFLECTION_PROMPT = _ais["reviewer_reflection_prompt"].replace("{current_round}", "2").replace("{num_reflections}", "2")

PAPER_TEMPLATE = """
Here is the paper you are asked to review:
```
{text}
```"""  # appended to the review form, as in perform_review
