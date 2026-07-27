"""pp_checker — Plan & Profile IRC compliance checker."""
from . import extract, ocr, parse, rules, validate, report
from .llm_fallback import LLMReader

__version__ = "0.1.0"


def run(pdf_path, road_class="2_lane", terrain="mountainous", api_key=None):
    """One-call pipeline: PDF -> (model, rule_result, cross_checks)."""
    reader = LLMReader(api_key=api_key)
    model = parse.assemble(pdf_path, llm_reader=reader if reader.available else None)
    rule_result = rules.check(model, road_class=road_class, terrain=terrain)
    xcheck = validate.cross_checks(model)
    return model, rule_result, xcheck
