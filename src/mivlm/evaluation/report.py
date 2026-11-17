"""
지표를 터미널에 표로 출력
"""
COLUMNS = ["n", "accuracy", "false_yes_rate", "false_no_rate", "yes_rate", "mean_margin", "n_tie", "n_boundary", "n_none"]


def format_value(column: str, value) -> str:
    """지표 하나를 표에 넣을 문자열로 바꾼다. 비율은 %, 값이 없으면 '-'"""
    if value is None:
        return "-"
    if column in ("accuracy", "false_yes_rate", "false_no_rate", "yes_rate"):
        return f"{value:.1%}"
    if column == "mean_margin":
        return f"{value:+.3f}"
    return str(value)


def print_table(title: str, summaries: dict[str, dict]) -> None:
    """summaries: {행 이름: mivlm.evaluation.metrics.summarize의 결과}"""
    print(f"\n[{title}]")
    print(f"{'':<18}" + "".join(f"{column:>16}" for column in COLUMNS))
    for name, summary in summaries.items():
        print(f"{name:<18}" + "".join(f"{format_value(column, summary[column]):>16}" for column in COLUMNS))
