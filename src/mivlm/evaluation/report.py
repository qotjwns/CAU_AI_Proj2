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


def print_layer_table(title: str, columns: dict, value_format: str) -> None:
    """행은 층, 열은 columns의 이름인 표를 출력한다

    columns: {열 이름: 층마다의 값, shape (층 수,)인 텐서}
    value_format: 값의 형식. 예: ".5f"
    """
    print(f"\n[{title}]")
    print("layer" + "".join(f"{name:>10}" for name in columns))
    for layer in range(len(next(iter(columns.values())))):
        print(f"{layer:>5}" + "".join(f"{values[layer].item():>10{value_format}}" for values in columns.values()))
