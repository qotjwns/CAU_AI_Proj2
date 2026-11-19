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


def print_class_table(title: str, table: dict[str, dict], columns: dict[str, tuple[str, str]]) -> None:
    """행은 클래스, 열은 columns인 표를 출력한다

    table: {클래스: {키: 값}}. columns: {열 이름: (키, 값의 형식)}. 예: {"yes PPP": ("yes_rate_PPP", ".0%")}
    """
    print(f"\n[{title}]")
    print(f"{'':<14}" + "".join(f"{header:>10}" for header in columns))
    for name, values in table.items():
        cells = [format(values[key], value_format) for key, value_format in columns.values()]
        print(f"{name:<14}" + "".join(f"{cell:>10}" for cell in cells))


def interval_text(mean: float, low: float, high: float) -> str:
    """값과 구간을 "+0.123 [+0.100, +0.150]" 모양으로 적는다"""
    return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"
