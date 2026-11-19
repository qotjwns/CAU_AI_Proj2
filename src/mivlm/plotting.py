"""
그림의 공통 모양: 색, matplotlib 설정, 제목, 저장
실험마다 있는 plot.py가 이 파일의 색과 함수를 써서 같은 모양의 그림을 만든다.
쓰는 쪽에서 plt.rcParams.update(STYLE)을 먼저 한 번 호출한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# 배경, 글자, 축의 색 (밝은 배경 기준)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"  # 강조하지 않는 표시
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# 서로 다른 대상을 구분하는 색. 대상이 하나면 BLUE만, 여럿이면 BLUE, ORANGE, AQUA 순서로 쓴다
# (색각 이상이 있어도 세 색이 서로 구분되는 조합)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"  # 위 세 색과 겹치지 않게 한 부분만 강조할 때

# 값이 클수록 진해지는 한 가지 색의 단계
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
}


def set_title(ax, title: str, subtitle: str) -> None:
    """축 왼쪽 위에 굵은 제목을, 그 아래에 작은 부제목을 단다"""
    ax.set_title(title, loc="left", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)


def figure_title(fig, title: str, subtitle: str) -> None:
    """칸이 여러 개인 그림의 제목과 부제목을 첫 칸의 왼쪽 끝에 맞춰 그림 위에 단다 (칸 이름은 칸마다 따로 단다)"""
    left = fig.axes[0].get_position().x0
    height = fig.get_figheight()
    fig.text(left, 0.88 + 0.72 / height, title, fontsize=13, fontweight="bold", va="bottom")
    fig.text(left, 0.88 + 0.46 / height, subtitle, fontsize=9, color=INK_SECONDARY, va="bottom")


def style_layer_axis(ax, y_min: float, y_label: str | None, note: bool = True) -> None:
    """층이 가로축인 칸의 축을 꾸민다. 마지막 항목은 최종 norm이 적용된 값이라 세로선으로 구분한다

    선을 먼저 그린 뒤에 호출한다 (층 수를 첫 선에서 읽는다). 세로축은 y_min부터 1까지다.
    y_label: 세로축 이름. None이면 달지 않는다 (세로축을 같이 쓰는 칸)
    note: 세로선 옆에 설명을 적을지. 칸이 여러 개면 한 칸에만 적는다
    """
    n_entries = len(ax.lines[0].get_xdata())
    ax.set_xlim(0, n_entries - 1)
    ax.set_ylim(y_min, 1.004)
    ax.set_xlabel("layer (0 = before the first layer)")
    if y_label:
        ax.set_ylabel(y_label)
    ax.grid(axis="y")
    ax.axvline(n_entries - 1.5, color=MUTED, linewidth=0.8, linestyle=":")
    if note:
        ax.text(n_entries - 1.7, y_min + 0.004, "last entry:\nafter final norm", ha="right", va="bottom", color=MUTED,
                fontsize=8)


def save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")
