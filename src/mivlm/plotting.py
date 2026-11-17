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


def save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")
