#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验一 Hopfield 模型的实现"
#let location = "K2-218"

#set document(title: title, author: author, date: date)
#set text(
    font: ((name: "lato", covers: "latin-in-cjk"), "noto serif cjk sc", "noto sans cjk sc"),
    size: zh(5),
    lang: "zh",
    region: "cn",
)
#set par(justify: true, first-line-indent: (amount: 2em, all: true))
#set page(
    paper: "a4",
    margin: (x: 35pt, y: 35pt),
)
#set heading(numbering: numbly("{1:一}", "{2:1}.", "({3:1})"))
#show heading: set text(size: zh(-4))
#{
    v(1fr)
    align(center, text(zh(1), weight: "bold")[实验报告])
    v(2fr)
    set text(zh(4))
    align(center, grid(
        row-gutter: 1em,
        align: (left, center),
        columns: (6em, auto),
        strong[实验名称：], title,
        strong[课程名称：], course,
        strong[实验地点：], location,
        strong[实验日期：], date.display("[year].[month].[day]"),
        strong[班级：], class,
        strong[姓名], author,
        strong[学号], id,
    ))
    v(3fr)
}
#pagebreak()
#counter(page).update(1)
#set page(
    footer: align(center, context counter(page).display("- 1 -")),
)
#show raw: set text(font: ("jetbrains mono", "noto serif cjk sc"))
#show raw.where(block: false): box.with(
    fill: luma(240),
    inset: (x: .3em, y: 0em),
    outset: (x: 0em, y: .3em),
    radius: .2em,
)
#show: codly-init
#codly(
    languages: codly-languages,
    zebra-fill: none,
    fill: luma(90.2%),
    stroke: .5pt + rgb("bfbfbf"),
    radius: 8pt,
)
#set enum(numbering: numbly("{1:1})", "{2:a}."))
#set list(indent: 10pt, marker: sym.bullet.tri)

= 实验目的：

- 理解离散 Hopfield 网络的反馈连接、能量下降与联想记忆机制；设计 $0$–$9$ 的数字点阵，
- 通过权重学习和异步状态更新，从受到像素翻转干扰的输入中恢复原有数字，并比较不同学习规则的记忆效果。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

+ 设计 $0$–$9$ 共十个 $6 times 5$ 数字点阵，有笔画的像素为 $1$，空白为 $-1$，逐行展平为长度为 $30$ 的向量。
+ 构建具有 $30$ 个神经元的离散 Hopfield 网络，采用对称权重，去除自连接，阈值为零。
+ 在数字点阵中随机选取不同位置，翻转像素符号，生成不同强度的噪声。
+ 将带噪输入送入网络迭代，记录恢复结果、完整恢复率和能量变化。

= 实验步骤：

== 点阵与网络设计

用六段长度为 $5$ 的字符串表示一个数字。读入时转换为 $-1$。例如，数字 $0$ 对应：

$
    mat(
        -1, 1, 1, 1, -1;
        1, -1, -1, -1, 1;
        1, -1, -1, -1, 1;
        1, -1, -1, -1, 1;
        1, -1, -1, -1, 1;
        -1, 1, 1, 1, -1;
        align: #right
    )
$

将十个行向量组成模式矩阵 $X in RR^(10 times 30)$，网络权重为$W in RR^(30 times 30)$。实验对比两种权重构造方法，均在构造后将对角线置零：

$
    W_"Hebb" = 1 / 30 X^T X, quad
    P = X^+ X, quad
    W_"伪逆" = P - op("diag")(P).
$

其中 $X^+$ 是 Moore–Penrose 伪逆，$op("diag")(P)$ 表示由 $P$ 的对角元素构成的对角矩阵。Hebb 规则直接累积像素相关性，数字笔画和空白区域的相关性容易造成记忆之间的干扰。伪逆规则把状态投影到存储模式张成的子空间，仍使用同一个离散 Hopfield 网络进行反馈更新。

由于 $P X^T = X^T$，对任一存储模式 $x$，有
$(W_"伪逆" x)_i = (1 - P_(i i)) x_i$。
本次点阵矩阵的秩为 $10$，实测最小对齐局部场
$min_(x,i) x_i (W_"伪逆" x)_i approx 0.5043 > 0$，
因此十个原始模式均为严格稳定状态。

== 更新规则与实验流程

令 $s in {-1, 1}^30$ 为当前状态，局部场为 $h_i = sum_j w_(i j) s_j$。
每轮按照 $0$–$29$ 的顺序逐个更新神经元，后面的神经元立即使用前面已经更新的状态：

$
    s_i <- cases(
        1 & h_i > epsilon,
        -1 & h_i < -epsilon,
        s_i & abs(h_i) <= epsilon,
    ), quad epsilon = 10^(-10).
$

零场附近保留原状态，避免浮点误差导致不必要的翻转。若一整轮状态不变则停止，最多运行 $100$ 轮；超过上限时明确报错，不把尚未收敛的状态当作结果。

网络能量定义为 $E(s) = -1 / 2 s^T W s$。由于权重对称且无自连接，单个神经元更新时 $Delta E = -(s_i^"新" - s_i^"旧") h_i <= 0$。
发生状态翻转时能量严格下降，状态空间有限，因此这种异步更新能够到达固定点。固定点也可能是其他数字或未存储的模式，收敛本身不等于识别正确。

#align(center)[
    点阵编码 $arrow.r$ 构造权重 $arrow.r$ 翻转像素 $arrow.r$ 异步恢复 $arrow.r$ 比较与绘图
]

测试使用 ```python np.random.default_rng(42)```，每次不放回地选择 $k$ 个位置并乘以 $-1$。取 $k in {0, 1, 2, 3, 6, 9}$；无噪声时每个数字测试一次，其余等级每个数字测试 $100$ 次。两种学习规则分别重置为同一随机种子，保证比较的输入完全一致。

完整恢复率定义为：

$
    R = "输出与原数字逐像素完全相同的样本数" / "测试样本总数" times 100%.
$

输出恰好等于某个存储模板时才读出该数字；其余输出标记为 `?`，避免用最近模板匹配掩盖网络的恢复失败。实验测试的是已存储模板的噪声恢复能力。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.oy
```

== 程序实验结果：

=== 噪声恢复率

非零噪声每个等级共测试 $10 times 100 = 1000$ 次。

#let results = csv("../output/results.csv").slice(1)
#let pct(n, d) = eval(mode: "math", str(calc.round(100 * float(n) / float(d), digits: 1)) + "%")
#figure(
    table(
        columns: (auto, auto, auto, auto, auto),
        inset: 7pt,
        align: center,
        table.header([*翻转数*], [*噪声比例*], [*样本数*], [*Hebb 恢复率*], [*伪逆恢复率*]),
        ..results
            .map(row => (
                row.at(0),
                pct(row.at(0), 30),
                row.at(1),
                pct(row.at(2), row.at(1)),
                pct(row.at(3), row.at(1)),
            ))
            .flatten(),
    ),
    caption: [不同噪声强度下的完整恢复率],
) <scores>

=== 点阵恢复与能量变化

#figure(
    image("../output/recall.svg"),
    caption: [十个数字的一次 $10%$ 噪声恢复示例。由上至下为原始、带噪和恢复后的点阵；黑色为 $1$，白色为 $-1$。],
) <recall>

#figure(
    image("../output/metrics.svg"),
    caption: [完整恢复率对比，以及数字 $0$ 的异步更新能量轨迹。横轴每一步表示更新一个神经元。],
) <metrics>

== 分析和改进：

*权重学习决定记忆能否保持。* 本实验只有 $30$ 个神经元，却需要同时存储十个相似数字。Hebb 权重叠加了共同笔画与空白的相关性，局部场不足以维持每个模式。伪逆学习通过模式子空间构造权重，保留了十个原始记忆；改进只涉及权重计算，状态仍由 Hopfield 网络自身逐个更新。

*抗噪能力受到点阵相似性与吸引域限制。* 本组模板间的最小汉明距离为 $4$，两个像素的改变就可能使输入处于两个模板之间。即便原始记忆稳定，也不能保证任意带噪输入都落入正确记忆的吸引域，网络更不保证输出全局最近的模板。本次全部单像素翻转可恢复，双像素开始出现歧义，高噪声下伪吸引子和错误记忆明显增多。

*结果应按统一标准解释。* 本实验用完整恢复率评价网络，恢复为错误数字和未存储状态都计为失败。固定种子和一致的噪声样本便于复现、比较；$1000$ 次随机测试仍是有限样本结果，不能推出对所有噪声位置均有效。增加点阵分辨率、减小模板相似性或比较不同更新顺序，可以作为后续实验方向。

= 实验小结：

实现了从点阵编码、网络学习、噪声生成到联想恢复的完整流程。伪逆 Hopfield 网络稳定记住了 $0$–$9$，能完整纠正全部单像素翻转，在本次 $10%$ 随机噪声测试中达到 $89.1%$ 的完整恢复率。能量下降解释了网络为何收敛，错误记忆和伪吸引子则说明收敛与正确识别之间仍有区别。

通过与 Hebb 权重的对比，可以看到存储模式的相关性、学习规则和输入噪声共同影响联想记忆。
