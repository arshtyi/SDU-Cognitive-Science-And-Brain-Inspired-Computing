#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验三 Hebb 学习"
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

- 理解 Hebb 学习依据神经元共同活动改变连接权重的原理，使用 NumPy 实现简单的自联想记忆网络。
- 设计数字 $0$–$2$ 的双极性点阵，学习像素之间的相关性，测试网络从带噪输入中恢复原始点阵的能力。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

+ 将数字 $0$、$1$、$2$ 设计为 $6 times 5$ 点阵，有笔画处记为 $1$，空白处记为 $-1$，展平后保存在 Python 列表中。
+ 创建 $30$ 个神经元的自联想网络，从零附近的小随机权重开始，使用 Hebb 规则学习三个点阵。
+ 对训练点阵进行随机像素翻转，将带噪点阵作为测试输入，记录网络输出、数字识别结果及不同噪声强度下的完整恢复率。

= 实验步骤：

== 数据与程序结构

`DIGITS` 保存三组字符串，每组包含六行、每行五列；`PATTERNS` 将字符串中的 `1` 转为整数 $1$、`0` 转为整数 $-1$，形成包含三个长度为 $30$ 的列表的二维列表。运行时转换为 $X in RR^(3 times 30)$，每行是一个训练模式。数字编号仅用于结果标注和评价，不参与权重更新。

== Hebb 学习规则与网络设计

$
    Delta w_(i j) = eta y_i x_j.
$

采用该规则的自联想存储方式：学习模式 $x^mu$ 时，由外部输入将各神经元状态固定为该模式，因此连接两端的活动为 $y_i=x_i^mu$、$x_j=x_j^mu$。这里训练时的状态由呈现的点阵固定；手册中的 $y_i=f(W_i^T x)$ 描述神经元自由响应，在恢复阶段使用。训练过程中不计算分类标签误差，也不使用反向传播。

学习率取 $eta=1/30$。固定种子 $42$，先独立生成 $R_(i j) tilde U(-0.001,0.001)$，对称化得到零附近的小随机初始权重 $W^(0)=(R+R^T)/2$。三个模式依次呈现一次：

$
    W <- W + eta x^mu (x^mu)^T, quad mu=0,1,2.
$

最终权重为：

$
    w_(i j) = cases(
        w_(i j)^(0) + 1/30 sum_(mu=0)^2 x_i^mu x_j^mu & i != j,
        0 & i = j.
    )
$

同号像素产生正增量，异号像素产生负增量，反映像素活动的相关性。对角线置零表示网络不使用自连接；对称化后的初始矩阵与每次外积更新均对称，因此最终权重保持对称。训练只需遍历三个模式一次，重复同样的呈现会继续累加权重，没有必要将它写成以分类误差为目标的多轮优化。

== 带噪输入与联想恢复

噪声通过在 $30$ 个位置中不放回抽取 $k$ 个位置并乘以 $-1$ 产生，确保恰好翻转 $k$ 个不同像素。函数复制原始点阵，避免训练模板被修改。采用 $k in {0,1,2,3,6,9}$，对应 $0%$、约 $3.3%$、约 $6.7%$、$10%$、$20%$、$30%$ 噪声。

将带噪点阵作为网络初始状态 $s$，阈值取零，每轮按索引 $0$–$29$ 顺序异步更新，后续神经元立即使用已经更新的状态：

$
    h_i = sum_j w_(i j) s_j, quad
    s_i <- cases(
        1 & h_i > epsilon,
        -1 & h_i < -epsilon,
        s_i & abs(h_i) <= epsilon,
    ), quad epsilon = 10^(-10).
$

零局部场附近保留原状态，使输出始终为 $-1$ 或 $1$。一整轮状态不变时停止；最多更新 $100$ 轮，超限则抛出 `RuntimeError`，不将未收敛状态当作识别结果。该恢复过程使用离散 Hopfield 式反馈动力学，存储权重全部由上述 Hebb 规则得到。

记录能量 $E(s)=-1/2 s^T W s$。因为 $W$ 对称且没有自连接，一次神经元更新满足$Delta E=-(s_i^"新"-s_i^"旧")h_i<=0$。实际翻转时能量严格下降，因此有限状态空间中的异步更新最终到达固定点。稳定状态未必是正确记忆，收敛与恢复成功应分别判断。

== 测试方法

无噪声时每个数字测试一次；其余等级每个数字随机测试 $100$ 次，每级 $300$ 次，共 $1503$ 次。权重初始化、统计测试和示例绘图各自创建种子为 $42$ 的生成器，使绘图不改变统计测试的随机序列。示例图单独生成，不作为统计表之外的额外成功样本。

评价按最终输出逐像素比较：与原始数字完全相同记为“完整恢复”，等于另一个存储数字记为“错误数字”，与三个模板均不相同记为“未匹配”，图中用 `?` 标注。完整恢复率定义为：

$
    "完整恢复率" = "恢复为原始数字的样本数" / "测试样本总数" times 100%.
$

这是对已存储模板的噪声恢复测试，不是对未见过的手写数字进行泛化测试。手册中的相近目标向量通过网络反馈恢复；该动力学并不保证任意输入都到达全局汉明距离最近的模板。因此保留真实网络输出，将误识别和未匹配均计为失败。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))


```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let results = csv("../output/results.csv").slice(1)
#let examples = csv("../output/examples.csv").slice(1)
#let pct(n, d) = eval(mode: "math", str(calc.round(100 * float(n) / float(d), digits: 1)) + "%")

=== 不同噪声强度的恢复结果

#figure(
    table(
        columns: (auto, auto, auto, auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*翻转数*], [*噪声比例*], [*样本数*], [*完整恢复*], [*错误数字*], [*未匹配*], [*恢复率*]),
        ..results
            .map(row => (
                row.at(0),
                pct(row.at(0), 30),
                row.at(1),
                row.at(2),
                row.at(3),
                row.at(4),
                pct(row.at(2), row.at(1)),
            ))
            .flatten(),
    ),
    caption: [直接读取 `results.csv` 的测试统计。后三类计数之和等于样本数。],
) <scores>

三个原始数字均保持稳定。本次每个点阵的最小对齐局部场的总体最小值为$min_(mu,i) x_i^mu (W x^mu)_i approx 0.4299 > 0$，说明三个点阵都是严格稳定状态。随机测试中，单像素与双像素翻转全部恢复；三像素翻转时恢复 $299/300$，一例收敛到未存储模式；九像素翻转时恢复 $249/300$，说明较强噪声可能进入其他吸引域。

=== 点阵与能量记录

#figure(
    image("../output/recall.svg"),
    caption: [数字 $0$–$2$ 的一次三像素翻转示例。由上至下为原始、带噪、恢复点阵；黑色为 $1$，白色为 $-1$。],
) <recall>

#figure(
    table(
        columns: (auto, auto, auto),
        inset: 6pt,
        align: center,
        table.header([*原始数字*], [*翻转数*], [*恢复后识别*]),
        ..examples.map(row => (row.at(0), [3], row.at(3))).flatten(),
    ),
    caption: [从 `examples.csv` 读取的示例识别编号；完整输入与输出向量保存在该文件中。],
)

三个图示样本全部恢复正确，但统计表仍包含一次三像素翻转失败。单次成功示例不能代表同一强度下所有噪声位置组合的表现。

#figure(
    image("../output/metrics.svg"),
    caption: [左：随机测试的完整恢复率。右：图示数字 $0$ 的能量轨迹，横轴每步更新一个神经元，包含初始能量及确认稳定的一轮。],
) <metrics>

== 分析和改进：

*调试与验证。* 首先检查点阵为 $3 times 30$ 的双极性数组，以及噪声函数不会修改原始模板；验证翻转 $0$、$1$、$3$、$9$、$30$ 个像素时改变的位置数准确。随后验证权重对称、对角线为零，且与 $X^T X/30$ 的非对角元素之差来自幅度小于 $0.001$ 的初始化扰动。对三个模式分别穷举全部单像素翻转和双像素翻转，共$3 times (30+binom(30, 2))=1395$ 个带噪输入，全部完整恢复；逐次更新能量均不增加，恢复结果再次输入网络后保持不变。

*抗噪能力的范围。* 三对模板的汉明距离分别为 $15$、$11$、$12$，但模板间距不能直接等同于反馈网络的吸引域半径。三像素翻转的某些输入仍可能落到未存储固定点；高噪声下错误数字与未匹配状态均增多。单、双像素穷举验证针对本组三个模板、固定权重和固定更新顺序成立；更高噪声的比例仅为有限随机样本结果。

*可改进方向。* 若进一步研究容量与抗噪性，可比较不同点阵相似性、存储数量和更新顺序对恢复率的影响；持续呈现更多数据时，也需考虑普通 Hebb 累加缺少权重约束的问题。

= 实验小结：

完成点阵列表设计、Hebb 网络创建与学习，以及带噪数字识别。三个原始模式均为稳定记忆，全部单、双像素翻转可恢复；本次随机测试在 $10%$ 和 $30%$ 噪声下的完整恢复率分别为 $99.7%$ 和 $83.0%$。

展示了局部相关更新如何形成联想记忆，也表明能量下降只保证达到稳定状态，不能保证恢复为正确数字。
