#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验二 感知器模型实现并进行分类"
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

- 理解感知器的加权求和、偏置和阶跃激活函数，掌握根据预测偏差逐样本更新参数的方法。
- 使用 NumPy 实现单层感知器，分别学习与、或、非运算，并通过完整真值表检验二分类结果。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

+ 基于与、或、非真值表构建训练数据，保存在 Python 列表中。
+ 创建感知器，使用手册给出的权重和偏置更新规则进行训练。
+ 将全部合法布尔输入送入训练好的网络，逐项比较预测与真值，记录参数、训练轮数和分类结果。

= 实验步骤：

== 数据与程序结构

与、或运算各有两个输入，输入顺序均为 $(0,0)$、$(0,1)$、$(1,0)$、$(1,1)$；与运算标签依次为 $0,0,0,1$，或运算标签依次为 $0,1,1,1$。非运算只有一个输入，两个样本为 $0 mapsto 1$、$1 mapsto 0$。

`TRUTH_TABLES` 按运算名称保存三个二维列表，每行最后一项为标签，前面的项为输入。训练时转换成 NumPy 数组：与、或的输入矩阵为 $X in RR^(4 times 2)$，非的输入矩阵为 $X in RR^(2 times 1)$。输入使用浮点数，标签及预测使用取值为 $0$ 或 $1$ 的整数。每种运算独立初始化并训练一个感知器。

== 感知器模型与学习规则

对一个输入向量 $x in RR^d$，权重为 $w in RR^d$，偏置为 $b in RR$，净输入和预测定义为：

$
    z = w^T x + b, quad
    y = cases(1 & quad z >= 0, 0 & quad z < 0).
$

这对应手册中的“输入—加权求和—阶跃函数—输出”网络，没有隐藏层。偏置可以看作固定输入 $1$ 的连接权重，使分类边界不必经过原点。实现明确约定净输入等于零时输出 $1$，训练与测试共用 `predict`，保持判定一致。

对真值标签 $t$，按手册更新参数：

$
    e = t - y, quad
    w <- w + eta e x, quad
    b <- b + eta e.
$

当预测正确时 $e=0$，参数不变；将正类错判成负类时 $e=1$，提高该样本的净输入；将负类错判成正类时 $e=-1$，降低该样本的净输入。这是带教师信号的在线学习，每次更新后的参数立即用于下一个样本。

使用零权重、零偏置初始化，固定学习率 $eta=1$，按列表顺序遍历数据，最多训练 $100$ 轮。输入与更新量均为整数值，可在本次小规模实验中准确表示，零阈值判断不需要另加浮点容差。训练不包含随机过程，使用相同配置可得到相同结果。

例如，与运算的首个样本是 $x=(0,0)$、$t=0$。初始净输入为零，预测为 $1$，因此 $e=-1$，权重仍为 $(0,0)$，偏置从 $0$ 更新为 $-1$。虽然输入全零，偏置仍然能够得到修正。

== 训练流程与测试方法

+ 由当前真值表得到输入维数，创建全零权重向量和零偏置。
+ 每轮将误分类计数清零，顺序遍历全部样本，调用 `predict`，计算 $t-y$ 并立即更新权重和偏置。
+ 保存本轮在线误分类次数；若整轮计数为零，则返回参数及训练记录，否则进入下一轮。
+ 若达到轮数上限仍未满足停止条件，抛出 `RuntimeError`。
+ 训练结束后固定参数，对真值表中的全部输入重新批量预测，记录净输入、预测值、真值及正确样本数。

停止条件使用“整轮没有误分类”，此时整轮参数没有变化，全部样本都由同一组最终参数正确分类。图中每轮的计数是在参数不断更新的过程中得到的，也是该轮实际修正参数的次数；它不同于用轮末参数重新评估得到的错误数，也不要求逐轮单调下降。报告中的训练轮数包含最后一轮无更新的确认过程。

与、或各穷举 $4$ 种输入，非穷举 $2$ 种输入，共 $10$ 项。由于真值表已包含布尔定义域的全部输入，测试复用这些输入来验证逻辑功能，不进行训练集与测试集划分。所得准确率表示布尔输入上的完整正确性，不代表对连续输入或带噪输入的泛化能力。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 程序实验结果：

=== 最终参数与分类准确率

#let models = csv("../output/models.csv").slice(1)
#let results = csv("../output/results.csv").slice(1)
#let pct(n, d) = eval(mode: "math", str(calc.round(100 * float(n) / float(d), digits: 1)) + "%")
#figure(
    table(
        columns: (auto, auto, auto, auto, auto, auto),
        align: center,
        inset: 7pt,
        table.header([*运算*], [*权重 $w$*], [*偏置 $b$*], [*训练轮数*], [*正确/总数*], [*准确率*]),
        ..models
            .map(row => (
                row.at(0),
                row.at(1),
                row.at(2),
                row.at(3),
                row.at(4) + "/" + row.at(5),
                pct(row.at(4), row.at(5)),
            ))
            .flatten(),
    ),
    caption: [从 `models.csv` 读取的训练参数与穷举测试结果。],
) <models>

三种运算均在设定上限内收敛，全部布尔输入预测正确。根据学得的参数，三个模型的判定分别为：

$
    y_"AND" & = cases(1 & quad 2 x_1 + x_2 - 3 >= 0, 0 & quad "其他"), \
     y_"OR" & = cases(1 & quad x_1 + x_2 - 1 >= 0, 0 & quad "其他"), \
    y_"NOT" & = cases(1 & quad -x_1 >= 0, 0 & quad "其他").
$

这些参数由学习过程得到，代码没有直接指定逻辑门的最终权重。与运算两个输入的权重不相等，但在布尔定义域上仍精确实现对称的与运算；这说明满足分类要求的参数并不唯一。

=== 全部输入的输出记录

#figure(
    table(
        columns: (auto, auto, auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*运算*], [*输入*], [*真值 $t$*], [*净输入 $z$*], [*预测 $y$*], [*判定*]),
        ..results
            .map(row => (
                row.at(0),
                row.at(1),
                row.at(2),
                row.at(3),
                row.at(4),
                if row.at(2) == row.at(4) { [正确] } else { [错误] },
            ))
            .flatten(),
    ),
    caption: [从 `results.csv` 读取的全部 $10$ 项测试记录。],
) <results>

与运算的 $(1,1)$、或运算的 $(0,1)$ 和 $(1,0)$、非运算的 $0$ 均恰好得到零净输入，按统一约定输出 $1$。这些边界样本表明，阶跃函数在零点的取值必须在训练、预测及结果解释中保持一致。

=== 训练过程

#figure(
    image("../output/training.svg"),
    caption: [三个感知器每轮在线误分类次数。最后一轮全部为零。],
) <training>

与、或、非分别需要 $6$、$4$、$3$ 轮。轮数差异与真值分布、固定样本顺序和更新轨迹有关，不能由这一次运行推出所有配置下相同的收敛快慢关系。

== 分析和改进：

*线性可分性决定单层模型的适用范围。* 与、或的两类点能被直线分开，非运算的一维样本能被阈值分开，因此本次感知器可以正确学习。对于异或，正类 $(0,1)$、$(1,0)$ 要求 $w_2+b >= 0$、$w_1+b >= 0$，相加得到 $w_1+w_2+2b >= 0$；负类 $(0,0)$、$(1,1)$ 要求 $b<0$、$w_1+w_2+b<0$，相加得到相反的不等式。因此单个线性感知器不能精确实现异或，增加训练轮数无法改变这一限制。

*偏置和阈值约定影响正确性。* 若将偏置固定为初始值 $0$，分类边界就必须经过原点；在本实验的零点输出约定下，全零输入始终预测为 $1$，与、或的全零样本就无法分类正确。学得的模型有多个样本恰好位于边界，说明布尔真值正确并不意味着存在较大的抗扰动余量。

*正确率需要结合测试范围解释。* 穷举结果证明三个模型在所有合法布尔输入上都正确，但没有验证连续域上的行为，也未测试噪声鲁棒性。在线误分类次数只反映当前学习轨迹，不能当作连续下降的损失函数。若后续研究抗扰动性，可以另行检查分类间隔；本实验只实现手册要求的感知器训练与逻辑分类。

= 实验小结：

本实验从列表形式的真值表出发，使用同一套 NumPy 感知器训练与预测函数分别学习与、或、非运算。按手册实现权重与偏置的在线更新，以整轮零误分类作为停止条件，并完成全部 $10$ 个布尔输入的分类测试，三个模型均达到 $100%$ 正确率。

实验说明，简单的加权求和与阶跃函数能够完成线性可分的逻辑分类；偏置、零阈值约定及停止条件是实现时需要保持一致的细节。
