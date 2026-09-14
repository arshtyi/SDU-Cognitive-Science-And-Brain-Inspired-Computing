#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验八 BP 算法分析与实现"
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

- 理解多层前馈网络中信号正向传播、误差反向传播与梯度下降的关系。
- 使用 NumPy 手工实现链式求导，计算各层权重和偏置的更新量。
- 对给定的输入和目标进行训练，使输出的平方和误差严格小于 $0.01$，并验证梯度与结果的正确性。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

输入为 `[[0.35], [0.9]]`，真实值为 `[[0.5]]`。将输入解释为*一个含两个特征的样本*，列向量形状为 $(2,1)$，目标形状为 $(1,1)$。创建一个简单人工神经网络，以平方和误差为目标函数，使用链式法则求出 $Delta w$，通过梯度下降使误差小于 $0.01$。

= 实验步骤：

== 网络结构与程序组织

网络采用 $2 arrow.r 2 arrow.r 1$ 的全连接结构：输入层有两个特征，隐含层有两个 sigmoid 神经元，输出层有一个 sigmoid 神经元。输入层不执行激活变换。两层连接之间无反馈。

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*参数*], [*形状*], [*作用*], [*标量个数*]),
        [$W_1$], [$2 times 2$], [输入到隐含层的权重], [4],
        [$b_1$], [$2 times 1$], [隐含层偏置], [2],
        [$W_2$], [$1 times 2$], [隐含层到输出层的权重], [2],
        [$b_2$], [$1 times 1$], [输出层偏置], [1],
        [合计], [—], [所有权重和偏置均参与学习], [9],
    ),
    caption: [参数矩阵的行对应接收神经元，列对应发送神经元。],
)

固定随机种子 $42$，权重用 `default_rng` 从 $[0,1)$ 均匀分布初始化，偏置初始化为零。随机初始化使两个隐含神经元具有不同的初始参数。

== 前向传播与误差定义

设输入 $x=(0.35,0.9)^T$，目标 $t=0.5$。定义 sigmoid 激活及其导数：

$ sigma(z) = 1/(1+exp(-z)), quad sigma'(z) = sigma(z)(1-sigma(z)). $

采用等价形式 $sigma(z)=(1+tanh(z/2))/2$，避免在 $z$ 为大负数时直接计算 $exp(-z)$ 产生溢出。前向传播为

$ z_1 = W_1 x+b_1, quad h=sigma(z_1), quad z_2=W_2 h+b_2, quad hat(y)=sigma(z_2). $

误差严格采用平方和，定义为

$ E=sum_(k=1)^1 (hat(y)_k-t_k)^2=(hat(y)-0.5)^2. $

这里不乘 $1/2$，因此求导时必须保留系数 $2$。只有一个输出，平方和在数值上等于均方误差，但代码仍显式使用 `sum`。

初始权重和偏置为（下列数值保留六位小数，实际运算使用 `float64`）：

$ W_1=mat(0.773956, 0.438878; 0.858598, 0.697368), quad b_1=mat(0; 0), $
$ W_2=mat(0.094177, 0.975622), quad b_2=mat(0). $

首次前向传播得到

$ z_1 approx mat(0.665875; 0.928141), quad h approx mat(0.660579; 0.716698), $
$ z_2 approx 0.761438, quad hat(y) approx 0.681665867, quad E approx 0.033002487. $

== 链式求导与参数更新

先在输出层计算损失对加权输入的导数：

$
    delta_2 = (partial E)/(partial z_2)
    = (partial E)/(partial hat(y)) (partial hat(y))/(partial z_2)
    = 2(hat(y)-t) hat(y)(1-hat(y)).
$

由 $z_2=W_2h+b_2$，得到输出层梯度

$ (partial E)/(partial W_2)=delta_2 h^T, quad (partial E)/(partial b_2)=delta_2. $

将误差信号传回隐含层，其中 $dot.o$ 表示逐元素乘法：

$ delta_1=(W_2^T delta_2) dot.o h dot.o (1-h), $
$ (partial E)/(partial W_1)=delta_1 x^T, quad (partial E)/(partial b_1)=delta_1. $

例如输入特征 $x_i$ 到隐含神经元 $j$ 的权重梯度为

$
    (partial E)/(partial (W_1)_(j i))
    = 2(hat(y)-t) hat(y)(1-hat(y)) (W_2)_(1 j) h_j(1-h_j)x_i.
$

该式依次连接平方误差、输出 sigmoid、输出层线性组合、隐含层 sigmoid 和输入线性组合的导数。令学习率 $eta=0.5$，对任意参数 $theta$ 使用

$ Delta theta=-eta (partial E)/(partial theta), quad theta_"new"=theta_"old"+Delta theta. $

梯度下降的负号使更新沿误差减小的局部方向移动。*所有梯度都由同一轮的旧参数计算完成后，再统一更新*，特别是计算 $delta_1$ 时不能使用已经更新过的 $W_2$。程序中的 `delta1`、`delta2` 表示误差信号，实际参数变化量则是 $-eta$ 乘相应梯度，两者含义不同。

首轮有 $delta_2 approx 0.078842083$、$delta_1 approx (0.001664823,0.015618013)^T$。例如第一条输入连接的梯度为 $0.001664823 times 0.35 approx 0.000582688$，更新量为 $-0.000291344$，权重由 $0.773956049$ 变成 $0.773664705$。

== 训练与检查流程

训练伪码如下，$k$ 表示已经完成的参数更新次数：

```text
初始化权重、偏置；核对初始梯度
对 k = 0, 1, ..., 1000：
    前向传播，计算并记录输出和平方和误差 E
    若 E 非有限：报错
    若 E < 0.01：返回训练记录
    若 k < 1000：
        使用旧参数计算全部链式梯度
        每个参数减去学习率乘其梯度
达到更新上限仍未满足阈值：报错
```

第 $0$ 行记录初始化状态；第 $k$ 行记录完成 $k$ 次更新后的状态。这样最终记录与保存模型一致，也避免更新前损失与更新后参数混用。阈值使用严格小于；等于 $0.01$ 时继续训练。

数值梯度仅用于验证手写梯度，训练更新始终使用 `backward` 的结果。对九个标量参数分别计算中心差分

$ g_"num"(theta)=(E(theta+epsilon)-E(theta-epsilon))/(2epsilon), quad epsilon=10^(-6). $

扰动一个参数时其余参数保持不变，计算后恢复原值。逐项要求

$ abs(g_"BP"-g_"num") <= 10^(-8)+10^(-5) abs(g_"num"). $

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let history = csv("../output/history.csv").slice(1)
#let first-step = csv("../output/first_step.csv").slice(1)
#let result = csv("../output/results.csv").at(1)
#let decimal(value, digits: 9) = eval(mode: "math", str(calc.round(float(value), digits: digits)))

=== 首轮参数更新

#figure(
    table(
        columns: (auto, auto, auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*参数（零起始下标）*], [*初始值*], [*链式梯度*], [*更新量*], [*更新后*]),
        ..first-step
            .map(row => (
                row.at(0) + "[" + row.at(1) + "," + row.at(2) + "]",
                decimal(row.at(3), digits: 6),
                decimal(row.at(4), digits: 6),
                decimal(row.at(6), digits: 6),
                decimal(row.at(7), digits: 6),
            ))
            .flatten(),
    ),
    caption: [从 `first_step.csv` 读取所有九个参数的首轮计算，表中保留六位小数。],
)

九个链式梯度均通过中心差分检查，最大绝对偏差约为 $2.023 times 10^(-11)$，远小于检查容差。首轮更新后的输出为 #decimal(history.at(1).at(1))，平方和误差为 #decimal(history.at(1).at(2))，比初始化时更小。

=== 训练记录与停止结果

#figure(
    table(
        columns: (auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*已完成更新次数*], [*网络输出*], [*平方和误差*]),
        ..history
            .map(row => (
                str(int(float(row.at(0)))),
                decimal(row.at(1)),
                decimal(row.at(2)),
            ))
            .flatten(),
    ),
    caption: [从 `history.csv` 读取全部记录，含初始化状态。],
)

#figure(
    image("../output/training.svg"),
    caption: [左图显示输出向目标靠近，右图显示平方和误差及停止阈值。],
)

完成 *#result.at(0) 次更新*后，网络输出为 *#decimal(result.at(2))*，平方和误差为 *#decimal(result.at(3))*，严格小于要求的 $0.01$。第 $5$ 次更新后的误差仍为 $0.010879259$，第 $6$ 次是首次满足停止条件的状态。

停止标准是平方和误差，因此 $E<0.01$ 对应 $abs(hat(y)-0.5)<0.1$。

== 分析和改进：

*调试与验证。* 实现时重点核对列向量方向、平方误差导数中的系数 $2$、隐含层误差的传递方向及同步更新顺序。另完成以下独立验证：

+ 检查初始梯度验证前后的参数逐元素一致，确保数值扰动没有改变训练起点。
+ 对给定输入、全零输入、正负混合输入等四组输入与目标，采用另一固定种子生成权重和偏置，检查共 $36$ 个标量导数。中心差分与反向传播最大绝对偏差约为 $9.99 times 10^(-11)$；隐含层及输出的形状分别为 $(2,1)$、$(1,1)$。
+ 检查目标已达到时零次更新返回；误差恰等于阈值时不通过；更新预算耗尽仍未达标时显式报错。另检查 sigmoid 在 $-1000,0,1000$ 输入下输出有限。

= 实验小结：

完成了指定输入、平方和误差、人工神经网络和链式求导更新的全部要求。用 NumPy 实现 $2 arrow.r 2 arrow.r 1$ 网络，在固定初值与学习率下经过 #result.at(0) 次更新，误差由 #decimal(history.first().at(2)) 降至 #decimal(result.at(3))，达到 $E<0.01$。通过逐参数梯度检查、训练复现、模型重载和停止边界检查，验证了代码与报告结果的一致性。
