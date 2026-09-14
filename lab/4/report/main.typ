#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验四 LSTM 的实现"
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

- 理解 LSTM 的遗忘门、输入门、输出门与细胞状态的作用，使用 NumPy 实现前向传播及沿时间反向传播。
- 将二进制加法建模为序列学习任务，让网络通过状态传递学习进位，并检验输出整数是否正确。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

+ 随机生成取值为 $0$–$127$ 的整数对，将两个输入与它们的和编码为八位二进制，构成大量训练样本。
+ 创建一个输入维数为 $2$、隐藏维数为 $16$、输出维数为 $1$ 的单层 LSTM，并训练其逐位预测加法结果。
+ 将网络的八位输出还原为整数，测试零输入、无进位、连续进位、最大和及全部合法输入组合，记录实际结果。

= 实验步骤：

== 数据与程序结构

=== 数据生成与划分

一个样本包含 $(a,b,s)$，其中 $a,b in {0,1,dots,127}$，$s=a+b$。输入最多只需七个有效位，补零为八位；结果最大为 $254$，也可由八位无符号整数完整表示。

固定随机种子 $42$，将 $0$–$16383$ 随机排列，并按 $a=floor(r/128)$、$b=r mod 128$ 映射为整数对。这等价于从全部有序整数对中随机无放回抽样，避免大量重复样本。前 $12288$ 组作为训练集，接下来的 $2048$ 组作为验证集，最后 $2048$ 组作为测试集，三者没有相同的有序整数对。

训练集覆盖合法组合的 $75%$，每轮重新打乱训练顺序，$40$ 轮共呈现 $491520$ 组样本。验证集仅用于记录训练过程，固定训练 $40$ 轮，不按验证或测试结果挑选模型；测试集仅在训练结束后评价。划分以有序对为单位，因此 $(a,b)$ 与 $(b,a)$ 可能分属不同集合；独立测试结果表示对未见过的有序输入对的表现。

`encode` 使用移位和按位与将整数转为取值为 $0$ 或 $1$ 的浮点序列。时间步 $t=0,dots,7$ 对应二进制位权 $2^t$，从最低位开始输入：

$ x_t=(a_t,b_t), quad y_t=s_t. $

这样，低位产生的进位可通过网络状态传到高位。例如 $5+3=8$ 时，按通常高位在前的写法，输入为 `00000101`、`00000011`，标签为 `00001000`；送入网络的顺序相反。第八步的两个输入位均为零，用于输出低七位加法产生的最终进位。模型不接收标签、正确进位或上一位的真实和作为输入。

对 $N$ 道题，输入数组形状为 $(8,N,2)$，标签形状为 $(8,N,1)$。输出中的二进制字符串统一采用高位在前的常规表示，保留前导零。

== LSTM 模型与前向传播

每道题开始时，隐藏状态 $h_(-1)$ 和细胞状态 $c_(-1)$ 均置零，不在不同题目或批次之间保留状态。按代码的行向量约定，拼接 $z_t=[x_t,h_(t-1)] in RR^18$，四组仿射变换合并为一次矩阵乘法：

$ [A_(f,t), A_(i,t), A_(g,t), A_(o,t)] = z_t W + b. $

其中 $W in RR^(18 times 64)$，$b in RR^64$，按遗忘门、输入门、候选状态、输出门的顺序切分。它与分开书写的四组权重等价，仅拼接顺序采用输入在前。状态更新为：

$
    f_t & = sigma(A_(f,t)), quad i_t = sigma(A_(i,t)), \
    g_t & = tanh(A_(g,t)), quad o_t = sigma(A_(o,t)), \
    c_t & = f_t dot.op c_(t-1) + i_t dot.op g_t, \
    h_t & = o_t dot.op tanh(c_t).
$

$dot.op$ 表示逐元素乘法。遗忘门控制旧细胞状态保留的比例，输入门控制候选信息的写入，输出门控制当前隐藏状态的输出。网络需要保留与后续运算有关的进位信息；隐状态是学习得到的连续向量，并未被手工指定为某个进位变量。

输出层为 $ell_t=h_t W_y+b_y$、$p_t=sigma(ell_t)$，其中 $W_y in RR^(16 times 1)$、$b_y in RR$。以 $p_t>=0.5$ 判为 $1$，否则判为 $0$；代码等价地判断 $ell_t>=0$。最终整数完全由预测位解码：

$ hat(s) = sum_(t=0)^7 hat(y)_t 2^t. $

总参数量为 $(2+16) times 64+64+16+1=1233$。权重按输入宽度缩放的零均值正态分布初始化，遗忘门偏置初始为 $1$，其余偏置为零；数据排列、参数初始化与训练打乱分别使用种子为 $42$ 的随机生成器。

== 损失函数与训练算法

使用八个时间步及批内所有样本的平均二元交叉熵。对批量大小 $B$：

$
    L = 1/(8 B) sum_(n=1)^B sum_(t=0)^7
    [ln(1+exp(ell_(t,n))) - y_(t,n) ell_(t,n)].
$

实现用 `np.logaddexp(0, logits)` 计算对数项，避免直接指数运算溢出；`sigmoid` 用等价的 $(1+tanh(x/2))/2$ 计算。输出层误差为：

$ (partial L)/(partial ell_(t,n)) = (p_(t,n)-y_(t,n))/(8 B). $

`backward` 从第 $7$ 位逆序到第 $0$ 位，累加输出层误差与后一时刻传回的隐藏状态梯度。令 $overline(h)_t$、$overline(c)_t$ 表示相应状态的累计梯度，则细胞状态分支先执行：

$
    overline(c)_t <- overline(c)_t + overline(h)_t dot.op o_t dot.op (1-tanh^2(c_t)).
$

四组激活前梯度按前向传播的相同顺序计算：

$
    D_(f,t) & = overline(c)_t dot.op c_(t-1) dot.op f_t dot.op (1-f_t), \
    D_(i,t) & = overline(c)_t dot.op g_t dot.op i_t dot.op (1-i_t), \
    D_(g,t) & = overline(c)_t dot.op i_t dot.op (1-g_t^2), \
    D_(o,t) & = overline(h)_t dot.op tanh(c_t) dot.op o_t dot.op (1-o_t).
$

拼接得到 $D_t$，以 $z_t^T D_t$ 累加 $W$ 的梯度，以批内求和累加 $b$ 的梯度。$D_t W^T$ 中对应隐藏状态的部分传回上一时刻，同时沿细胞状态分支传回 $overline(c)_t dot.op f_t$。两个路径都保留，执行完整八步 BPTT，最后丢弃零初始状态的梯度。

每批 $128$ 组，每轮 $96$ 次更新，共 $3840$ 次。使用 Adam，学习率 $0.01$，一阶与二阶矩系数分别为 $0.9$、$0.999$，分母稳定项为 $10^(-8)$，矩估计按全局更新次数进行偏差修正。更新前将所有梯度的全局二范数裁剪至不超过 $1$，防止梯度过大；不改变前向输出或测试结果。每轮结束后使用同一组参数重新评价整个训练集和验证集，第 $0$ 轮表示未训练状态。因此曲线中的损失是轮末损失，并非批内在线损失的平均。

== 测试与输出设计

逐位准确率统计全部 $8N$ 个二进制位的正确比例；整题准确率要求一道题的八个位全部正确，等价于预测整数与真实和相等。测试既记录独立测试集结果，也检查全部 $128 times 128=16384$ 种合法有序组合。后者包括训练数据，是有限定义域内的功能验证，不能代替独立测试集指标。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))


```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let results = csv("../output/results.csv").slice(1)
#let examples = csv("../output/examples.csv").slice(1)
#let history = csv("../output/history.csv").slice(1)
#let pct(value) = eval(mode: "math", str(calc.round(100 * float(value), digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(float(value), digits: 6)))
#let names = (train: "训练集", validation: "验证集", test: "测试集", all: "全部组合")

=== 准确率与损失

#figure(
    table(
        columns: (auto, auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*数据范围*], [*正确 / 总数*], [*逐位准确率*], [*整题准确率*], [*BCE*]),
        ..results
            .map(row => (
                names.at(row.at(0)),
                row.at(2) + " / " + row.at(1),
                pct(row.at(3)),
                pct(row.at(4)),
                decimal(row.at(5)),
            ))
            .flatten(),
    ),
    caption: [从 `results.csv` 读取的最终结果。全部组合包含前三个集合。],
)

独立测试集 $2048$ 道题全部正确；全集 $16384$ 道题、$131072$ 个输出位全部正确。网络在指定的有限输入范围内实现了正确的八位加法，未通过算术规则修补预测。

=== 训练过程

#figure(
    image("../output/training.svg"),
    caption: [左：轮末训练与验证 BCE，纵轴为对数刻度。右：验证集整题准确率。],
)

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*轮数*], [*训练 BCE*], [*验证 BCE*], [*验证整题准确率*]),
        ..history
            .filter(row => (0, 1, 2, 5, 10, 20, 40).contains(int(row.at(0))))
            .map(row => (
                row.at(0),
                decimal(row.at(1)),
                decimal(row.at(2)),
                pct(row.at(3)),
            ))
            .flatten(),
    ),
    caption: [从 `history.csv` 选取的训练记录，包含未训练的第 $0$ 轮。],
)

验证整题准确率由初始的 $0%$ 上升至第一轮的约 $35.74%$，第二轮达到 $100%$。之后继续按固定轮数训练，交叉熵仍降低，表示正确位的预测置信度进一步提高。$0%$ 的初始整题准确率并不意味着每一个二进制位都预测错误；只要八位中有一位错误，整题即判错。

=== 边界与进位示例

#figure(
    table(
        columns: (auto, auto, auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*$a$*], [*$b$*], [*真实和*], [*预测和*], [*八位预测*], [*所属集合*]),
        ..examples
            .map(row => (
                row.at(0),
                row.at(1),
                row.at(2),
                row.at(3),
                text(font: "jetbrains mono", row.at(4)),
                names.at(row.at(5)),
            ))
            .flatten(),
    ),
    caption: [从 `examples.csv` 读取的预设示例。二进制字符串为高位在前，示例已包含在全集统计中。],
)

$0+0$ 检查全零结果，$0+127$ 检查零加数，$5+10$ 检查无进位相加；$15+1$、$63+1$ 与 $127+1$ 检查逐渐加长的连续进位。$127+1=128$ 需要从最低位一直传递进位至第八位，$127+127=254$ 则覆盖最大合法和。

== 分析和改进：

*调试与验证。* 首先检查全部整数对均处于 $0$–$127$、不重复，输入和标签形状正确，八位编码解码后分别等于原整数与真实和。反向传播采用中心有限差分进行核对：在固定初始化和前三个样本上，以扰动量 $10^(-5)$ 对全部 $1233$ 个参数逐个检查。解析梯度与数值梯度的最大绝对误差约为 $1.72 times 10^(-11)$，通过绝对容差 $10^(-8)$、相对容差 $10^(-4)$ 的比较，验证了门导数、跨时间梯度和损失归一化。

*结果解释。* 从低位到高位的输入顺序与进位方向一致，LSTM 状态可以将低位信息传递至后续时间步。验证损失与训练损失接近，独立测试集全部正确；全集穷举进一步确认了本次保存模型在给定范围内的功能。但八步加法实验本身不足以证明模型能处理任意长度的长期依赖，也没有测试 $128$–$255$ 的加数、更长位数或其他随机种子。

= 实验小结：

完成了随机八位二进制训练数据生成、LSTM 创建与训练、加法结果输出三项要求。包含三类门控、细胞状态、八步 BPTT 和 Adam 更新。

固定种子 $42$、训练 $40$ 轮后，独立测试集的 $2048$ 道题、全部合法组合的 $16384$ 道题均预测正确，覆盖零输入、连续进位和最大和。
