#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验六 TopoICA 模型实现"
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

- 理解 TopoICA 对普通 ICA 独立性假设的放宽，区分成分值的线性相关与成分能量的高阶相关。
- 掌握 PCA 白化、拓扑邻域构造、近似似然的梯度学习和正交化，使用 Python 实现一个简单的 TopoICA 模型。
- 使用 MNIST 官方训练集学习特征，在独立测试集上评价特征的分类效果和拓扑组织情况。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

下载 MNIST，构建 TopoICA 模型，用全部 $60000$ 张训练图像学习模型，用全部 $10000$ 张测试图像测试。

TopoICA 本身是无监督特征模型，不直接输出数字标签。为得到明确的手写数字测试结果，在学到的特征上拟合一个带 L2 正则化的线性分类器。标签仅用于分类器训练和结果统计，不参与 PCA 与 TopoICA 的拟合。

= 实验步骤：

== 数据准备与程序组织

下载地址为 #link("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")[MNIST 的 `mnist.npz` 文件]。每张 $28 times 28$ 灰度图按行展开为 $784$ 维向量，以 `float64` 存储，像素除以 $255$。保持官方训练、测试划分，不抽取缩小版数据集。

== PCA 降维与白化

设训练图像矩阵为 $X in RR^(60000 times 784)$。仅用训练集估计每个像素的均值 $mu$，计算总体协方差：

$ X_c = X - mu, quad C = 1/N X_c^T X_c. $

通过对称矩阵特征分解，取特征值最大的 $d=100$ 个方向。记相应特征向量为 $U_d$，特征值对角矩阵为 $Lambda_d$，则

$ V = U_d Lambda_d^(-1/2), quad Z = X_c V, quad D = Lambda_d^(1/2) U_d^T. $

`whitening` 为 $V in RR^(784 times 100)$，`dewhitening` 为 $D in RR^(100 times 784)$，满足 $D V = I_d$。白化后的训练数据近似满足 $Z^T Z/N=I_d$。保留的特征值若小于等于 $10^(-8)$，报错，避免在无有效方差的方向上除零。

本次前 $100$ 个主成分保留训练图像总方差的 $91.4629%$。降维既减少计算量，也避开 MNIST 边缘恒为零的像素导致的秩亏。这里没有对每幅图像单独减去亮度均值。测试数据只计算 $(X_"test"-mu)V$，不重新估计均值、特征向量或缩放系数。

== TopoICA 的拓扑与学习目标

=== 从独立成分到邻域能量

将 $100$ 个成分按行排列在 $10 times 10$ 网格上。网格上下、左右首尾相接，形成环面；每个单元的邻域为含自身的 $3 times 3$ 方形区域。对每一坐标轴采用循环距离 $min(abs(a-b), 10-abs(a-b))$，两个轴的距离都不大于 $1$ 时，令 $H_(k j)=1/9$，否则为 $0$。

$H$ 固定、对称，每行恰有九个非零元素，行和为 $1$。例如左上角单元 $0$ 的邻居索引为 $0,1,9,10,11,19,90,91,99$。这个网格描述的是*成分之间*的拓扑关系，与输入图像的像素网格不同。

记白化空间中的分离矩阵为 $W in RR^(100 times 100)$，每一行是一个分离向量。线性响应及邻域幅度为

$ S = Z W^T, quad E_(t k) = sum_(j=1)^d H_(k j) S_(t j)^2, quad P_(t k)=sqrt(E_(t k)+epsilon), $

其中 $epsilon=10^(-8)$。平方、邻域加权和、开平方依次对应能量计算、局部汇集与幅度响应。邻域在优化过程中直接参与计算，因此这里的拓扑组织来自学习过程。

=== 近似似然、梯度及正交化

选择平滑的 $G(e)=-sqrt(e+epsilon)$，并对白化后的分离矩阵施加 $W W^T=I_d$，此时 $abs(det W)=1$，行列式项恒为零。于是最大化近似对数似然等价于最小化

$ J(W) = 1/N sum_(t=1)^N sum_(k=1)^d sqrt(epsilon+sum_(j=1)^d H_(k j) S_(t j)^2). $

PCA 白化、正交约束以及平滑平方根非线性的选择，也可参见 #link("https://redwood.berkeley.edu/wp-content/uploads/2020/08/hyvarinen-hoyer01.pdf")[Hyvärinen、Hoyer 与 Inki（2001）的原始论文]第 3–4 节。这里优化的是近似模型的目标，不将数值 $J$ 称为归一化概率或分类损失。

用 $P^(-1)$ 表示逐元素倒数，$dot.o$ 表示逐元素乘法，则链式法则给出

$ nabla_W J = 1/N (S dot.o (P^(-1) H))^T Z. $

平方的导数系数 $2$ 与开平方的导数系数 $1/2$ 相消。程序先用 `sources**2 @ neighborhood.T` 得到各邻域能量，再用 `(1 / amplitudes) @ neighborhood` 将梯度传回各成分；这两次邻域运算的方向不同。

固定随机种子 $42$，以标准正态矩阵经正交化得到初始 $W$。使用全部 $60000$ 张训练图像计算每一步梯度，学习率 $eta=0.5$，固定执行 $200$ 次更新：

$ W' = W - eta nabla_W J, quad W' = U Sigma V^T, quad W <- U V^T. $

最后一步为 SVD 极分解，等价于 $(W' W'^T)^(-1/2)W'$。它保持各成分单位方差和互不相关，避免只最小化幅度时出现 $W=0$ 的退化解。保存更新前与每次更新后的目标值，共 $201$ 条记录。训练次数预先固定，不根据测试准确率选择模型，也不声称已经找到全局最优解。

若把邻域设为单位矩阵，目标退化为各成分平滑绝对值之和，成分间不再有邻域耦合。此退化情形用于检查实现，不作为另行训练的对照模型。

=== 输入空间中的滤波器与重建含义

因为 $S=(X-mu)V W^T$，所以对应原图像的逆滤波器矩阵为 $F=W V^T in RR^(100 times 784)$。把每行变回 $28 times 28$ 图像后，按训练时的拓扑顺序排列，即得到后面的滤波器图。没有按类别或视觉相似性重新排序。

生成方向可写为 $hat(X)=mu+S W D$。由于保留了全部 $100$ 个白化成分且 $W$ 正交，这个重建等价于投影回 PCA 保留子空间，不能恢复已丢弃的 $684$ 个方向。因此不能用该重建误差证明 TopoICA 比同维 PCA 更好，也不能声称学到了原始 $784$ 维空间的可逆分解。

== 分类器与测试方法

每张图像的分类特征为线性响应和邻域幅度的拼接：$f=[s_1,dots,s_100,p_1,dots,p_100] in RR^200$。前半部分保留响应的正负号，后半部分提供依赖邻域能量的非线性信息。

只在训练特征上按维估计均值 $m$ 和总体标准差 $v$，将标准差下限设为 $10^(-8)$。标准化后加一个常数 $1$ 作为截距，形成 $Q in RR^(60000 times 201)$。标签转换为十类独热矩阵 $Y$，拟合

$ min_B norm(Q B-Y)_F^2 + alpha sum_(i=1)^200 sum_(j=1)^10 B_(i j)^2, quad alpha=1. $

通过 `np.linalg.solve` 求解 $(Q^T Q + alpha R)B=Q^T Y$，其中 $R=op("diag")(1,dots,1,0)$，截距不参与惩罚。预测取十个线性得分中的最大值所对应的数字。这些得分不是类别概率。

模型保存像素均值、白化及逆白化矩阵、初始及最终分离矩阵、邻域矩阵、特征标准化参数与分类器权重。测试时固定全部参数，记录总体准确率、每类召回率、混淆矩阵和逐样本预测；另测量测试集上的未池化成分能量相关性。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let results = csv("../output/results.csv").slice(1)
#let topology = csv("../output/topology.csv").slice(1)
#let timing = csv("../output/timing.csv").slice(1)
#let pct(value) = eval(mode: "math", str(calc.round(100 * float(value), digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(float(value), digits: 6)))
#let seconds(value) = str(calc.round(float(value), digits: 2)) + " 秒"
#let names = (train: "训练集", test: "测试集")
#let stages = (initial: "初始化", trained: "训练后")

=== 训练目标与学习到的滤波器

#figure(
    image("../output/training.svg"),
    caption: [完整训练集上的 TopoICA 目标，每步使用全部训练图像，记录从初始化到第 $200$ 次更新的结果。],
)

目标由 $96.066897$ 降至 $88.658623$，本次运行的每一步均下降。后期下降幅度变小，但最后十步仍下降约 $0.0411$，所以本实验按预设预算结束，没有把它解释为严格收敛。没有使用测试样本作提前停止判断。

#figure(
    image("../output/filters.svg"),
    caption: [训练后 $100$ 个输入空间滤波器，按 $10 times 10$ 拓扑网格排列。每个小图采用自身最大绝对值确定对称色阶，红蓝表示相反符号；上下、左右边界在模型中相连。],
)

这些滤波器是从 MNIST 整幅手写数字中学到的投影方向。图中可观察其笔画、边缘及位置结构，但图像本身不能证明统计独立或生物学意义上的复杂细胞特性；也没有使用自然图像来复现原论文的类 Gabor 感受野结果。

=== 成分能量的拓扑相关性

为检验学习是否将高阶依赖组织到邻域内，对每对不同成分计算 Pearson 相关系数 $op("corr")(s_i^2,s_j^2)$。这里只使用*未经邻域池化*的平方响应，避免相邻池化窗口共享成分而人为造成高相关。每对只计一次：八邻接且不含自身的近邻共 $400$ 对，其余 $4550$ 对归为非邻近。比较同一初始化和训练后的平均值：

#figure(
    table(
        columns: (auto, auto, auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*数据集*], [*阶段*], [*TopoICA 目标*], [*近邻能量相关*], [*非邻近能量相关*]),
        ..topology
            .map(row => (
                names.at(row.at(0)),
                stages.at(row.at(1)),
                decimal(row.at(2)),
                decimal(row.at(3)),
                decimal(row.at(4)),
            ))
            .flatten(),
    ),
    caption: [直接读取 `topology.csv`。训练集和测试集分别统计，测试时固定模型。],
)

随机初始化时近邻与非邻近的能量相关均约为 $0.04$，差异很小；训练后，测试集近邻平均相关约为 $0.086796$，非邻近约为 $0.001490$。这支持本次训练把较强的能量依赖组织到拓扑近邻的判断。它是平均意义上的证据，不代表每个近邻对都强相关，也不代表非邻近成分严格统计独立；平均相关较小还可能掩盖个别成分对的依赖。

=== 总体与分类别识别结果

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*数据集*], [*样本数*], [*正确数*], [*准确率*]),
        ..results
            .filter(row => row.at(1) == "all")
            .map(row => (
                names.at(row.at(0)),
                row.at(2),
                row.at(3),
                pct(row.at(4)),
            ))
            .flatten(),
    ),
    caption: [训练与测试的总体结果，从 `results.csv` 读取。],
)

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*真实数字*], [*测试样本数*], [*正确数*], [*召回率*]),
        ..results
            .filter(row => row.at(0) == "test" and row.at(1) != "all")
            .map(row => (
                row.at(1),
                row.at(2),
                row.at(3),
                pct(row.at(4)),
            ))
            .flatten(),
    ),
    caption: [各数字的测试召回率，分母为该真实类别的全部测试样本数。],
)

训练集正确 $55284/60000$，准确率为 $92.14%$；测试集正确 $9274/10000$，准确率为 $92.74%$，错误 $726$ 张。数字 $1$ 的测试召回率最高，为 $1115/1135 approx 98.24%$；数字 $5$ 最低，为 $782/892 approx 87.67%$。这里的识别结果属于“TopoICA 特征加线性分类器”的完整流程，不是 TopoICA 自身的无监督目标值。

#figure(
    image("../output/confusion.svg"),
    caption: [完整测试集混淆矩阵，行是真实数字，列是预测数字，单元格为样本数。],
)

#figure(
    image("../output/examples.svg"),
    caption: [上行固定显示测试集前十张图像；下行按数据集顺序显示前十个错误。标题依次为索引、真实数字、预测数字，索引从 $0$ 开始。],
)

本次前十张测试图像均识别正确；第一个错误为索引 $33$ 的数字 $4$ 被预测为 $0$。较多的混淆包括 $5 arrow.r 3$ 的 $42$ 张、$7 arrow.r 9$ 的 $38$ 张、$4 arrow.r 9$ 的 $36$ 张和 $9 arrow.r 4$ 的 $31$ 张。

== 分析和改进：

*调试与数值检查。* 将程序分解为白化、邻域、梯度、正交化和分类器进行检查。重点处理了 MNIST 协方差的秩亏、平方根零点、邻域矩阵的转置方向，以及报告图表与保存模型的一致性。使用以下实际检查：

+ 对固定种子 $123$ 生成的 $23 times 7$ 随机输入、$5 times 7$ 权重和非对称邻域，逐个检查全部 $35$ 个权重的中心有限差分，扰动 $10^(-5)$；解析梯度最大绝对误差约为 $2.29 times 10^(-10)$，小于 $10^(-8)$。特意使用非对称邻域，避免对称性掩盖转置错误。
+ 将邻域幅度与逐样本、逐邻域直接循环求和结果比较；验证全零输入的目标为 $d sqrt(epsilon)$、梯度为零，并验证 $H=I$ 的普通 ICA 特例。
+ 检查每个网格单元恰有九个邻居、行和为 $1$、环面角点的邻居索引正确；对随机矩阵的 SVD 正交化结果检查 $W W^T=I$。
+ 用 $250 times 120$ 随机矩阵检查训练白化均值、协方差和 $D V=I$。完整 MNIST 训练白化协方差相对单位矩阵的最大绝对偏差约为 $4.44 times 10^(-13)$；训练后 $W W^T$ 的最大绝对偏差约为 $1.55 times 10^(-15)$。
+ 检查全部 $201$ 条目标记录、本次目标逐步下降，以及重新计算的最终目标与 CSV 一致。两组特征形状分别为 $(60000,200)$、$(10000,200)$，所有数值有限；将测试样本改为每批 $37$ 张提取后，结果在 $10^(-12)$ 的绝对容差下保持一致。
+ 加载保存模型重新预测训练集与全部测试集，核对总体和各类计数、全部测试预测、混淆矩阵及四组能量相关矩阵。混淆矩阵总和为 $10000$，对角线之和为 $9274$；分类器正规方程的相对残差约为 $9.40 times 10^(-16)$。

*结果分析。* 白化与正交约束消除了训练成分的二阶相关，而平方响应仍可以相关，体现了“不相关”不等于“独立”。相比随机初始化，训练后近邻与非邻近能量相关的差异扩大，并在独立测试样本上保持，符合 TopoICA 的建模目的。

分类仍受 PCA 信息损失、成分数、固定邻域大小及线性分类器表达能力限制。$[s,p]$ 中的 $s$ 在白化空间里只是正交旋转，若仅使用 $s$ 配合线性分类器，不能因此获得新的非线性分类边界；加入 $p$ 才提供邻域能量的非线性信息。相似笔画的数字仍会混淆，不能仅靠测试准确率推断模型已经完整刻画数字结构。

*可改进之处。* 后续可以从训练集划出验证集，用于选择成分数、邻域半径或训练步数，并与普通 ICA、PCA 特征在相同分类器下进行对照；也可研究局部图像块以减少整幅图像的位置依赖。

= 实验小结：

完成了：下载并校验 MNIST；用 NumPy 实现具有固定拓扑邻域、能量耦合目标、解析梯度和正交约束的 TopoICA；使用官方全部训练集训练并在全部测试集上测试。训练与测试的数据处理参数严格分离，最终测试准确率为 $92.74%$。

TopoICA 可以在保持成分线性不相关的同时，将较强的能量相关组织到拓扑近邻，并将这些响应用于手写数字识别。
