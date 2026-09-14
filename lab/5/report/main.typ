#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验五 HMAX 模型实现"
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

- 理解 HMAX 模型中简单细胞的特征选择与复杂细胞的最大值池化，掌握 S1、C1、S2、C2、VTU 五层的计算过程。
- 使用 MNIST 训练集建立模板库并训练分类器，在独立测试集上评价手写数字识别效果。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

+ 下载 MNIST 数据集，保留官方训练集与测试集的划分。
+ 构建包含四方向、十六尺度 Gabor 滤波、相邻尺度和空间最大值池化、随机模板匹配、全局最大值池化与分类器的 HMAX 模型。
+ 从训练集抽取模板，用全部训练集的 C2 特征训练 VTU，在全部测试集上统计总体准确率、各数字召回率和混淆矩阵，记录预测与错误示例。

= 实验步骤：

== 数据与程序结构

MNIST 使用 #link("https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py")[Keras 官方加载器]采用的 `mnist.npz` 文件。

训练集包含 $60000$ 张图像，测试集包含 $10000$ 张图像，均为 $28 times 28$ 的灰度手写数字，标签为 $0$–$9$。像素转换为 `float64` 并除以 $255$，映射到 $[0,1]$。不缩小数据集，不将测试样本用于模板抽取、标准化参数估计或分类器拟合；超参数预先固定，不按测试准确率选择模型。

== S1：多方向、多尺度的简单细胞响应

Gabor 函数：

$
         u & = x cos(theta) + y sin(theta), \
         v & = -x sin(theta) + y cos(theta), \
    G(x,y) & = exp(-(u^2 + gamma^2 v^2)/(2 sigma^2)) cos((2 pi u)/lambda).
$

方向为 $theta in {0 degree,45 degree,90 degree,135 degree}$。十六个尺度的支持边长依次为 $7,9,dots,37$ 像素；对于边长 $d$，设 $sigma=0.3d$、$lambda=2.5sigma$、$gamma=0.5$。每个滤波器在自身 $d times d$ 支持区域内减去均值并除以二范数，得到零均值、单位范数的滤波器 $F$；随后居中补零到 $37 times 37$，便于批量计算。

对每幅图像进行零边界延拓的线性卷积。采用 $64 times 64$ 实数 FFT，是因为 $28+37-1=64$，该尺寸足以避免循环卷积混叠。逆变换后取索引 `18:46` 的中心区域，保持 $28 times 28$ 大小。滤波器关于中心对称，因此此处卷积与使用同一滤波器的互相关等价。

为减小局部亮度幅度的影响，对响应取绝对值，并除以对应 $d times d$ 图像窗口的二范数：

$
    "S1"_(d,theta)(p) = abs((I * F_(d,theta))(p)) /
    max(sqrt(sum_(q in W_d(p)) I(q)^2), 10^(-8)).
$

图像窗口能量用平方图的积分图计算，边界外像素为零。依据柯西不等式，单位范数滤波器的归一化响应应处于 $[0,1]$；实现裁剪浮点舍入造成的微小越界。全黑图响应为零。每张图最终产生 $16 times 4=64$ 张响应图。

#figure(
    image("../output/filters.svg"),
    caption: [展示边长 $7$、$17$、$27$、$37$ 的四组 Gabor 滤波器，每组保留四个方向。所有滤波器均显示在补零后的 $37 times 37$ 画布上；各小图的颜色范围分别对称设置。],
)

== C1：空间与相邻尺度的最大值池化

先在每张 S1 响应图上进行 $8 times 8$ 最大值池化，步长为 $4$，相邻窗口重叠一半。窗口左上角为 $0,4,8,12,16,20$，所以每个方向得到 $6 times 6$ 的响应图；窗口不额外补零，恰好覆盖到第 $27$ 个像素。

然后把十六个尺度按 $(7,9)$、$(11,13)$、$dots$、$(35,37)$ 分成八个子带。在每个子带内，对两个尺度上相同方向、相同位置的池化值再取最大值：

$
    "C1"_(b,theta)(i,j) = max_(d in B_b) max_(0 <= a,c < 8)
    "S1"_(d,theta)(4i+a,4j+c).
$

最大值只跨空间位置和相邻尺度，*不跨方向*。单张图像的 C1 数组形状为 $(8,4,6,6)$，即八个子带各保留四方向响应。池化提高对局部位置及尺度扰动的容忍度，但不保证任意平移、缩放下完全不变。

== 训练模板与 S2：局部特征组合匹配

固定随机种子 $42$，在每类数字的训练图像中无放回抽取 $20$ 张，共 $K=200$ 张。对每张图像计算 C1，均匀随机选取一个子带及一个合法左上角位置，在四方向图上同时截取 $3 times 3$ 区域。一个模板包含 $3 times 3 times 4=36$ 个数，按 $(4,3,3)$ 的轴顺序存储。

模板库只建立一次，之后保持固定；类别标签仅用于让各类模板数相同，不参与模板数值的计算。

S2 将每个模板与八个子带的所有合法 $3 times 3$ 四方向窗口比较。每个子带有 $4 times 4$ 个窗口，先计算欧氏距离的平方，再由高斯核映射为相似度：

$
    "S2"_(b,k)(i,j) = exp(-norm(X_(b,i,j)-P_k)^2/(2 beta^2)), quad beta=1.
$

相同窗口和模板的响应为 $1$，差异越大，响应越低。为批量计算，使用$norm(x-p)^2=norm(x)^2+norm(p)^2-2x^T p$，并将浮点误差导致的负距离裁剪为零。这里执行模板距离比较，不将模板当作普通卷积核。每个模板产生八张响应图，共 $8K=1600$ 张 $4 times 4$ 的响应图。

== C2：每个模板的全局最大响应

$ "C2"_k = max_(b,i,j) "S2"_(b,k)(i,j), quad k=1,dots,K. $

对一个模板在全部子带、全部空间位置的 S2 响应取最大值，最终每张图像得到 $200$ 维向量。保留“某个局部特征组合是否出现”的信息，同时丢弃最佳匹配的具体位置与子带。按每批 $32$ 张图像依次完成 S1–C2，避免保存全部图像的中间响应。

#figure(
    table(
        columns: (auto, 1fr, 1fr),
        inset: 6pt,
        table.header([*层*], [*单张图像输出形状*], [*主要操作*]),
        [输入], [$(28,28)$], [灰度值除以 $255$],
        [S1], [$(16,4,28,28)$], [Gabor 滤波及局部归一化],
        [C1], [$(8,4,6,6)$], [空间及相邻尺度最大值],
        [S2], [$(8,4,4,200)$], [四方向模板的高斯相似度],
        [C2], [$(200,)$], [每个模板的全局最大值],
        [VTU], [$(10,)$ → 一个数字], [线性得分及最大得分类别],
    ),
    caption: [程序中各层的数组形状，均省略批维；S2 的两个 $4$ 表示空间维度。],
)

== VTU：正则化线性分类器

选择带 L2 正则化的多输出最小二乘分类器，用 NumPy 直接求解，无需额外框架。先在全部训练 C2 特征上按维计算均值 $mu$ 和总体标准差 $s$，将标准差下限设为 $10^(-8)$，再构造增广向量：

$ z = [("C2"-mu)/s,1] in RR^201. $

设 $Z in RR^(60000 times 201)$ 为训练设计矩阵，$Y in RR^(60000 times 10)$ 为标签的独热编码，求解：

$
    min_W norm(Z W-Y)_F^2 + alpha sum_(i=1)^200 sum_(j=1)^10 W_(i,j)^2,
    quad alpha=1.
$

最后一行对应截距，不参与正则化。令 $D=op("diag")(1,dots,1,0)$，则

$ (Z^T Z + alpha D) W = Z^T Y. $

使用 `np.linalg.solve` 求线性方程组，不显式求逆。预测时沿用训练阶段的 $mu$、$s$、$W$，取十个线性得分中最大者对应的数字。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let results = csv("../output/results.csv").slice(1)
#let timing = csv("../output/timing.csv").slice(1)
#let pct(value) = eval(mode: "math", str(calc.round(100 * float(value), digits: 2)) + "%")
#let seconds(value) = str(calc.round(float(value), digits: 2)) + " 秒"
#let names = (train: "训练集", test: "测试集")

=== 总体与各类别识别结果

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*数据范围*], [*样本数*], [*正确数*], [*准确率*]),
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
    caption: [从 `results.csv` 读取的总体结果；训练集包含用于建立模板库的 $200$ 张图像。],
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
    caption: [按真实数字分组统计的测试集召回率，分母是该类的全部测试图像数。],
)

总体准确率为预测标签与真实标签相等的样本数除以总样本数。训练集指标反映拟合结果；测试集指标用于评价对官方保留测试样本的识别能力。

本次训练集正确识别 $55575/60000$，测试集正确识别 $9299/10000$，测试准确率为 $92.99%$，错误 $701$ 张。数字 $1$ 的测试召回率最高，为 $1116/1135 approx 98.33%$；数字 $8$ 最低，为 $788/974 approx 80.90%$。

=== 混淆矩阵与图像示例

#figure(
    image("../output/confusion.svg"),
    caption: [完整测试集混淆矩阵：行是真实数字，列是预测数字，单元格为实际样本数；原始计数保存于 `confusion.csv`。],
)

#figure(
    image("../output/examples.svg"),
    caption: [上行固定展示测试集前十张图像；下行按测试集顺序展示最先出现的十个错误。标题为“样本索引：真实数字 → 预测数字”，均包含在总体统计中。],
)

#figure(
    image("../output/layers.svg"),
    caption: [测试集首张图像的分层响应：S1 取尺度 0、方向 0；C1 取子带 0、方向 0；S2 取子带 0、模板 0；C2 展示全部 $200$ 个特征。各响应图使用 $[0,1]$ 的色阶。],
)

单个 S1 或 S2 响应图并不构成最终判别结果；VTU 使用全部模板的 C2 响应。错误示例用于观察失败情况，而非再次调整超参数的数据来源。

混淆矩阵中较多的错误包括 $8 arrow.r 0$ 的 $51$ 张、$5 arrow.r 3$ 的 $46$ 张和 $8 arrow.r 5$ 的 $39$ 张。测试集前十张中，第 $7$ 号图像的真实数字为 $9$，预测为 $4$；固定前十张示例同时保留正确和错误结果。

=== 运行时间

#figure(
    table(
        columns: (1fr, auto),
        inset: 6pt,
        table.header([*阶段*], [*实测时间*]),
        [数据读取、模板抽取、训练特征提取与 VTU 拟合], seconds(timing.at(0).at(1)),
        [模型保存、测试特征提取、预测及结果输出], seconds(timing.at(1).at(1)),
        [合计], seconds(timing.at(2).at(1)),
    ),
    caption: [读取 `timing.csv` 的单次完整运行用时；本次运行使用已下载并校验的数据，时间不含首次下载、开发调试及报告编译。],
)

== 分析和改进：

*数值验证。* 对两张固定随机图像的全部 $64$ 个滤波器，将 FFT S1 输出与独立的空间窗口逐项点积及归一化结果比较，最大绝对误差约为 $1.86 times 10^(-15)$，通过 $10^(-12)$ 的绝对容差检查；另验证滤波器零均值、单位范数及全黑输入的零响应。C1 使用各方向不同的标记值确认没有跨方向取最大值，再将每个子带、方向、位置的输出与直接截取的空间窗口最大值比较。S2 在两组随机 C1 图、三个模板的全部位置上与直接距离计算比较，精确匹配响应为 $1$；C2 与展平空间和子带后的全局最大值一致。这些检查针对卷积裁剪、轴顺序和模板距离等容易出错的环节。

*完整结果核对。* 验证两组 C2 特征分别为 $(60000,200)$、$(10000,200)$，数值有限且位于 $[0,1]$；按同一种子重新抽取模板，内容与保存模型完全相同，来源索引互不重复且每类恰好 $20$ 张。改变提取批量后前十一张测试图像的特征仍一致。加载保存模型重新预测全部测试集，与逐样本 CSV 完全一致；逐类计数与总体指标一致，混淆矩阵总和为 $10000$、对角线之和为 $9299$。VTU 正规方程的相对残差约为 $1.61 times 10^(-15)$。

*结果解释。* Gabor 滤波响应表示不同方向和尺度上的局部笔画，S2 模板表示四方向响应的局部组合。C1 与 C2 的最大值池化减少对匹配位置和尺度的依赖，但全局池化也丢弃了数字笔画之间的相对空间位置；局部结构相近而整体布局不同的数字可能得到相似特征。随机抽取 $200$ 个模板与线性 VTU 的表达能力也有限，因此结果应结合实际混淆矩阵分析，不能解释为已经达到 MNIST 的最优识别水平。

= 实验小结：

完成 MNIST 下载及校验，构建 S1、C1、S2、C2、VTU 五层模型，并使用全部 $60000$ 张训练图像和 $10000$ 张测试图像完成训练与测试，测试准确率为 $92.99%$。

得到可追溯的模板库、分类模型、逐样本测试预测和完整统计结果，展示了 HMAX 如何通过固定滤波、模板组合与交替池化提取用于分类的特征，也体现了池化不变性与空间信息保留之间的取舍。
