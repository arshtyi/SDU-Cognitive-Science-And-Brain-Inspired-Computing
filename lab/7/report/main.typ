#import "@preview/numbly:0.1.0": numbly
#import "@preview/pointless-size:0.1.2": zh, zihao
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let course = "认知科学与类脑计算"
#let author = "彭靖轩"
#let id = "202400130242"
#let class = "24智能"
#let date = datetime.today()
#let title = "实验七 CNN 做手写体识别"
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

- 理解卷积神经网络的局部连接、权值共享、ReLU 激活和最大池化。
- 使用 Python 构建简单 CNN，通过反向传播学习手写数字的特征和分类规则。
- 使用 MNIST 训练集训练，在独立测试集上取得不低于 $90%$ 的识别准确率。

= 实验环境：

- macos: 26.6.2
- else: see `uv.lock`

= 实验内容：

下载 MNIST，构建包含卷积、激活、池化及全连接层的网络。使用官方全部 $60000$ 张训练图像进行监督学习，固定训练 $5$ 轮后，在全部 $10000$ 张测试图像上评价最终模型，检查测试准确率是否达到 $90%$。

= 实验步骤：

== 数据准备与程序组织

数据使用 #link("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")[MNIST 的 `mnist.npz` 文件]。

每张图像为 $28 times 28$ 灰度图，标签为数字 $0$–$9$。图像转换为 `float32`，增加通道维度，得到 $(N,1,28,28)$ 的张量；标签为 `int64`，形状为 $(N,)$。先将像素除以 $255$，再仅由训练集的全部像素估计一个全局均值和总体标准差：

$ mu = 1/M sum_(i=1)^M x_i, quad sigma = sqrt(1/M sum_(i=1)^M (x_i-mu)^2), quad x' = (x-mu)/sigma. $

$mu approx 0.130660$，$sigma approx 0.308108$；训练图像整体标准化后近似为零均值、单位方差。测试集直接使用这两个训练统计量，不重新拟合。这里不对每张图像单独标准化，也不从测试集估计任何参数。

== 卷积神经网络结构

采用两组“卷积—ReLU—最大池化”，再接两层全连接。卷积核为 $5 times 5$，步长 $1$，不填充；最大池化窗口为 $2 times 2$，步长 $2$。以下尺寸省略批次维度：

#figure(
    table(
        columns: (auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*层与设置*], [*输出尺寸*], [*可训练参数量*]),
        [输入], [$1 times 28 times 28$], [$0$],
        [卷积 $1 arrow.r 6$，$5 times 5$ + ReLU], [$6 times 24 times 24$], [$156$],
        [最大池化 $2 times 2$], [$6 times 12 times 12$], [$0$],
        [卷积 $6 arrow.r 16$，$5 times 5$ + ReLU], [$16 times 8 times 8$], [$2416$],
        [最大池化 $2 times 2$], [$16 times 4 times 4$], [$0$],
        [展平], [$256$], [$0$],
        [全连接 $256 arrow.r 64$ + ReLU], [$64$], [$16448$],
        [全连接 $64 arrow.r 10$], [$10$], [$650$],
        [*合计*], [十类得分], [$19670$],
    ),
    caption: [CNN 的层级结构与参数量，包含各层偏置。],
)

卷积通过局部连接和空间位置间的权值共享提取特征。PyTorch 的卷积层实际计算互相关，即不翻转卷积核；

$ Y_(o,i,j) = b_o + sum_(c=0)^(C-1) sum_(u=0)^4 sum_(v=0)^4 W_(o,c,u,v) X_(c,i+u,j+v). $

空间边长按 $floor((H+2P-K)/S)+1$ 计算。ReLU 为 $max(0, x)$，引入非线性；最大池化取局部窗口最大值，逐步压缩空间尺寸。第一层学习局部笔画响应，第二层进一步组合局部特征，展平后由全连接层输出十个类别的 logits。这里没有手工设置笔画模板，卷积和全连接参数都参与学习。网络构建和自动微分用法参考 #link("https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html")[PyTorch 官方神经网络教程]。

== 训练算法与测试流程

对批次内第 $n$ 个样本，设十类输出为 $z_(n,k)$，真实类别为 $y_n$，使用平均交叉熵：

$ L = 1/B sum_(n=1)^B (log(sum_(k=0)^9 exp(z_(n,k))) - z_(n,y_n)). $

`CrossEntropyLoss` 接收未经 softmax 的 logits 和整数标签，内部采用数值稳定的计算；因此输出层不再附加 softmax。预测为 $hat(y)_n = arg max_k z_(n,k)$。

超参数在测试前固定：随机种子 $42$，批次大小 $128$，训练轮数 $5$，Adam 学习率 $0.001$；Adam 的其余参数采用默认值，其中 $beta_1=0.9$、$beta_2=0.999$、$epsilon=10^(-8)$，不使用权重衰减。训练使用 CPU 并开启确定性算法。算法步骤为：

+ 初始化 CNN，另用固定种子的随机生成器产生每轮训练样本的随机排列。
+ 将全部训练样本分批；每轮 $469$ 批，最后一批为 $96$ 张，不丢弃尾批。
+ 每批依次清空旧梯度、前向计算、计算交叉熵、调用 `backward()` 反向传播，再由 Adam 更新参数。
+ 将各批次更新前的损失按样本数加权、正确数累加，保存为该轮的在线训练指标。
+ 完成全部五轮后，调用 `model.eval()`，在 `inference_mode()` 下分别评估完整训练集、测试集；预测保留原数据顺序。

测试集不参与参数更新、模型选择或提前停止。这里没有划出验证集，因为网络和训练预算预先固定；报告只评价第五轮结束后的模型。保存 `state_dict`、训练像素均值和标准差等信息到 `model.pt`，重建同一结构并加载权重即可复现推理。

== 实验程序：

#raw(block: true, lang: "python", read("../main.py"))

```sh
uv sync
uv run main.py
```

== 程序实验结果：

#let results = csv("../output/results.csv").slice(1)
#let history = csv("../output/history.csv").slice(1)
#let losses = csv("../output/loss.csv").slice(1)
#let timing = csv("../output/timing.csv").slice(1)
#let names = (train: "训练集", test: "测试集")
#let stages = (
    prepare: "数据准备与初始化",
    training: "训练",
    evaluation: "两组数据评估",
    output: "保存与绘图",
    total: "总计",
)
#let pct(value) = eval(mode: "math", str(calc.round(100 * float(value), digits: 2)) + "%")
#let decimal(value) = eval(mode: "math", str(calc.round(float(value), digits: 6)))
#let seconds(value) = str(calc.round(float(value), digits: 2)) + " 秒"

=== 训练过程

#figure(
    table(
        columns: (auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*轮次*], [*在线训练交叉熵*], [*在线训练准确率*]),
        ..history.map(row => (str(int(float(row.at(0)))), decimal(row.at(1)), pct(row.at(2)))).flatten(),
    ),
    caption: [从 `history.csv` 读取五轮训练记录。],
)

#figure(
    image("../output/training.svg"),
    caption: [每轮在线训练交叉熵与准确率。],
)

在线训练交叉熵由 $0.325085$ 降至 $0.040839$，准确率由 $90.68%$ 提高至 $98.695%$。这些指标混合了一轮中不同更新时刻的网络状态，不等于轮末固定模型重新评估的结果，也不是验证集或测试集曲线。

=== 最终模型识别结果

#figure(
    table(
        columns: (auto, auto, auto, auto, auto),
        align: center,
        inset: 6pt,
        table.header([*数据集*], [*样本数*], [*正确数*], [*准确率*], [*平均交叉熵*]),
        ..results
            .filter(row => row.at(1) == "all")
            .map(row => (
                names.at(row.at(0)),
                row.at(2),
                row.at(3),
                pct(row.at(4)),
                decimal(losses.filter(loss => loss.at(0) == row.at(0)).first().at(1)),
            ))
            .flatten(),
    ),
    caption: [最终模型在完整训练集与测试集上的结果，从 `results.csv` 和 `loss.csv` 读取。],
)

训练集正确 $59483/60000$，准确率约为 $99.14%$；测试集正确 $9885/10000$，准确率为 *$98.85%$*，错误 $115$ 张，超过 $90%$ 的要求 $8.85$ 个百分点。最终训练与测试交叉熵分别约为 $0.028578$、$0.036526$。

#figure(
    table(
        columns: (auto, auto, auto, auto),
        align: center,
        inset: 5pt,
        table.header([*真实数字*], [*测试样本数*], [*正确数*], [*召回率*]),
        ..results
            .filter(row => row.at(0) == "test" and row.at(1) != "all")
            .map(row => (row.at(1), row.at(2), row.at(3), pct(row.at(4))))
            .flatten(),
    ),
    caption: [每类召回率的分母为该真实数字的全部测试样本数。],
)

数字 $1$ 的召回率最高，为 $1130/1135 approx 99.56%$；数字 $9$ 最低，为 $990/1009 approx 98.12%$。各数字的召回率均高于 $98%$。

#figure(
    image("../output/confusion.svg"),
    caption: [测试集混淆矩阵：行是真实数字，列是预测数字，单元格为样本数。],
)

#figure(
    image("../output/examples.svg"),
    caption: [上行固定展示测试集前十张图像，下行按原顺序展示前十个错误；标题为从零开始的索引、真实标签和预测标签。],
)

测试集前十张图像均识别正确。第一个错误是索引 $62$ 的数字 $9$ 被预测为 $5$。较多的混淆包括 $4 arrow.r 9$、$5 arrow.r 3$ 各 $7$ 张，以及 $7 arrow.r 2$、$9 arrow.r 5$、$6 arrow.r 5$ 各 $6$ 张。笔画形状、闭合程度和书写差异可能造成这些混淆，但单凭示例不能确定每次错误的原因。

== 分析和改进：

*调试与检查。* 首先核对两次卷积和池化后的尺寸，确保全连接输入为 $16 times 4 times 4=256$；输出保留 logits，标签使用整数类别，避免重复 softmax。检查训练时每批先清空梯度，并按样本数统计尾批；测试阶段关闭梯度记录。实际完成以下验证：

+ 检查 MNIST 两组图像的形状、数值范围、数据类型和十类标签；标准化后的训练像素均值与 $0$、总体标准差与 $1$ 的偏差均小于 $10^(-6)$。
+ 用三张图像逐层前向传播，核对所有中间张量尺寸及总参数量 $19670$。对初始化网络的一个 $128$ 张样本批次反向传播，各层参数梯度均有限且非零；与同种子初始化比较，训练后所有参数张量都发生了更新。
+ 重新加载 `model.pt`，仅使用保存的均值和标准差，对全部 $70000$ 张图像重新推理；总体和各类统计均与 `results.csv` 一致，两组平均交叉熵与记录的差异小于 $10^(-12)$。
+ 逐项核对全部 $10000$ 个测试预测和重算的混淆矩阵。矩阵元素总和为 $10000$，对角线之和为 $9885$，与测试正确数一致。

*结果分析。* 这个较小的网络通过局部连接与参数共享提取数字特征，卷积参数与分类器在同一交叉熵目标下联合学习，能够完成本次十类识别任务。训练与测试准确率相差约 $0.29$ 个百分点，本次划分上泛化差距较小，但这不保证在其他来源的手写图像上也有相同表现。卷积具有平移等变的结构特点，池化可以降低对小幅位移的敏感性；由于边界、下采样和全连接层的影响，本网络并不具有严格的任意平移不变性。

*可改进之处。* 若进一步调整模型，应先从训练集划分验证集，再比较训练轮数、卷积通道数和学习率；也可在训练集上采用小幅平移、旋转等数据增强。最终测试集应继续留作固定模型的评价。

= 实验小结：

完成了：提供 MNIST 下载与校验流程；构建可训练的卷积神经网络；使用官方全部训练集训练，并在全部测试集上测试。固定五轮训练后的测试准确率为 $98.85%$，达到 $90%$ 以上的要求。
