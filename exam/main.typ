#import "@preview/ezexam:0.3.1": EXAM, choices, exam-info, question, setup, title

#show: setup.with(
    mode: EXAM,
    resume: false,
    heading-top: 0em,
    heading-bottom: .4em,
    line-height: .65em,
    par-spacing: .65em,
    enum-spacing: .65em,
    list-spacing: .65em,
)
#set par(justify: true)
#let Title = [计科#underline[认知科学与类脑计算]期末考试]
#let author = "arshtyi"
#let date = datetime.today()
#set document(title: Title, date: date, author: author)
#title(Title)
#exam-info(info: (
    班级: "24智能",
    教师: "吴强",
    时间: datetime(year: 2026, month: 12, day: 2).display("[year].[month].[day]"),
    源码: link("https://github.com/arshtyi/SDU-Cognitive-Science-And-Brain-Inspired-Computing", "link"),
))
