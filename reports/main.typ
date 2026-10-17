  // format text
#set text(
  font: "New Computer Modern",
  size: 13.5pt,
  lang: "vi",
)


#set page(
  width: 210mm,
  height: 297mm,
  margin: (x: 1.5cm, y: 1.5cm),
)

// Trang bìa
#include "Chapters/cov.typ"

#show outline.entry.where(level: 1): it => {
  strong(it)
}

// format page
#set page(
  margin: (top: 2.5cm, bottom: 2.5cm, right: 3cm, left: 3cm),
  number-align: center,
  numbering: "1",
  header: [
    #set text(12pt)
    #smallcaps[Nhập môn thị giác máy tính]
    #h(1fr) CS231.Q21
    #line(length: 100%)
  ],
  footer: context [
    #line(length: 100%)
    #smallcaps[Đồ án môn học]
    #h(0.5fr) Trang
    #counter(page).display(
      "1",
    )

  ],
)
#show heading: set text(size: 15pt)

#set page(
  margin: (top: 3cm, bottom: 3cm, right: 2cm, left: 2cm),
)

#show outline.entry.where(level: 1): it => {
  strong(it)
}

#show outline.entry: it => {
  set text(fill: rgb("1a4fff"))
  it
}

#set par(justify: true)

#include "Chapters/pref.typ"
#include "Chapters/member.typ"
#outline()
#pagebreak()

#include "Chapters/chap1.typ"
#include "Chapters/chap2.typ"
#include "Chapters/chap3.typ"
#include "Chapters/chap4.typ"
#include "Chapters/chap5.typ"
#include "Chapters/chap6.typ"
#include "Chapters/ref.typ"








