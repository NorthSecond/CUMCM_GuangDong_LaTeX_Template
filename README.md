# CUMCM_GuangDong_LaTeX_Template

满足广东省要求的数学建模国赛 $\LaTeX$ 模板，适配到2022年。

## 编译环境

使用本地 $\TeX$ Live 2022 亲测在 Windows 和 Linux（WSL2）下均可以顺利编译，编译顺序为

$$
XeLaTeX \rightarrow bibTeX \rightarrow xeLaTeX \rightarrow xeLaTeX
$$

## TODO

目前有一个问题没有解决，是在使用过程中节标题 `\section{}` `\subsection{}` `\subsubsection{}` 指令在使用 XeLaTeX 进行编译的过程中会出现warning，目前分析是关于中文的适配性问题？但是实测书签目录也可以正常生成，作者暂未找到一个合适的解决方案，欢迎PR。
