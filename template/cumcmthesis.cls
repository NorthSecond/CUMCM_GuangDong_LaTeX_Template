% author:yangyf83
% date:2022-08-15
% version:1.0
% a part of the repo CUMCM_GuangDong_LaTeX_Template
% https://github.com/NorthSecond/CUMCM_GuangDong_LaTeX_Template
% copyright with MIT Lisence.

% 全国大学生数学建模竞赛广东省论文LaTeX模板
% 目前适配到2022年国赛和广东省的论文格式要求
% 本模板在 https://github.com/latexstudio/CUMCMThesis 
% 和 https://github.com/personqianduixue/CUMCM_LaTeX_Template 
% 的基础上进行了修改，感谢原作者的贡献。

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{cumcmthesis}
              [2022/08/15 v1.0 Standard LaTeX Template for CUMCM GuangDong]

%% Inital Code
\newcommand\mcm@tokens@keywords{}
%% Declaration of Options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

%% Executing of Options
\ExecuteOptions{colorprint}
\ProcessOptions\relax
%% Loading Class

% 正文使用小四字号
\LoadClass[a4paper,12pt]{article}
%% Loading Package
\RequirePackage{ifxetex}
\RequireXeTeX

% 编译工具检查
\ifxetex\else
\ClassError{mcmthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi

% 支持中文的 ctex 宏包
\RequirePackage{ctex}
% 页面布局
\RequirePackage{geometry}
% 数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
%\RequirePackage{titletoc}
% 关于颜色的 xcolor 宏包
\RequirePackage{xcolor}
% 插入图片
\RequirePackage{graphicx}
\RequirePackage{float}
% 表格
\RequirePackage{array}
%% 长表格
\RequirePackage{longtable}
%% booktabs 提供了\toprule 等命令.
\RequirePackage{booktabs,tabularx}
%% multirow 支持在表格中跨行
\RequirePackage{multirow}
%% 调整间隔, 让表格更好看些
\RequirePackage{bigstrut}
%% 在跨行表格中输入定界符
\RequirePackage{bigdelim}
% 保护脆落命令
\RequirePackage{cprotect}
% 设置代码环境
\RequirePackage{listings}
\RequirePackage{xcolor}
% 插入链接
\RequirePackage{url}
% 绘图的包
%\RequirePackage{tikz}
%\usetikzlibrary{snakes}
% 字体与数学字体
\RequirePackage{fontspec}
\RequirePackage{times}
% \RequirePackage{fourier}
%% \usepackage{utopia}
%% \usepackage{fourier}

% 伪代码
\RequirePackage{algorithm}
\RequirePackage{algorithmicx}
\RequirePackage{algpseudocode}
% Use Input in the format of Algorithm
\renewcommand{\algorithmicrequire}{ \textbf{Input:}}     
% UseOutput in the format of Algorithm
\renewcommand{\algorithmicensure}{ \textbf{Output:}}    

% 字图与子标题
\RequirePackage{subcaption}
\RequirePackage[titles]{tocloft}
%\renewcommand{\cftdot}{$\cdot$}
%\renewcommand{\cftsecdotsep}{1.5}
\renewcommand{\cftsecdotsep}{4.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
%\setlength{\cftbeforesubsecskip}{5pt}

% section字体为黑体小四
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\heiti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
% 代码块行内间距
\renewcommand{\cftsecaftersnumb}{\hskip.4em}

% 不使用目录，因此删去所有目录相关的内容

% 关于代码环境的有关颜色的设置
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.95,0.95,0.95}

% 代码 Hightlight 环境的设置
\lstset{
	basicstyle=\small\tt,
	backgroundcolor=\color{backcolour},   
	commentstyle=\color{codegreen},
	keywordstyle=\color{magenta},
	numberstyle=\tiny\color{codegray},
	stringstyle=\color{codepurple},
	breakatwhitespace = false,         
	breaklines=true,                 
	captionpos=b,
	keepspaces=true,                 
	numbers=left,
	numbersep=10pt, 
	showspaces=false,                
	showstringspaces=false,
	showtabs=false, 
	%escapebegin=\begin{CJK*}{GBK}{hei},escapeend=\end{CJK*},
	tabsize=4
}

% 首行缩进
\RequirePackage{indentfirst}
% 设置浮动体的标题
\RequirePackage{caption}
% 定制列表环境
\RequirePackage{enumitem}
% 下划线
\RequirePackage{ulem}
% 尺寸计算
\RequirePackage{calc}

\RequirePackage[titletoc,title]{appendix}

\RequirePackage{etoolbox}
\AtBeginEnvironment{thebibliography}{%
    \phantomsection
    \addcontentsline{toc}{section}{\refname}
    }

% 超链接 hyperref 的设置
\RequirePackage{hyperref}
% 智能自动化交叉引用
\RequirePackage{cleveref}

% 页面布局
% 页边距：四侧均为25mm页边距：
% 本来想设置成国标的，但是发现国标好像不满足广东省论文边距的要求
% 那就卡下限四面一样宽吧
\geometry{top=25mm,bottom=25mm,left=25mm,right=25mm}

% 行距：1.25倍行距
\renewcommand*{\baselinestretch}{1.25}

% 修改tabular 环境, 设置表格中的行间距为正文行间距.
\let\mcm@oldtabular\tabular
\let\mcm@endoldtabular\endtabular
\renewenvironment{tabular}%
	{\bgroup%
	\renewcommand{\arraystretch}{1.38}%
	\mcm@oldtabular}%
	{\mcm@endoldtabular\egroup}

% 每行缩进两个汉字
\setlength\parindent{2em}

% 设置字体
% 可使用 fc-list :lang=zh-cn 命令来获得自己电脑上的中文字体对应的简写
% 参考https://zhuanlan.zhihu.com/p/260989874
% 这里使用 Windows 内部自带的一些字体，
% 在字体相对完整的 Windows 下应该可以不考虑字体自动运行
\setmainfont{Times New Roman}
\setsansfont{Arial}

% 设置等宽字体
\setmonofont{Consolas}

% 中文字体：宋体天下第一[doge]
\setCJKfamilyfont{song}[AutoFakeBold]{SimSun}
\newcommand*{\song}{\CJKfamily{song}}

\renewcommand\normalsize{%
	\@setfontsize\normalsize{12.05}{14.45}
	\abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
	\abovedisplayshortskip \z@ \@plus3\p@
	\belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}

% 数学环境, 定理等设置
\newtheorem{definition}{\mcm@cap@definition}
\newtheorem{theorem}{\mcm@cap@theorem}
\newtheorem{lemma}{\mcm@cap@lemma}
\newtheorem{corollary}{\mcm@cap@corollary}
\newtheorem{assumption}{\mcm@cap@assumption}
\newtheorem{conjecture}{\mcm@cap@conjecture}
\newtheorem{axiom}{\mcm@cap@axiom}
\newtheorem{principle}{\mcm@cap@principle}
\newtheorem{problem}{\mcm@cap@problem}
\newtheorem{example}{\mcm@cap@example}
\newtheorem{proof}{\mcm@cap@proof}
\newtheorem{solution}{\mcm@cap@solution}

% 浮动环境设置
% 默认情况下, \LaTeX{} 要求每页的文字至少占据 20%,否则该页就只单独放置一个浮动环境,
% 而这通常不是我们想要的, 我们将这个要求降低到 5%.
 \renewcommand*{\textfraction}{0.05}
% 有时如果多个浮动环境连续放在一起, \LaTeX{}会将它们分在几个不同页，即使它们可在同一页放得下. 我们可以通过修改 |\topfraction| 和 |\bottomfraction| 分别设置顶端和底端的浮动环境的最大比例.
 \renewcommand*{\topfraction}{0.9}
 \renewcommand*{\bottomfraction}{0.8}
% 有时\LaTeX{}会把一个浮动环境单独放在一页,
% 我们要求这个环境至少要占据 85% 才能单独放在一页.
% 注意:  |\floatpagefraction| 的数值必须小于 |\topfraction|.
 \renewcommand*{\floatpagefraction}{0.85}
% 关于图片 graphicx
% 如果图片没有指定后缀, 依次按下列顺序搜索
% 冷知识：pdf是在latex中插入矢量图亲测支持最好的格式
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
% 设置图表搜索路径, 可以给图表文件夹取如下名字
\graphicspath{{figures/}{figure/}{pictures/}%
  {picture/}{pic/}{pics/}{image/}{images/}}
  
% 图表标题
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour}, % 图的字体, 宋体小四
	position=bottom % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}

% 表格标题
\captionsetup[table]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour}, % 表的字体, 宋体小四
	position=bottom % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}

% 列表环境设置
\setlist{%
	topsep=0.3em, % 列表顶端的垂直空白
	partopsep=0pt, % 列表环境前面紧接着一个空白行时其顶端的额外垂直空白
	itemsep=0ex plus 0.1ex, % 列表项之间的额外垂直空白
	parsep=0pt, % 列表项内的段落之间的垂直空白
	leftmargin=1.5em, % 环境的左边界和列表之间的水平距离
	rightmargin=0em, % 环境的右边界和列表之间的水平距离
	labelsep=0.5em, % 包含标签的盒子与列表项的第一行文本之间的间隔
	labelwidth=2em % 包含标签的盒子的正常宽度；若实际宽度更宽，则使用实际宽度。
}

% 超链接
	 \hypersetup{%
%         xetex,
		 unicode=true, % hyperref 和 xetex 同时使用时不能开启 Unicode 选项.
         pdfstartview=FitH,
         CJKbookmarks=true,
         bookmarksnumbered=true,
         bookmarksopen=true,
         colorlinks, %注释掉此项则交叉引用为彩色边框(将colorlinks和pdfborder同时注释掉)
         pdfborder=001,   %注释掉此项则交叉引用为彩色边框
		 allcolors=black,
		 breaklinks=true}

\pdfstringdefDisableCommands{%
	\def\cftdotfill{ }%
}
\lstnewenvironment{tcode}
  {
 \lstset{
 	language=TeX,
 	backgroundcolor=\color{backcolour},   
 	commentstyle=\color{codegreen},
 	keywordstyle=\color{magenta},
 	numberstyle=\tiny\color{codegray},
 	stringstyle=\color{codepurple},
 	basicstyle=\small\yaheiconsola,
 	breakatwhitespace=false,         
 	breaklines=true,                 
 	captionpos=b,                    
 	keepspaces=true,                 
 	numbers=left,                    
 	numbersep=10pt,                  
 	showspaces=false,                
 	showstringspaces=false,
 	showtabs=false, 
 	%escapebegin=\begin{CJK*}{GBK}{hei},escapeend=\end{CJK*},
 	tabsize=4
 }
  }
  {}
  
%%% crefformat settings
\crefformat{figure}{#2图~#1#3}
\crefrangeformat{figure}{图~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{figure}{图~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{table}{#2表#1#3}
\crefrangeformat{table}{表(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{table}{表~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{equation}{#2~(#1#3)}
\crefrangeformat{equation}{~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{equation}{~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{definition}{#2\mcm@cap@definition~#1#3}
\crefrangeformat{definition}{\mcm@cap@definition~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{definition}{\mcm@cap@definition~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{theorem}{#2\mcm@cap@theorem~#1#3}
\crefrangeformat{theorem}{\mcm@cap@theorem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{theorem}{\mcm@cap@theorem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{lemma}{#2\mcm@cap@lemma~#1#3}
\crefrangeformat{lemma}{\mcm@cap@lemma~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{lemma}{\mcm@cap@lemma~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{corollary}{#2\mcm@cap@corollary~#1#3}
\crefrangeformat{corollary}{\mcm@cap@corollary~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{corollary}{\mcm@cap@corollary~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{assumption}{#2\mcm@cap@assumption~#1#3}
\crefrangeformat{assumption}{\mcm@cap@assumption~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{assumption}{\mcm@cap@assumption~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{conjecture}{#2\mcm@cap@conjecture~#1#3}
\crefrangeformat{conjecture}{\mcm@cap@conjecture~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{conjecture}{\mcm@cap@conjecture~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{axiom}{#2\mcm@cap@axiom~#1#3}
\crefrangeformat{axiom}{\mcm@cap@axiom~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{axiom}{\mcm@cap@axiom~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{principle}{#2\mcm@cap@principle~#1#3}
\crefrangeformat{principle}{\mcm@cap@principle~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{principle}{\mcm@cap@principle~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{problem}{#2\mcm@cap@problem~#1#3}
\crefrangeformat{problem}{\mcm@cap@problem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{problem}{\mcm@cap@problem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{example}{#2\mcm@cap@example~#1#3}
\crefrangeformat{example}{\mcm@cap@example~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{example}{\mcm@cap@example~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{proof}{#2\mcm@cap@proof~#1#3}
\crefrangeformat{proof}{\mcm@cap@proof~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{proof}{\mcm@cap@proof~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{solution}{#2\mcm@cap@solution~#1#3}
\crefrangeformat{solution}{\mcm@cap@solution~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{solution}{\mcm@cap@solution~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Document Markup
% 生成标题
\renewcommand{\maketitle}{\par
\begingroup
	\newpage
	\global\@topnum\z@   % Prevents figures from going at top of page.
	\@maketitle
\endgroup
\global\let\thanks\relax
\global\let\maketitle\relax
\global\let\@maketitle\relax
\global\let\@thanks\@empty
\global\let\@author\@empty
\global\let\@date\@empty
\global\let\@title\@empty
\global\let\title\relax
\global\let\author\relax
\global\let\date\relax
\global\let\and\relax
}

% override the default title page
\def\@maketitle{
  \null
% 设置页码
\setcounter{page}{1}
% 显示标题
{\centering \zihao{-2} \bfseries \@title \par}
\vskip1ex
}

% 中文标题名称设置
\renewcommand\contentsname{\mcm@cap@contentsname}
\renewcommand\listfigurename{\mcm@cap@listfigurename}
\renewcommand\listtablename{\mcm@cap@listtablename}
\renewcommand\refname{\mcm@cap@refname}
\renewcommand\indexname{\mcm@cap@indexname}
\renewcommand\figurename{\mcm@cap@figurename}
\renewcommand\tablename{\mcm@cap@tablename}
\renewcommand\appendixname{\mcm@cap@appendixname}
\renewcommand\abstractname{\mcm@cap@abstractname}

% 中文标题格式设置
% 节标题格式, 居中, 使用\chinese命令修改计数器
\def\@seccntformat#1{\csname the#1\endcsname\ }

\renewcommand\thesection{\chinese{section}、}
\renewcommand\thesubsection{\arabic{section}\thinspace.\thinspace\arabic{subsection}}
\renewcommand\thesubsubsection{\thesubsection\thinspace.\thinspace\arabic{subsubsection}}
% 节标题全部用宋体加粗，根据要求定字号
\renewcommand\section{\@startsection{section}{1}{\z@}%
 	{-3.5ex \@plus -1ex \@minus -.2ex}%
	{2.3ex \@plus.2ex}%
	{\centering\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
	{-3.25ex\@plus -1ex \@minus -.2ex}%
	{1.5ex \@plus .2ex}%
	{\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
	{-3.25ex\@plus -1ex \@minus -.2ex}%
	{1.5ex \@plus .2ex}%
	{\normalfont\normalsize\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
	{3.25ex \@plus1ex \@minus.2ex}%
	{-1em}%
	{\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
	 {3.25ex \@plus1ex \@minus .2ex}%
	 {-1em}%
	 {\normalfont\normalsize\bfseries}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define new environment and command
%% 摘要两个字设置为 4 号.
%% 定义摘要环境
%% 摘要和关键字单独一页。
\renewenvironment{abstract}{%
\if@twocolumn
		\section*{\abstractname}%
	\else
	 \begin{center}%
	{\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
	 \end{center}%
 	 \quotation
	\fi}
{\if@twocolumn\else\endquotation\newpage\null\fi}

% 重定义引用环境
\renewenvironment{quotation}
	{\list{}{\listparindent 2em%
	 \itemindent \listparindent
	 \rightmargin\z@
	 \leftmargin\z@
	 \parsep \z@ \@plus\p@}%
	\item\relax}
	{\endlist}

% 关键字
\newcommand\keywords[1]{%
	\renewcommand{\mcm@tokens@keywords}{#1}
	\par
	\vskip1ex
	{\noindent\zihao{-4}\heiti\mcm@cap@keywordsname：}~{\mcm@tokens@keywords}
	}
% 定义附录

% \renewcommand{\setthesection}{\appendixname\Alph{section}}
% \renewcommand\appendix{\par
	% \setcounter{section}{0}%
	 % \setcounter{subsection}{0}%
	% \gdef\thesection{\appendixname\@Alph\c@section}}
% 重定义参考文献环境
\renewenvironment{thebibliography}[1]
     {\section*{\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
 %%%数学定理相关的常量
\newcommand*{\mcm@cap@definition}{定义}
\newcommand*{\mcm@cap@theorem}{定理}
\newcommand*{\mcm@cap@lemma}{引理}
\newcommand*{\mcm@cap@corollary}{推论}
\newcommand*{\mcm@cap@assumption}{假设}
\newcommand*{\mcm@cap@conjecture}{猜想}
\newcommand*{\mcm@cap@axiom}{公理}
\newcommand*{\mcm@cap@principle}{定律}
\newcommand*{\mcm@cap@problem}{问题}
\newcommand*{\mcm@cap@example}{例}
\newcommand*{\mcm@cap@proof}{证明}
\newcommand*{\mcm@cap@solution}{解}
 %%%  中文标题名称
\newcommand*{\mcm@cap@contentsname}{目录}
\newcommand*{\mcm@cap@listfigurename}{插图清单}
\newcommand*{\mcm@cap@listtablename}{附表清单}
\newcommand*{\mcm@cap@refname}{参考文献}
\newcommand*{\mcm@cap@indexname}{索引}
\newcommand*{\mcm@cap@figurename}{图}
\newcommand*{\mcm@cap@tablename}{表}
\newcommand*{\mcm@cap@appendixname}{附录}
\newcommand*{\mcm@cap@abstractname}{摘$\quad$要}
 %%%  中文关键字
\newcommand*{\mcm@cap@keywordsname}{关键字}

% 上标文献
\newcommand{\upcite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}
\newcommand{\supercite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}
\endinput
