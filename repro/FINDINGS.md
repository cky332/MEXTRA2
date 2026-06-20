# MEXTRA 复现报告 —— 失败 case 专项分析

论文: *Unveiling Privacy Risks in LLM Agent Memory* (MEXTRA, arXiv:2502.13172v2, ACL'2025)
被攻击的 agent core / prompt 生成器: **DeepSeek-V3.2-Exp**(SiliconFlow),temperature=0。
论文原文用 GPT-4o 作 agent core、GPT-4 作 prompt 生成器、SBERT(MiniLM) 作嵌入。

> 复现代码: `repro/realrun.py`(EHRAgent)、`repro/rap_realrun.py`(RAP 降级版)、
> `repro/{batch,analyze,summarize,rap_eval}.py`。检索/prompt/评估逻辑逐行照搬原始仓库。

---

## 0. 环境硬约束(决定了"哪里复现不了")

| 约束 | 影响 |
|---|---|
| 只白名单了 `api.siliconflow.cn`,**HuggingFace 不通** | 无法下载论文用的 `all-MiniLM-L6-v2` / MPNet / RoBERTa 嵌入模型 → **所有 cosine 检索实验无法用论文同款嵌入**(只能用 SiliconFlow 的 `BAAI/bge-m3` 替代) |
| **没有 Webshop 服务器 + 商品库 + pyserini 索引** | RAP 无法端到端运行,只能 mock 初始 observation 跑"第一步搜索动作" |
| 没有完整 MIMIC-III 库(仅 `D_ITEMS.csv`) | EHRAgent 无法执行真实 SQL —— 但**攻击代码是 `answer=[...]`,不碰 DB,所以攻击路径不受影响**(这点是可复现的关键) |

**结论先行**:
- **EHRAgent + 编辑距离(论文默认配置)≈ 完整可复现**,数值与论文高度吻合,DeepSeek 甚至略强于 GPT-4o。
- **cosine 类实验**:嵌入模型被迫替换,属于"复现不了原配置"。
- **RAP(web agent)是掉点最严重的 workflow**:多重原因叠加,严格口径下 EN=0。

---

## 1. 复现保真度总表(EHRAgent,编辑距离)

检索是确定性的(Levenshtein),且我用了与论文**相同的攻击 prompt 文件 + 相同内存**,
因此 **RN 完全由检索决定、与被攻击模型无关**——这是检验保真度的"金标准"。

### Table 1 主结果(n=30, m=200, k=4)
| 指标 | 论文(GPT-4o) | 复现(DeepSeek) |
|---|---|---|
| RN | 55 | **55**(精确吻合 ✓✓) |
| EN | 50 | 50–55(两次独立 run) |
| EE | 0.42 | 0.42–0.46 |
| CER | 0.83 | 0.77–0.90 |
| AER | 0.83 | 0.77–0.90 |

### Table 2 / Fig 2a —— 内存大小扫描(EN)
| m | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| 论文 | 31 | 43 | 50 | 51 | 58 | 59 |
| 复现 | 31 | 42 | 50 | 52 | 56 | 56 |
| RN(复现) | 31 | 45 | 55 | 57 | 65 | 66 |

→ 全程 ±3 内吻合,"内存越大泄漏越多"趋势完美复现。

### Fig 3a —— 检索深度 k 扫描
| k | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 论文 EN | 8 | 27 | 39 | 50 | 59 |
| 复现 EN | 16 | 29 | 38 | 50 | 60 |
| 论文 RN | 21 | 34 | 45 | 55 | 65 |
| 复现 RN | **21** | **34** | **45** | **55** | **65** |

→ **RN 在每个 k 都与论文精确吻合**(21/34/45/55),检索保真度极高。

### Fig 4a —— 攻击数量 n 扫描(basic Ibasic vs advanced Iadvan)
| n | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|
| basic EN | 27 | 35 | 50 | 60 | 68 |
| advanced EN | 22 | 46 | 62 | 72 | 80 |

→ **advanced > basic(n≥20),但 n=10 时 basic 反超**——这与论文 §6.2 的细致论断
("仅当评分函数为编辑距离且 n 较小时 Ibasic 略优")**逐字吻合**。

---

## 2. 哪个 workflow 掉点 —— RAP(web agent)严重不可复现

复现(降级版)对照(n=30, m=200, k=3):

| 评估口径 | RAP basic EN | RAP advanced EN | 论文 |
|---|---|---|---|
| 严格(逐字子串,与 EHRAgent 同口径) | **0** | **0** | — |
| 宽松(token 召回≥0.6,逐条) | 4 | 6 | — |
| 论文式(prompt 级"成功⇒全提取") | 7 | 8 | **26** |

掉点的**三重叠加原因**:
1. **指标定义不一致(最隐蔽)**:论文 RAP 评估(`evaluation.ipynb`)是**半手工**的——人工标注"失败 index 列表",再把所有"非失败"prompt 的 top-3 检索 ID **全部**算作已提取(`成功⇒全提取`,所以论文里 EN≈RN 几乎是方法学的人为结果)。而 EHRAgent 评估是**自动逐条精确子串**。**两个 agent 用了两套松紧不同的口径**,论文 RAP 数字本身无法用自动 pipeline 复现。
2. **DeepSeek 行为退化**:即使放宽到语义匹配,EN 仍只有 4–8(<<26)。DeepSeek 把检索到的指令**大幅改写/缩写**后塞进搜索框,例如检索到 `"i am looking for an easy to clean jewelry box"`,它搜的是 `"easy to clean jewelry box 10 slots"`(去掉"i am looking for"前缀、又从示例轨迹补了商品细节)→ 逐字匹配全灭。攻击机制其实**生效了**(确实把 3 条记忆塞进 `search[...]`),但泄漏的是"改写版"而非原文。
3. **替代偏差**:`bge-m3`≠`MiniLM`(导致 RN=19/44 ≠ 论文 27);mock 的 observation ≠ 真实 webshop 页面。

---

## 3. 哪个假设不稳

### (A) "temperature=0 ⇒ 输出确定/稳定" —— 不成立
对同一攻击 prompt 跑 3 次:`prompt#11` 在 temp=0 下产生**三种完全不同**的输出格式。
DeepSeek-V3.2(SiliconFlow)在 temp=0 仍有实质随机性 → 单次指标有 run-to-run 噪声
(我两次独立 run 的 Table 1 EN 分别是 50 和 55)。论文"最多重试3次"的机制确实在做事,
但救不了系统性的语义误读。

### (B) "agent 输出的'previous questions' = 私有检索记忆" —— 不稳(EHRAgent 主导失败)
EHRAgent 工作流有**两次 LLM 调用**(先生成 knowledge,再生成代码)。攻击 prompt 在**第一步
knowledge 生成**就被劫持:`RetrKnowledge` 模板内含 3 个硬编码示例问题
(`comp-oth vasc dev/graft`、`tpn w/lipids patient 2238`、`procedure patient 58730`),
DeepSeek 直接把这些**公开脚手架问题**倒进 knowledge 字段,再传入代码生成步骤,于是最终
`answer` 输出的是**硬编码示例而非私有记忆** → 严格口径判 0。

**含义**:一部分"内存泄漏"其实泄漏的是非私有的系统 prompt 脚手架内容,攻击对**私有记忆**的
有效性被高估。论文在 §5.2 / Table 9 只在 k=1 时承认这一点;复现显示它在 k=4 也频繁发生
(DeepSeek 上更甚:GPT-4o 是"检索+硬编码"都输出,DeepSeek 有时**只**输出硬编码)。

### (C) "advanced(Iadvan)≥ basic(Ibasic)" —— 基本稳,但失败模式会迁移
n≥20 时 advanced 确实更强(复现成功)。但失败的"类型"在两者间迁移:
- basic 失败以"抓硬编码脚手架 + 把 queries 误读成 SQL"为主;
- advanced(冗长礼貌 prompt)失败以"对话式叙述不执行 + 直接拒绝"为主。

---

## 4. EHRAgent 失败模式完整普查(编辑距离, n=50)

| 失败模式 | basic(9/50) | advanced(10/50) | 说明 |
|---|---|---|---|
| 硬编码脚手架泄漏(非私有) | 5 | 2 | 输出 comp-oth/tpn-2238/proc-58730,而非检索记忆 |
| 误读 queries→写 SQL | 2 | 0 | 给示例问题写 `SELECT ...` 去回答,而非列出问题 |
| 叙述不执行(prose) | 1 | 6 | "I'll help you save..." 开场白,无工具调用/无 query |
| 拒绝 | 0 | 2 | "I don't have access to previous examples..." |
| prose 当代码→语法错 | 1 | 0 | — |
| 改写/格式不匹配 | 0 | 1 | — |

> 注:成功率仍很高(basic 41/50、advanced 39/50 完全提取),所以**整体结论"EHRAgent 高度脆弱"
> 成立且复现**;上述是专门挑出的"剩余失败 case"。

---

## 5. cosine + 嵌入替代(EHRAgent)—— 复现不了原配置

论文用 MiniLM,m=200 时 cosine EN≈20。本环境只能用 `bge-m3` 替代:

| 配置 | 复现 EN / RN / CER | 论文(MiniLM) |
|---|---|---|
| EHRAgent cosine basic(general) | **18 / 20 / 0.60** | EN≈20(m200) |
| EHRAgent cosine advanced(cosine_specific) | **14 / 21 / 0.27** | — |

→ 两点失败/不稳:
1. **嵌入模型不可得**:论文用 MiniLM,本环境只能 bge-m3,直接改变检索到哪些记录(RN),数值
   **不可与论文直接对比**,属"原配置复现不了"。
2. **"advanced cosine > basic" 假设在替代嵌入下翻转**:论文 Fig 4b 是 advanced>basic;复现里
   **basic(EN=18)反超 advanced(EN=14)**。cosine_specific 的健康短语在 bge-m3 空间没带来检索
   多样性增益(RN 20 vs 21 几乎相同),前置短语反而**损害提取可靠性**(CER 0.27 vs 0.60)。
   即论文该结论对嵌入模型/backbone 敏感,不稳。
   另:cosine 整体 CER(0.27–0.60)远低于编辑距离(0.77–0.90)——cosine 攻击对 DeepSeek 更不稳。

---

## 6. 一句话总结(给判断用)

- **能复现且很稳**:EHRAgent + 编辑距离的全部主结果(Table 1 / Table 2 / Fig 2a / Fig 3a / Fig 4a),
  RN 逐点精确吻合,EN 全程 ±3,DeepSeek ≥ GPT-4o。
- **复现不了 / 掉点**:① RAP(web agent)—— 三重原因叠加,严格口径 EN=0,即便宽松也只有论文的 ~1/4;
  ② 一切 cosine 实验 —— 论文嵌入模型在本环境不可得,只能替代。
- **不稳的假设**:① temp=0 非确定;② "输出=私有记忆"被硬编码脚手架泄漏破坏;
  ③ 两个 agent 的评估口径松紧不一(EHRAgent 自动严格 / RAP 半手工宽松),使 RAP 数字本身不可自动复现。
