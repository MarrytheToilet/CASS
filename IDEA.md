# CASS: Compositional Adaptive Subspace Steering
## KDD 2027 投稿完整研究计划书(v1.0,2026-07-03)

**一句话**:从LLM激活数据中挖掘"技能子空间字典",对未见任务用组稀疏编码零训练地合成转向算子,提升few-shot任务表现。

**硬约束**:摘要 2026-07-19(AoE),全文 2026-07-26(AoE);单卡 RTX 3090 24GB + LLM API;单人。

**贡献声明(写作时逐条对打的四点)**
1. **方法**:首个训练无关的未见任务转向合成框架——技能子空间字典 + 组稀疏系数闭式求解 + 查询自适应仿射注入(vs. ELICIT 只检索同款 / ATV 需训练生成器 / conceptor 只组合已知行为)。
2. **机制发现**:任务向量共享一个"通用任务分量",它是朴素向量组合失效的主因;正交投影剥离后组合性显著恢复(消融直接验证)。
3. **理论**:给出字典块相干度条件下的技能支撑集恢复保证,以及"重构残差 → 转向性能"的误差传递界——该研究线上首个恢复性理论。
4. **规模化叙事(KDD锚点)**:字典规模—合成质量的正向曲线 + 每查询token成本对比,把"激活结构挖掘"讲成可复用的知识资产。

---

# 第一部分:故事线(对应论文逐节)

## 1.1 标题候选
- CASS: Training-Free Skill Composition for Unseen Tasks via Adaptive Subspace Steering
- Mining Reusable Skill Subspaces from LLM Activations for Compositional Task Adaptation
- From Task Vectors to Skill Dictionaries: Training-Free Synthesis of Steering Operators for Unseen Tasks

## 1.2 Abstract 草稿(英文,投稿用底稿)
> Large language models encode task information as manipulable structures in their activation space: task vectors extracted from in-context demonstrations can trigger task execution when injected elsewhere. However, existing steering approaches treat each task in isolation—vectors are extracted per task, stored, and retrieved, leaving models unable to handle tasks absent from the library. We ask: can steering operators for *unseen* tasks be *synthesized* from a dictionary of known skills, without any training? We propose CASS (Compositional Adaptive Subspace Steering). CASS mines a dictionary of low-rank skill subspaces from contrastive activations, after projecting out a shared "task-generic" component that we identify as the primary cause of interference in naive vector composition. Given only k≤4 demonstrations of a novel task, CASS solves a group-sparse coding problem with closed-form block updates to select and weight relevant skills, then applies a query-adaptive affine steering operator. We prove support-recovery guarantees under a block-coherence condition on the dictionary and bound the steering sub-optimality by the reconstruction residual. Across 30+ tasks and three 7B-scale LLMs, CASS recovers XX% of oracle task-vector performance on held-out tasks, correctly identifies constituent skills of compound tasks, and improves monotonically with dictionary size—while eliminating the per-query token cost of full in-context prompting.

(XX% 由 E1 实验回填;若走"族内成立"叙事,abstract 相应改写,见 §4.4 预案。)

## 1.3 引言逐段逻辑(1.25页)
1. **段1(现象)**:LLM 激活空间存在任务结构——不同任务诱导分区化隐藏表示(引 SEAP, arXiv:2503.07605 的多任务聚类实验);任务信息可提取为 task/function vector 并因果地驱动任务执行(引 Hendel et al. 2023; Todd et al. 2024)。
2. **段2(矛盾)**:但现有方法把任务当孤岛。单向量表示高秩任务失效(引 ICLR 2026 task vector 秩限制论文);库+检索(ELICIT)只能查同款;自适应生成(ATV)要训练网络。**未见任务 = 现有方法的集体盲区**。
3. **段3(直觉)**:人解决新问题不造新工具,而是组合旧工具。任务表示是否也具有这种组合结构?我们发现:朴素相加失败不是因为组合性不存在,而是因为所有任务向量共享一个通用分量,叠加时相互污染——剥离它,组合性浮现。
4. **段4(方案)**:CASS 三步——挖字典(共性分离+低秩子空间)、解组合(组LASSO闭式块更新)、自适应注入(仿射矫正)。全程零训练零梯度,毫秒级求解。
5. **段5(贡献清单)**:上面四点 + 实验数字预告。

## 1.4 Related Work 分组与差异句(0.75页,四组)
| 组 | 代表工作 | 一句话差异 |
|---|---|---|
| Task/function vectors | Hendel 2023; Todd 2024; Learned Task Vectors; ICLR26秩限制 | 它们提取与分析单任务向量;我们**合成**未见任务的算子,且以子空间为原语回应其高秩失效 |
| 向量库与自适应 | ELICIT(检索); ATV(训练生成) | 检索无法泛化到库外;生成器需训练。我们训练无关且显式给出组合结构(可解释系数) |
| 激活转向 | ActAdd; RepE; conceptor组合转向 | 它们组合**已知**行为目标;我们从few-shot示例**推断**未知任务的组合。conceptor作为组合算子baseline进入实验 |
| 任务结构挖掘 | SEAP(任务激活聚类→剪枝); Deja Vu(上下文稀疏) | 同样利用任务激活结构,但用于删计算;我们用于**增效果**(转向),且处理未见任务的合成 |

## 1.5 KDD 适配写法备忘
- 引言首段场景:大规模多任务 LLM 服务管线,长尾任务不断出现,每任务 full ICL 的 token 成本 = k-shot 上下文 × 百万查询;CASS 一次挖掘、处处合成。
- "挖掘"用词贯穿:dictionary **mining**、skill **discovery**、activation **data**。
- E5(规模曲线)与 token 成本表放主实验区,机制发现放分析节——重心是效果与规模,机制是解释。

---

# 第二部分:方法(完整数学)

## 2.0 记号
- 模型 $M$,层数 $L$,残差流维度 $d$。Llama-3.1-8B-Instruct: $d{=}4096, L{=}32$;Qwen2.5-7B-Instruct: $d{=}3584, L{=}28$;Mistral-7B-Instruct-v0.3: $d{=}4096, L{=}32$。
- $h_\ell(p) \in \mathbb{R}^d$:prompt $p$ 在层 $\ell$ 最后一个 token 的残差流隐状态。
- 基任务集 $\mathcal{T} = \{1,\dots,T\}$($T \approx 30$),每任务数据集 $D_t = \{(x_i, y_i)\}$。

## 2.1 模块一:技能子空间字典挖掘

### 2.1.1 对比激活提取
对任务 $t$、样本 $i$,构造成对 prompt:
- $p_i^{t,+}$:10-shot 正确示例 + 查询 $x_i$(标准 ICL prompt);
- $p_i^{t,-}$:同样 10 个示例但**标签随机置换** + 同一查询(保留格式与词表,破坏任务映射;function vector 文献的标准 corrupted prompt)。

差分激活:
$$g_i^{(t)} = h_{\ell^*}(p_i^{t,+}) - h_{\ell^*}(p_i^{t,-}) \in \mathbb{R}^d$$
每任务取 $n{=}100$ 对,堆叠 $G_t = [g_1^{(t)},\dots,g_n^{(t)}] \in \mathbb{R}^{d\times n}$。层 $\ell^*$ 的选择见 §2.3.1。

**实现要点**:hook 用 nnsight 或 baukit 的 TraceDict;bf16 前向;每任务提取约 200 次前向,单任务 < 2 分钟(3090)。存储:$34 \times 4096 \times 100 \times 2\text{B} \approx 28$ MB/模型,直接存 `.pt`。

### 2.1.2 共性分离(方法的机制核心)
拼接所有任务的**均值向量** $\bar{g}_t = \frac{1}{n}\sum_i g_i^{(t)}$ 为 $\bar{G} = [\bar{g}_1,\dots,\bar{g}_T] \in \mathbb{R}^{d\times T}$,对 $\bar{G}$ 做 SVD,取前 $r_0$ 个左奇异向量为共享子空间 $U_0 \in \mathbb{R}^{d\times r_0}$。

**假设 H1**:$U_0$ 编码"正在执行某个 ICL 任务"的通用信息(格式遵循、输出模式),它在所有 $\bar{g}_t$ 中占据大量能量且方向相近,是朴素向量相加时的干扰源。

去共性:
$$\tilde{G}_t = (I - U_0 U_0^\top)\, G_t$$

$r_0$ 的选择:扫描 $r_0 \in \{0,1,2,4,8,16\}$,两个观测指标——(a) 去共性后任务均值向量两两余弦相似度矩阵的中位数(应显著下降);(b) E1 恢复率。$r_0{=}0$ 即"无共性分离"消融。预期最优在 1–4。

### 2.1.3 任务子空间提取
对每个 $\tilde{G}_t$ 做截断 SVD:$\tilde{G}_t = U\Sigma V^\top$,取满足谱能量 $\frac{\sum_{j\le r_t}\sigma_j^2}{\sum_j \sigma_j^2} \ge \tau$ 的最小 $r_t$($\tau{=}0.90$,上限 $r_t \le 16$),得正交基 $U_t \in \mathbb{R}^{d\times r_t}$,并记录任务锚点 $\mu_t = \bar{g}_t$(去共性后)。

字典:$\mathcal{D} = \{(U_t, \mu_t, \sigma_t^{(1:r_t)})\}_{t=1}^T$。附带产出(论文分析图):各任务的谱衰减曲线——预言"高秩任务"(衰减慢)正是单向量方法失效的任务,与 ICLR26 结论交叉验证。

## 2.2 模块二:组稀疏合成(未见任务)

### 2.2.1 目标查询表示
新任务 $t^\star$ 给 $k \le 4$ 个示例。用同样的 ±prompt 差分法提取 $z_j, j{=}1..k$(每个示例做一对),取均值并去共性:
$$z = (I - U_0U_0^\top)\,\frac{1}{k}\sum_j z_j$$
(子空间版:$k{\ge}4$ 时可对 $[z_1..z_k]$ 取秩-$r_z$ 主子空间,$r_z \le 2$;默认用均值向量,子空间版进消融。)

### 2.2.2 组 LASSO 求解
$$\min_{c_1,\dots,c_T}\; \frac{1}{2}\Big\| z - \sum_{t=1}^{T} U_t c_t \Big\|_2^2 + \lambda \sum_{t=1}^{T} \sqrt{r_t}\,\|c_t\|_2, \qquad c_t \in \mathbb{R}^{r_t}$$

**闭式块坐标下降**(利用 $U_t^\top U_t = I_{r_t}$,每块更新是精确的组软阈值):

```
初始化 c_t = 0;重复直到 max_t ||Δc_t|| / (||c_t||+1e-8) < 1e-6 或 200 轮:
  for t in 随机置换(1..T):
      r = z - Σ_{s≠t} U_s c_s          # 残差
      b = U_t^T r                       # r_t 维
      c_t = max(0, 1 - λ√r_t / ||b||_2) · b   # 组软阈值
```
复杂度每轮 $O(d\sum_t r_t)$,$d{=}4096, \sum r_t \approx 200$ 时全程毫秒级(CPU numpy 即可)。

**λ 选择(无验证集,数据仅 k 个示例)**:λ 路径($\lambda_{max}$ 到 $0.01\lambda_{max}$ 对数取 20 点,$\lambda_{max} = \max_t \|U_t^\top z\|_2/\sqrt{r_t}$),用 **k 折留一示例重构误差**(用 k−1 个示例的均值解系数,在留出示例上算残差)选 λ;k=1 时退化为固定 $\lambda = 0.3\lambda_{max}$(消融给敏感性曲线)。

**求解器消融组**:(a) 普通最小二乘(无稀疏,全字典);(b) Block-OMP(贪心,支撑集大小 ≤3);(c) 单纯形约束($c$ 标量化、$\sum w_t = 1, w_t \ge 0$);(d) 主方案组 LASSO。

产出:系数 $\{c_t^*\}$、支撑集 $S = \{t: \|c_t^*\| > 0\}$、合成向量 $\Delta = \sum_{t\in S} U_t c_t^*$、重构残差 $\varepsilon = \|z - \Delta\|_2 / \|z\|_2$。**$\varepsilon$ 同时是理论量、go/no-go 指标和推理期的置信信号(过大 → 系统回退 full ICL,这就是"成本感知自适应"预案的钩子)。**

## 2.3 模块三:查询自适应注入

### 2.3.1 注入层选定(库构建期一次性)
标准 causal patching 扫描:对每层 $\ell$,把每个基任务的 oracle 均值向量加到该任务 corrupted prompt 的层-$\ell$ 最后 token 隐状态上,测 50 条查询的准确率恢复量;取跨任务平均恢复量最高的层为 $\ell^*$(7B 级模型经验上在中部,约 $\ell \in [L/3, L/2]$)。全程约 $L \times 34 \times 50$ 次前向 ≈ 3090 上一夜,只跑一次。消融:top-1 层 vs top-3 层同时注入。

### 2.3.2 仿射注入算子
令 $P_S = U_S U_S^\top$($U_S$:$\{U_t\}_{t\in S}$ 列拼接后 QR 正交化),目标锚点 $\mu_S = \sum_{t\in S} w_t \mu_t$,$w_t = \|c_t^*\|_2 / \sum_{s\in S}\|c_s^*\|_2$。对查询前向中层 $\ell^*$ 最后 token 的 $h$:

$$h \leftarrow h + \alpha(h)\,\big[\underbrace{\gamma\,\Delta}_{\text{方向注入}} + \underbrace{P_S(\mu_S - h)_{\parallel}}_{\text{子空间内矫正}}\big], \quad \alpha(h) = \min\!\Big(\alpha_{max},\; \beta\,\frac{\|(I-P_S)(h-\mu_S)\|_2}{\|h\|_2}\Big)$$

直觉:$h$ 偏离合成任务流形越远,矫正越强;已在流形内则几乎不动。超参默认 $\gamma{=}1, \beta{=}4, \alpha_{max}{=}2$,粗网格 $\beta\in\{2,4,8\}, \alpha_{max}\in\{1,2,4\}$ 在 3 个基任务上定一次、全局冻结(避免 per-task 调参嫌疑)。消融:纯加法 $h{+}\Delta$ vs 仿射版。

## 2.4 理论(0.75页,两条命题)

**命题1(支撑集恢复)** 定义字典块相干度 $\mu_B(\mathcal{D}) = \max_{s\neq t} \sigma_{max}(U_s^\top U_t)$(正交块下即最大主角余弦)。设真实表示 $z = \sum_{t\in S_0} U_t c_t^0 + e$,$|S_0| = k_0$,$\|e\| \le \epsilon$。则当
$$k_0 < \tfrac{1}{2}\big(\mu_B^{-1} + 1\big)$$
且 λ 与 $\epsilon$ 适配时,组 LASSO 的解支撑集 $\hat{S} \subseteq S_0$ 且系数误差 $O(\epsilon)$。
*证明路线*:直接套用块稀疏恢复 / union-of-subspaces 理论(Eldar & Mishali 2009, IEEE TIT; Eldar, Kuppinger & Bölcskei 2010 的块相干度条件;group lasso 一致性 Yuan & Lin 2006)。**不自造定理,做条件验证**:附录报告真实字典的 $\mu_B$ 实测值及共性分离前后的对比——H1 的第二重证据(分离显著降低 $\mu_B$,理论与机制闭环)。

**命题2(转向误差传递)** 设任务性能可由局部线性探针近似:目标 logit 差 $m(h) = w^\top h + O(\|h\|^2_{loc})$。oracle 注入 $\Delta^{or}$ 与合成注入 $\Delta$ 的性能差满足
$$|m(h + \Delta) - m(h + \Delta^{or})| \le \|w\|_2 \cdot \big(\varepsilon\|z\| + \delta_{est}\big)$$
其中 $\delta_{est}$ 是有限样本子空间估计误差,由 Davis–Kahan $\sin\Theta$ 定理以 $O(\sigma_{noise}/(\sqrt{n}\,\text{gap}))$ 控制。
*意义*:把 go/no-go 指标(重构残差 $\varepsilon$)与最终任务指标定量焊接;实验里画 $\varepsilon$ vs 恢复率的散点验证(预期负相关,这张图同时是理论验证和方法诊断)。

## 2.5 备选方案(E1 失败预案,代码 90% 复用)
- **预案B(族内成立)**:叙事转为"技能族结构的发现与利用"——层次字典(族间路由 + 族内组合),主张改为"组合性是族局部的,CASS 自动发现族边界"(系数支撑集的块结构就是证据)。
- **预案C(全面失败)**:转做"子空间注入修复单向量高秩失效"(§2.1.3 + §2.3.2 已覆盖方法),对打 ICLR26 论文的失效任务,贡献降维但仍完整。

---

# 第三部分:实验方案(逐实验协议)

## 3.0 环境与资源
- **硬件**:RTX 3090 24GB。bf16 推理:8B 模型权重约 16GB,batch=8、序列 ≤512 时峰值 ≈ 19–21GB,安全;若 OOM 降 batch=4。
- **软件栈**:`transformers` + `accelerate` + `nnsight`(或 `baukit`)做 hook;`numpy/scipy` 做 SVD 与组 LASSO(可选 `skglm`/`celer` 验证自写求解器);`pandas + matplotlib/seaborn` 出图;实验管理用简单 CSV + git tag,不引入重框架。
- **模型清单**(HF 权重,3090 全部可跑，可以参考../../models中的模型，看看那个可以使用):
  1. Llama-3.1-8B-Instruct(主模型,所有实验)
  2. Qwen2.5-7B-Instruct(跨模型验证)
  3. Mistral-7B-Instruct-v0.3(跨模型验证)
  4. GPT-J-6B(备用:对齐 function vector 老文献,若时间不够则砍)
- **代码底座**(第1天动作:全部 clone 并跑通其 demo):
  - Todd et al. function vectors 官方 repo(任务数据 + 提取/注入参考实现;搜 `function_vectors github Todd`)
  - ICLR 2026 秩限制论文官方 repo(34 任务评测框架;搜 `ICL-TaskVector github`)
  - ELICIT、ATV 官方 repo(baseline;搜 `ELICIT task vector library github` / `Adaptive Task Vectors github`)
  - SEAP repo:`github.com/IAAR-Shanghai/SEAP`(仅借聚类分析脚本思路，可以不clone)
  - conceptor steering repo(baseline;搜 OpenReview `From Steering Vectors to Conceptors` 附带代码)

## 3.1 任务底座(以 Todd repo 实际清单为准,按四族组织,目标 T≈30)
- **知识映射族**:country-capital, country-currency, landmark-country, national-park-country, person-occupation, person-sport, person-instrument, product-company
- **语言学变换族**:antonym, synonym, present-past, singular-plural, capitalize, lowercase, adjective-superlative, verb-noun
- **翻译族**:english-french, english-german, english-spanish, french-english, german-english, spanish-english
- **算法/符号族**:next-item, prev-item, alphabetically-first, alphabetically-last, choose-first-of-list, choose-last-of-list, word-length, first-letter
- 每任务:字典构建用 100 样本,评测用**不相交的** 50 查询;5 个随机种子重采样示例组合。
- 指标:exact-match accuracy(首 token 或首词匹配,沿用底座 repo 的判定函数,不自造)。

## 3.2 E1 主实验:Leave-One-Task-Out 合成(go/no-go 载体)
**目的**:未见任务的转向能否由其余任务字典合成。
**协议**:
```
for 模型 in {Llama-3.1-8B}:                    # go/no-go 阶段只跑主模型
  构建全字典 D(34任务), 选定注入层 ℓ*
  for 留出任务 t* in 全部任务:
    D_-t = D \ {t*}
    for k in {1, 2, 4}:                        # few-shot 数
      for seed in {1..5}:
        抽 k 个示例 → z → 组LASSO(D_-t) → Δ, S, ε
        在 50 条评测查询上注入, 记 acc_syn
    记录: acc_zs(零样本), acc_icl(10-shot full ICL),
          acc_oracle(t* 自己的子空间注入), acc_naive(全字典均值向量相加)
```
**核心指标**:恢复率 $\rho = \dfrac{acc_{syn} - acc_{zs}}{acc_{oracle} - acc_{zs}}$(分母 < 5pt 的任务剔除并单独报告);辅助:$\varepsilon$、支撑集大小、$\rho$ vs $\varepsilon$ 散点。
**呈现**:按任务族分组的柱状图(Fig 3)+ 全任务明细表(附录)。
**耗时估算**:34 任务 × 3k × 5 seed × 50 查询 ≈ 2.6 万次生成(短输出)+ 注入前向 ≈ 3090 上 8–10 小时,一夜可完。

**Go/No-Go 判定表(7/9 晚执行,k=4 档)**:
| 档 | 判定条件 | 动作 |
|---|---|---|
| A 全速 | ≥50% 任务 ρ≥0.6,且 ρ 中位数 ≥0.5 | 按主叙事推进 |
| B 族内 | 存在 ≥2 个族内部 ρ 中位数 ≥0.6,跨族合成 ρ<0.3 | 切预案B(层次字典/族结构叙事),实验矩阵不变 |
| C 止损 | 全体 ρ 中位数 <0.3 且无族内亮点 | 切预案C(高秩修复),砍 E2/E5,加 ICLR26 失效任务复现 |

## 3.3 E2 组合任务:支撑集恢复(可解释性主图)
**构造**:用"可叠加算子"链式合成 8–10 个复合任务,例如 country-capital∘capitalize(输出大写首都)、english-french∘lowercase、antonym∘capitalize、singular-plural∘first-letter 等(以两成分为主,1–2 个三成分)。复合任务**不入字典**。
**指标**:支撑集 precision / recall(命中成分任务视为正确)、系数热力图(Fig 4:行=复合任务,列=字典任务,预期对角块点亮)、复合任务上的 ρ。
**对照**:ELICIT 检索(预期检索到单个最相似任务,拿不到组合)、朴素两向量平均。

## 3.4 E3 高秩修复:秩扫描
在 ICLR26 论文标注的单向量失效任务上(以其 repo 复现清单为准,预期为多对多映射类),扫 $r_t \in \{1,2,4,8,16\}$,画性能—秩曲线(Fig 5);r=1 应复现其失效,r 增大应恢复。同图叠加该任务谱衰减,验证 §2.1.3 的预言。

## 3.5 E4 消融矩阵(每项只在主模型 + 8 个代表任务上跑)
| 消融轴 | 取值 | 验证什么 |
|---|---|---|
| 共性分离 r0 | 0 / 1 / 2 / 4 / 8 / 16 | H1 机制主张(r0=0 时组合应显著变差) |
| 求解器 | 组LASSO / LS / Block-OMP / 单纯形 | 组稀疏的必要性 |
| 注入形式 | 仿射自适应 / 纯加法 / 仅 P_S 矫正 | 模块三增益 |
| 注入层 | ℓ* / ℓ*±4 / top-3 层 | 层选择稳健性 |
| k(示例数) | 1 / 2 / 4 / 8 | few-shot 极限 |
| λ | 路径 20 点 | 敏感性曲线 |
| n(每任务样本) | 25 / 50 / 100 | 字典构建成本下限(联动命题2的 δ_est) |

## 3.6 E5 字典规模曲线(KDD 主打图)
$T' \in \{5,10,15,20,25,30\}$ 随机子字典 ×5 次抽样,对固定的 6 个留出任务测 ρ,画均值±CI 曲线(Fig 6)。同节给 **token 成本表**:full 10-shot ICL vs CASS(k=4 一次性提取 + 后续零上下文)在 10/100/1000 查询下的每任务累计 token 数与延迟。

## 3.7 E6 跨模型 + 统计
- Qwen2.5-7B 与 Mistral-7B 上重跑 E1(k=4)与 E2,验证结论不依赖单模型。
- 所有对比:5 seed,配对 bootstrap(10k 重采样)给 95% CI;主表加显著性标记。
- Baseline 复现优先级:朴素相加、LS、conceptor(自实现成本低)必做;ELICIT、ATV 官方 repo 若 7/12 前跑不通,改为在其原始任务配置上并列报告其论文数字并明确标注(诚实处理,评审可接受)。

## 3.8 计算预算总账(3090 小时)
| 项目 | 估算 |
|---|---|
| 字典提取 ×3 模型 | 3×1.5h = 4.5h |
| 层扫描 ×3 模型 | 3×8h = 24h(夜间) |
| E1 主模型 + 跨模型 | 10h + 2×6h = 22h |
| E2/E3 | 6h |
| E4 消融 | 12h |
| E5 | 6h |
| 缓冲(重跑/调参) | ×1.5 系数 |
| **合计** | ≈ 110 GPU·h ≪ 23天×24h,**算力不是瓶颈,人是** |

---

# 第四部分:逐日排期(2026-07-03 → 07-26,均为当日目标)

**周一(7/3–7/6)基建周**
- 7/3(五):clone 五个 repo;Llama-3.1-8B 跑通 Todd repo 的提取+注入 demo;确定任务清单终版。
- 7/4(六):写完对比激活提取管线(±prompt 构造、hook、存储);对 5 个任务抽查提取质量(oracle 向量注入应恢复 ICL 性能的 70%+,这是管线正确性的验收标准)。
- 7/5(日):全 34 任务字典提取;层扫描过夜跑。
- 7/6(一):共性分离 + 子空间提取;出三张诊断图(任务均值余弦矩阵、谱衰减、μ_B 分离前后)。**当晚检查点:H1 的相似度矩阵是否如预期(分离前中位余弦 >0.4,分离后 <0.15 量级)——不符则 r0 扫描提前。**

**周二(7/7–7/9)go/no-go 周**
- 7/7:组 LASSO 求解器实现 + 单元测试(合成数据上验证支撑集恢复);注入代码完成。
- 7/8:E1 全量过夜跑(k∈{1,2,4} × 5 seed)。
- 7/9:E1 分析,**按 §3.2 判定表定档**;当晚把结果与判定发给可信的同学/导师做外部 sanity check。

**周三(7/10–7/16)主实验周**
- 7/10–7/11:baseline 组(朴素相加/LS/conceptor 自实现;ELICIT、ATV repo 攻坚,7/12 截止决定复现 or 引用数字)。
- 7/12:E2 复合任务构造与运行;系数热力图初版。
- 7/13:E3 秩扫描;E4 的 r0 与求解器两轴。
- 7/14:E4 其余轴;**写作启动:方法节(§2 直接翻译成英文,你的数学已定稿)**。此后每天 ≥2h 写作雷打不动。
- 7/15:E5 规模曲线 + token 成本表;写引言。
- 7/16:E6 跨模型过夜;写 related work + 理论节。

**周四(7/17–7/23)写作冲刺周**
- 7/17:全部主图定稿(Fig 1 教学图手绘线框 → 制图);实验节初稿。
- 7/18:abstract 定稿 + 全文串读;**7/19(日)AoE 前提交摘要**(注意 AoE = UTC-12,北京时间 7/20 上午 8 点前,别卡点)。
- 7/20–7/21:补洞实验(评审视角自查:每个贡献声明是否有对应图表?);附录(证明细节、全任务表、超参表)。
- 7/22:理论节复核(命题条件与实测 μ_B 数字对齐);限制与伦理声明节。
- 7/23:初稿完成,冷冻 12 小时。

**周五(7/24–7/26)收尾**
- 7/24:全文修订轮1(逻辑与主张一致性);图表编号/引用格式清理(ACM 双栏模板,页数限制以 KDD 2027 CFP 为准,历年研究track正文9页+参考文献不限)。
- 7/25:修订轮2(语言);可复现性清单;匿名化检查(repo 链接匿名镜像)。
- 7/26:**AoE 截止前 6 小时提交全文**;arXiv 预印本待会后再挂(遵守 KDD 双盲政策,确认 CFP 对预印本的规定)。

---

# 第五部分:论文结构与图表清单

**结构(按 9 页正文预算)**:1 Introduction(1.25p)/ 2 Related Work(0.75p)/ 3 Skill Structure in Activation Space:现象与 H1(1p)/ 4 CASS 方法(2p)/ 5 Theoretical Analysis(0.75p)/ 6 Experiments(2.75p)/ 7 Conclusion(0.25p)+ 附录(证明、任务表、超参、额外消融)。

**图表清单(共 6 图 4 表)**
- Fig 1 教学图:左"工具箱比喻+字典→合成→注入"流程,右一个真实复合任务的系数条形图(一图讲完论文)。
- Fig 2 方法管线(三模块)。
- Fig 3 E1 主结果(按族分组柱状,带 oracle/ICL 上界虚线)。
- Fig 4 E2 系数热力图。
- Fig 5 E3 秩—性能曲线(叠谱衰减)。
- Fig 6 E5 字典规模曲线。
- Tab 1 主对比表(CASS vs 全 baseline,3 模型)。
- Tab 2 消融总表。Tab 3 token 成本。Tab 4 μ_B 与命题1条件实测。

---

# 第六部分:风险登记表

| # | 风险 | 概率 | 影响 | 缓解 / 触发预案 |
|---|---|---|---|---|
| R1 | 组合性全面不成立 | 20–25% | 致命 | 7/9 三档判定;预案B/C 代码复用 90% |
| R2 | ELICIT/ATV 复现失败 | 40% | 中 | 7/12 截止改引用其论文数字并标注 |
| R3 | 撞车(同命题 arXiv 先发) | 15% | 高 | 每周一/四扫 arXiv(cs.CL+cs.LG, 关键词 task vector, steering, composition);若撞,差异化到子空间+理论+自适应注入三点 |
| R4 | 提取质量差(oracle 注入都不恢复) | 15% | 高 | 7/4 验收标准拦截;换层/换 corrupted prompt 构造(空示例版) |
| R5 | 写作时间被实验挤占 | 50% | 高 | 7/14 起每日 2h 硬约束;7/19 摘要日强制引言+方法成稿 |
| R6 | KDD 评审嫌"太 ICLR" | 30% | 中 | §1.5 写法备忘;E5+token成本前置;若 7 月中自评气质不合,9 月转投 ICLR 2027 的 plan B 心理建设 |
| R7 | 3090 故障/占用冲突 | 10% | 中 | 关键跑批存 checkpoint;API 侧无依赖(本项目不依赖 LLM API) |

# 第七部分:第一周精读清单(7/3–7/6 内完成,按优先级)
1. Todd et al., Function Vectors in Large Language Models(ICLR 2024,arXiv:2310.15213)——提取/注入协议的母本。
2. Hendel et al., In-Context Learning Creates Task Vectors(arXiv:2310.15916)。
3. ICLR 2026 秩限制论文(检索:task vector representative demonstration high-rank ICLR 2026)——失效任务清单与理论视角。
4. ELICIT(检索:In-context learning capability library task vector)与 ATV(检索:Adaptive Task Vectors query)——baseline 与差异句。
5. conceptor 组合转向(OpenReview: From Steering Vectors to Conceptors)——组合算子 baseline。
6. SEAP(arXiv:2503.07605)——聚类现象引用 + KDD 叙事参照;作者有人大资助背景,可发邮件要多任务聚类实验细节。
7. Eldar & Mishali 2009(块稀疏/union of subspaces)+ Yuan & Lin 2006(group lasso)——命题1的引用基座,只读定理陈述。
8. Davis–Kahan sinΘ 的任一现代讲义版——命题2引用。

(注:第 3、4、5 条的精确 arXiv 编号以你检索到的为准,本文档不臆造编号;第 1、2、6 条编号可信度高但入库前请核对一次。)