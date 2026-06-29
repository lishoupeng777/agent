# Gold Dataset Specification v1.0

# 黄金评测集数据集规范

**版本**：v1.0  
**日期**：2026-06-29  
**关联规范**：[Judge Constitution v1.1](judge_constitution.md)  
**用途**：LLM-as-Judge 的 Gold Label 数据集，用于人工一致性实验、Judge Prompt 调优、Spearman/Kappa/ICC 实验、Calibration Set。

---

## 一、数据集定位

本数据集**不是**训练大模型的数据，而是**评估裁判（Judge）本身**的基准数据集。

用途：
- **Calibration Set**：Judge 上线前的校准基准（第九章）
- **Prompt 调优**：Judge Prompt 迭代的评估基准
- **一致性实验**：人工标注员间一致性、人机一致性验证
- **阈值实验**：pass/review/fail 阈值敏感性分析
- **论文实验**：所有定量指标的数据基础

---

## 二、数据集规模

### 2.1 总规模

**总计 200 条**样本。

### 2.2 划分方案

| 集合 | 数量 | 占比 | 用途 |
|---|---|---|---|
| train | 100 条 | 50% | Judge Prompt 调优、DSPy 优化 |
| validation | 40 条 | 20% | 阈值调优、早停、模型选择 |
| test | 60 条 | 30% | 最终评估，**开发过程中严禁查看** |

### 2.3 划分原则

- **分层抽样**：每个错误类别在 train/val/test 中的占比一致
- **难度均衡**：easy/medium/hard 在各集合中均匀分布
- **判定均衡**：pass/review/fail 在各集合中大致均匀（各约 33%）
- **无泄漏**：同一原文的不同变体不能分到不同集合

---

## 三、覆盖方案

### 3.1 错误类别覆盖

共 16 个类别，每类在 train/val/test 中分配如下：

| 类别 | 代码 | train | val | test | 合计 | 占比 | 说明 |
|---|---|---|---|---|---|---|---|
| 良好治理 | GOOD | 8 | 3 | 5 | 16 | 8% | 格式优化、标点统一、完全无改动 |
| 过度清洗-轻微 | OC-LIGHT | 7 | 3 | 4 | 14 | 7% | 删除括号说明、删除修饰语 |
| 过度清洗-严重 | OC-HEAVY | 8 | 3 | 5 | 16 | 8% | 删除具体措施、量化目标、关键信息 |
| 语义漂移 | SEM | 6 | 2 | 4 | 12 | 6% | 主题偏移、因果扭曲、意图改变 |
| 数值误改 | FA-NUM | 8 | 3 | 5 | 16 | 8% | 金额、百分比、数量篡改 |
| 日期误改 | FA-DATE | 5 | 2 | 3 | 10 | 5% | 生效日期、截止日期、时间范围篡改 |
| 机构名称误改 | FA-NAME | 5 | 2 | 3 | 10 | 5% | 法定全称缩写、名称替换 |
| 法律条款删除 | OC-LEGAL | 7 | 3 | 4 | 14 | 7% | 条文引用删除、限定条件删除、兜底条款删除 |
| 删除限定词 | OC-LIMIT | 6 | 2 | 4 | 12 | 6% | "书面"、"正式"、"在...前提下"等限定删除 |
| 删除括号说明 | OC-PAREN | 5 | 2 | 3 | 10 | 5% | 括号内补充说明删除 |
| 表格破坏 | STR-TBL | 6 | 2 | 4 | 12 | 6% | 表格转纯文本、表格行列丢失 |
| 列表破坏 | STR-LST | 6 | 2 | 4 | 12 | 6% | 列表转段落、编号丢失 |
| 可读性下降 | READ | 6 | 2 | 4 | 12 | 6% | 过度缩写、句子不通顺、逻辑断裂 |
| 合理脱敏 | GOOD-PRIV | 4 | 2 | 3 | 9 | 5% | 合理的个人信息遮蔽 |
| 不合理脱敏 | BAD-PRIV | 4 | 2 | 3 | 9 | 5% | 业务信息、技术参数被遮蔽 |
| 综合错误 | MIXED | 9 | 3 | 5 | 17 | 9% | 同一样本存在 2 种以上错误类型 |
| **合计** | | **100** | **36** | **60** | **200** | **100%** | |

### 3.2 难度分布

| 难度 | 定义 | 占比 | 数量 |
|---|---|---|---|
| easy | 错误明显，标注员 100% 一致 | 30% | 60 条 |
| medium | 错误需要对比才能发现 | 50% | 100 条 |
| hard | 错误隐蔽，标注员可能分歧 | 20% | 40 条 |

### 3.3 判定分布

| 判定 | 占比 | 数量 | 说明 |
|---|---|---|---|
| pass | ~33% | ~66 条 | 四维度均 ≥ 0.75 且加权总分 ≥ 0.80 |
| review | ~34% | ~68 条 | 不满足 pass 和 fail |
| fail | ~33% | ~66 条 | 任一维度 ≤ 0.25 或加权总分 < 0.50 |

### 3.4 文本类型分布

| 文本类型 | 占比 | 数量 | 说明 |
|---|---|---|---|
| 政务通告 | 25% | 50 条 | 通知、公告、管理办法 |
| 法律合同 | 20% | 40 条 | 合同、协议、条款 |
| 企业公告 | 20% | 40 条 | 年报、战略规划、会议纪要 |
| 技术文档 | 15% | 30 条 | 系统说明、技术规范 |
| 一般公文 | 20% | 40 条 | 信函、说明、报告 |

---

## 四、JSON Schema

### 4.1 单条样本结构

```json
{
  "id": "gold-train-001",
  "category": "OC-HEAVY",
  "difficulty": "medium",
  "text_type": "政务通告",
  "before_text": "原文内容",
  "after_text": "治理后内容",
  "gold_scores": {
    "semantic_consistency": 0.50,
    "over_cleaning": 0.25,
    "readability_structure": 0.75,
    "factual_accuracy": 0.75
  },
  "overall": {
    "weighted_score": 0.50,
    "verdict": "review"
  },
  "anchors": [
    {
      "rule_id": "OC-03",
      "severity": "major",
      "before_snippet": "预计新增营收15亿元",
      "after_snippet": null,
      "location": {
        "segment_id": "1",
        "start_char": 45,
        "end_char": 55
      },
      "affected_dimension": "over_cleaning",
      "description": "量化目标被删除"
    }
  ],
  "rationale": "核心战略方向保留，但具体措施和量化目标被删除，属于过度清洗",
  "annotation_meta": {
    "annotator": "annotator_01",
    "annotation_time": "2026-06-29T10:30:00",
    "reviewed_by": "annotator_02",
    "review_time": "2026-06-29T14:00:00"
  }
}
```

### 4.2 字段说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| id | string | ✅ | 唯一标识，格式：`gold-{split}-{序号}` |
| category | string | ✅ | 错误类别代码（见第三章表格） |
| difficulty | string | ✅ | 难度：easy / medium / hard |
| text_type | string | ✅ | 文本类型：政务通告/法律合同/企业公告/技术文档/一般公文 |
| before_text | string | ✅ | 治理前原文 |
| after_text | string | ✅ | 治理后文本 |
| gold_scores | object | ✅ | 四维度 Gold Label 分数 |
| gold_scores.semantic_consistency | float | ✅ | 语义一致性，取值 0.00/0.25/0.50/0.75/1.00 |
| gold_scores.over_cleaning | float | ✅ | 过度清洗，取值 0.00/0.25/0.50/0.75/1.00 |
| gold_scores.readability_structure | float | ✅ | 可读性与结构，取值 0.00/0.25/0.50/0.75/1.00 |
| gold_scores.factual_accuracy | float | ✅ | 事实准确性，取值 0.00/0.25/0.50/0.75/1.00 |
| overall | object | ✅ | 综合判定 |
| overall.weighted_score | float | ✅ | 加权总分 |
| overall.verdict | string | ✅ | 判定：pass / review / fail |
| anchors | array | ✅ | 瑕疵锚点列表（可为空数组） |
| anchors[].rule_id | string | ✅ | Rule ID（如 OC-03、FA-01） |
| anchors[].severity | string | ✅ | 严重程度：critical / major / minor |
| anchors[].before_snippet | string | ✅ | 原文片段（被影响的内容） |
| anchors[].after_snippet | string\|null | ✅ | 治理后对应片段（删除时为 null） |
| anchors[].location | object | ✅ | 位置信息 |
| anchors[].location.segment_id | string | ✅ | 段落 ID |
| anchors[].location.start_char | int | ✅ | 起始字符偏移 |
| anchors[].location.end_char | int | ✅ | 结束字符偏移 |
| anchors[].affected_dimension | string | ✅ | 影响的维度 |
| anchors[].description | string | ✅ | 简短描述 |
| rationale | string | ✅ | 整体评分理由 |
| annotation_meta | object | ✅ | 标注元数据 |
| annotation_meta.annotator | string | ✅ | 首次标注员 ID |
| annotation_meta.annotation_time | string | ✅ | 首次标注时间（ISO 8601） |
| annotation_meta.reviewed_by | string | ✅ | 审核员 ID |
| annotation_meta.review_time | string | ✅ | 审核时间（ISO 8601） |

### 4.3 分数约束

- 每个维度分数必须是以下离散值之一：`0.00, 0.25, 0.50, 0.75, 1.00`
- `overall.weighted_score` = semantic × 0.30 + over_cleaning × 0.30 + readability × 0.20 + factual × 0.20
- `overall.verdict` 由分数和阈值自动计算，不允许手动填写不一致的值

---

## 五、目录结构

```
shixun/
├── dataset/
│   ├── train.json              # 训练集（100 条）
│   ├── validation.json         # 验证集（40 条）
│   ├── test.json               # 测试集（60 条，开发期间不查看）
│   ├── calibration.json        # 校准集（50 条，从 train 中抽取）
│   └── schema.json             # JSON Schema 定义文件
├── docs/
│   ├── judge_constitution.md   # 裁判法
│   ├── dataset_specification.md # 本文件
│   └── annotation_guideline.md # 旧版标注规范（参考）
└── python_version/
    └── data/
        └── eval_dataset.json   # 现有 20 条（迁移到 dataset/ 后废弃）
```

---

## 六、数据来源建议

### 6.1 真实文本来源（建议 40%，约 80 条）

| 来源 | 适用类别 | 获取方式 |
|---|---|---|
| 政府网站公开通告 | 政务通告类 | 爬取 gov.cn 公开文件 |
| 裁判文书网 | 法律合同类 | 公开法律文书 |
| 企业年报/公告 | 企业公告类 | 巨潮资讯网等公开信息 |
| 技术文档 | 技术文档类 | GitHub 开源项目文档 |
| 公文范文 | 一般公文类 | 公文写作教材/范文库 |

**使用规则**：
- 真实文本必须脱敏后使用（隐去真实人名、身份证号、手机号）
- 治理后文本由人工改写，模拟真实治理场景
- 每条真实文本必须标注来源

### 6.2 人工构造（建议 30%，约 60 条）

| 适用场景 | 说明 |
|---|---|
| 边界案例 | 测试 Judge 在模糊场景下的判断能力 |
| 极端案例 | 测试 Judge 在极端错误下的表现 |
| 组合错误 | 同一样本存在多种错误类型 |
| 合理治理 | 测试 Judge 不会误判合理操作 |

**构造原则**：
- 人工构造的文本必须符合真实文本的语言风格
- 不能过于明显或过于隐蔽，要符合实际治理场景
- 每条必须有明确的 Gold Label 依据

### 6.3 AI 辅助生成（建议 30%，约 60 条）

| 适用场景 | 说明 |
|---|---|
| 基础错误模板 | 从现有 20 条扩展，生成同类变体 |
| 错误注入 | 对真实文本程序化注入特定错误 |
| 翻译变体 | 用 WMT-MQM 数据集的中文样本适配 |

**生成流程**：
1. 选择一个基础样本
2. 按目标错误类型注入错误
3. 人工审核并修正 Gold Label
4. 确保生成样本不与现有样本重复

**质量控制**：
- AI 生成的样本必须经过人工审核
- 人工审核通过率低于 80% 的生成方式应停止使用
- 最终 Gold Label 必须由人工确认

---

## 七、Calibration Set 设计

### 7.1 抽取规则

从 train.json 中随机抽取 50 条，构成 calibration.json。

**要求**：
- 覆盖全部 16 个错误类别
- 覆盖全部 5 个分值等级
- 覆盖 pass/review/fail 三类判定
- 难度分布与 train 一致

### 7.2 使用规则

- Calibration Set 在项目周期内**不得修改**
- 任何 Judge 进入生产前必须在 Calibration Set 上通过校准（Spearman ≥ 0.8）
- Prompt 变更后必须重新校准

---

## 八、质量保证

### 8.1 标注流程

每条样本必须经过以下流程：

1. **首标**：标注员 A 按 Judge Constitution 第七章流程标注
2. **审核**：标注员 B 独立审核，标注同意/不同意及理由
3. **仲裁**：首标与审核不一致时，由第三位标注员仲裁
4. **确认**：最终 Gold Label 由项目负责人确认

### 8.2 一致性要求

- 标注员间 Spearman ≥ 0.8
- 加权 Kappa ≥ 0.6
- 如不一致率 > 20%，暂停标注，重新培训

### 8.3 版本管理

- 数据集有版本号（v1.0, v1.1, ...）
- 每次修改必须记录变更日志
- 论文实验必须注明使用的数据集版本

---

## 九、与现有数据集的关系

| 数据集 | 状态 | 说明 |
|---|---|---|
| python_version/data/eval_dataset.json | 现有 20 条 | 迁移到 dataset/ 后作为 train 的一部分 |
| data/eval_dataset_factual.json | 现有 96 条 | 不直接使用，可参考结构 |
| data/eval_dataset_summeval.json | 现有 60 条 | 不直接使用，可参考结构 |

**迁移计划**：
1. 现有 20 条按新 Schema 重新标注
2. 作为 train.json 的前 20 条
3. 原 eval_dataset.json 标记为 deprecated

---

## 十、实施计划

| 阶段 | 任务 | 产出 |
|---|---|---|
| Phase 1 | 设计规范 | 本文件（dataset_specification.md） |
| Phase 2 | 构造 train 样本 | train.json（100 条） |
| Phase 3 | 构造 val/test 样本 | validation.json + test.json |
| Phase 4 | 人工审核与标注 | Gold Label 确认 |
| Phase 5 | 抽取 Calibration Set | calibration.json |
| Phase 6 | 质量验证 | 一致性报告 |
