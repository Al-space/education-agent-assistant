# 教育智能题库助手 —— RAG 设计文档

| 项目 | 内容 |
|---|---|
| 文档版本 | V1.0 |
| 状态 | 设计稿 |
| 适用范围 | 知识库上传、解析、切片、索引、检索全链路 |
| 关联文档 | 教育智能题库助手-开题报告、开发环境搭建指南 |

---

## 1. 目标与边界

### 1.1 目标

1. 支持教师上传教材、讲义、题库（PDF / DOCX / MD / TXT），构建可复用的知识库。
2. 将资料切分为带来源信息的检索单元，支持按学科、年级、文档、内容类型过滤。
3. 对用户问题执行向量 + 关键词混合检索，返回可用于引用的原文片段。
4. 证据不足时明确返回“无可靠依据”，交由上层 Agent 决定是否联网补充。
5. 所有回答均可回溯到具体文档与页码。

### 1.2 不在本期范围

- OCR（扫描版 PDF 的图片文字识别）
- Rerank 模型二次排序
- 增量索引与索引热更新
- 多知识库联邦检索
- 公式/图表的语义解析

以上能力预留接口，但不进入第一版实现。

---

## 2. 总体架构

```
上传文件
   │
   ▼
[loader] 解析 ──► [splitter] 切片 ──► [embedding] 向量化
   │                                        │
   │                                        ▼
   │                                  [indexer] 建索引
   │                                        │
   ▼                                        ▼
data/uploads/{kb_id}/            data/indexes/{kb_id}/
                                       vectors.faiss
                                       chunks.jsonl
                                       bm25.pkl
                                       manifest.json
                                            │
用户问题 ──► [retriever] 混合检索 ◄─────────┘
                │
                ▼
         Top-K 片段 + 来源 + 是否证据不足
                │
                ▼
        Agent / 知识库工具 knowledge_search
```

RAG 只负责“找依据”，不负责生成讲解。生成逻辑在 Agent 层，两者通过 `RetrievedChunk` 结构解耦。

---

## 3. 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 中间数据格式 | LangChain `Document` | 与生态兼容，减少自定义结构 |
| 切片策略 | 题目感知优先，递归兜底 | 题库必须保证题干/选项/答案同块 |
| 检索方式 | FAISS 向量 + BM25 各 0.5 | 兼顾语义召回与术语、公式、编号精确匹配 |
| 索引更新 | 每次全量重建 | 第一版避免增量与删除的一致性复杂度 |
| 向量索引 | `IndexFlatIP` + L2 归一化 | 数据量级（万级切片）下无需 ANN，余弦更直观 |
| 中文分词 | jieba + 停用词表 | BM25 对中文必须分词，否则排序失真 |
| 证据不足 | 分数阈值 + 空结果双判定 | 防止把不相关内容硬塞给模型 |
| 引用 | metadata 携带文件名与页码 | 满足“可追溯”验收要求 |

---

## 4. 数据契约

### 4.1 文档元数据

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `doc_id` | string | 是 | 文档唯一 ID，建议 `doc_` + 短哈希 |
| `chunk_id` | string | 是 | `{doc_id}-p{page}-c{序号}` |
| `file_name` | string | 是 | 原始文件名 |
| `file_type` | string | 是 | pdf / docx / md / txt |
| `page` | int | 是 | 页码，从 1 开始；无分页格式记 1 |
| `subject` | string | 否 | 学科，来自知识库或上传参数 |
| `grade` | string | 否 | 学段/年级 |
| `section` | string | 否 | 章节标题（能识别时填写） |
| `content_type` | string | 是 | `question` / `explanation` / `text` |
| `char_count` | int | 是 | 切片字符数 |
| `tokens` | list[string] | 是 | jieba 分词结果，供 BM25 重建 |

### 4.2 切片示例（`chunks.jsonl` 单行）

```json
{
  "chunk_id": "doc_a1b2-p12-c03",
  "text": "3. 已知函数 f(x)=x^2-4x+3，求其顶点坐标。\nA. (2,-1)\nB. (-2,1)\nC. (2,1)\nD. (1,2)\n答案：A\n解析：配方得 f(x)=(x-2)^2-1，故顶点为 (2,-1)。",
  "metadata": {
    "doc_id": "doc_a1b2",
    "file_name": "二次函数讲义.pdf",
    "file_type": "pdf",
    "page": 12,
    "subject": "数学",
    "grade": "高一",
    "section": "二次函数",
    "content_type": "question",
    "char_count": 118
  },
  "tokens": ["已知", "函数", "顶点", "坐标", "配方", "二次函数"]
}
```

### 4.3 索引目录

```
data/indexes/{kb_id}/
├── vectors.faiss
├── chunks.jsonl
├── bm25.pkl
└── manifest.json
```

`manifest.json` 字段：

```json
{
  "kb_id": "kb_001",
  "embedding_model": "BAAI/bge-m3",
  "dimension": 1024,
  "chunk_count": 1280,
  "files": [{"doc_id": "doc_a1b2", "sha256": "...", "page_count": 24}],
  "built_at": "2026-10-26T10:30:00+08:00",
  "schema_version": 1
}
```

加载索引前必须校验 `embedding_model` 与 `dimension`；不一致直接报错，禁止跨模型检索。

---

## 5. 模块设计

### 5.1 [loader.py](../app/rag/document_loader.py)

```python
def load_document(file_path: Path, doc_id: str, extra_meta: dict) -> list[Document]
```

职责：

1. 按扩展名分派解析器。
2. 文本清洗与页眉页脚、水印去除。
3. 返回逐页（PDF）或整篇（MD/TXT）的 `Document` 列表。

解析规则：

- **PDF**：PyMuPDF `page.get_text("text")`，逐页读取；空白页跳过；记录 `page`。
- **DOCX**：正文段落按顺序读取；表格按行拼为 `| 单元格 | 单元格 |`；分页信息不可得时 `page=1`。
- **MD/TXT**：优先 `utf-8`，失败回退 `gbk`，仍失败用 `errors="replace"` 并记录警告。

清洗规则：

- 行尾无标点且下一行不以题号/序号开头时，拼接断行。
- 统计每行出现页数，出现率 > 50% 且长度 < 30 的行判定为页眉页脚并删除。
- 同一短行在 ≥ 3 页重复出现，判定为水印并删除。
- 连续空行压缩为一个。

验收：PDF、DOCX、MD 各解析一份，页数、字符数正确，前 3 页文本连贯。

### 5.2 [splitter.py](../app/rag/text_splitter.py)

```python
def split_documents(docs: list[Document], chunk_size: int = 600,
                    chunk_overlap: int = 100) -> list[Document]
```

题目边界正则（可配置）：

```text
^\s*第?\s*\d+\s*[题、.．]
^\s*[（(]\s*\d+\s*[)）]
^\s*[一二三四五六七八九十]+\s*[、.．]
```

流程：

1. 统计每页题号命中数。
2. 命中 ≥ 3：按题号切分，保证题干、选项、答案、解析在同一块。
3. 单题超过 800 字符：按 `\n\n` → `\n` → `。` → `；` 递归切分，重叠 100。
4. 命中 < 3：使用 `RecursiveCharacterTextSplitter`，分隔符顺序 `\n\n`、`\n`、`。`、`；`、`，`、空格。
5. 丢弃 < 30 字符的碎片，但包含“答案/解析/解：”的碎片保留。
6. 写入 `chunk_id`、`content_type`、`char_count`。

验收：抽 3 道题检查题干、选项、答案、解析未被拆散；切片长度集中在 600–800。

### 5.3 embedding 服务（[services/embedding.py](../app/services/embedding.py)）

```python
def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]
def embed_query(text: str) -> list[float]
```

约定：

- 通过 OpenAI 兼容接口调用，模型由 `EMBEDDING_MODEL` 指定。
- 批量请求，失败重试 2 次，仍失败则中止构建并保留旧索引。
- 返回向量维度必须与 `manifest.dimension` 一致。
- 构建与查询必须使用同一模型，禁止混用。

### 5.4 [indexer.py](../app/rag/hybrid_retriever.py)

```python
def build_index(kb_id: str, chunks: list[Document]) -> BuildResult
def load_index(kb_id: str) -> IndexBundle
```

向量：

1. 批量 embedding，转 `float32`。
2. L2 归一化后写入 `faiss.IndexFlatIP`。
3. 记录维度与切片数到 manifest。

BM25：

1. jieba 分词，过滤停用词、单字、纯数字与标点。
2. 分词结果写入 `chunks.jsonl` 的 `tokens` 字段。
3. 用 `rank_bm25.BM25Okapi` 在内存重建，`bm25.pkl` 仅保存对象。

写入顺序：先写 `*.tmp`，再 `os.replace` 替换正式文件；四件套全部成功后才更新 `manifest.json`。

验收：重新 `load_index` 后 `ntotal == chunk_count`；manifest 字段完整；中断构建不破坏旧索引。

### 5.5 [retriever.py](../app/rag/bm25_retriever.py)

```python
@dataclass
class RetrievedChunk:
    text: str
    score: float
    vector_score: float
    bm25_score: float
    metadata: dict

@dataclass
class RetrieveResult:
    chunks: list[RetrievedChunk]
    insufficient: bool

def search(kb_id: str, query: str, top_k: int = 6,
           filters: dict | None = None) -> RetrieveResult
```

检索流程：

1. `fetch_k = top_k * 4`。
2. 向量召回：查询向量化 → FAISS 取 `fetch_k`。
3. 关键词召回：jieba 分词 → BM25 `get_scores` → 取 `fetch_k`。
4. 两路分数分别 min-max 归一化；单条候选该路记 1.0；分母为 0 时整体记 0。
5. 加权：`score = 0.5 * vector_score + 0.5 * bm25_score`。
6. 合并去重：同一 `doc_id`、相邻页码且文本 Jaccard > 0.8 时保留高分项。
7. 过滤：`subject`、`grade`、`doc_id`、`content_type` 在合并前生效。
8. 取 `top_k`，按分数降序。
9. 判定：候选为空或最高分 < `MIN_SCORE`（初值 0.15，需标定）时 `insufficient=True`。

验收：20 条标注查询中 Recall@5 ≥ 0.8；无答案问题能被判为证据不足。

### 5.6 [kb_service.py](../app/services/kb_service.py)

```python
def create_kb(name: str, subject: str, grade: str) -> KnowledgeBase
def list_kbs() -> list[KnowledgeBase]
def delete_kb(kb_id: str) -> None
def ingest(kb_id: str, file_path: Path) -> IngestResult
def search_kb(kb_id: str, query: str, top_k: int = 6) -> RetrieveResult
```

`ingest` 顺序：

1. 校验扩展名白名单与 `MAX_UPLOAD_MB`。
2. 计算 `sha256`，与已有文档比对去重。
3. 复制到 `data/uploads/{kb_id}/{doc_id}_{原文件名}`。
4. 解析 → 切片 → embedding → 建索引。
5. 写 SQLite：`document` 记录状态、页数、切片数。
6. 任一步失败：文档标记 `failed`，保留错误原因，不更新索引。

重建策略：同一知识库新增或删除文档后，合并全部有效文档的切片后全量重建，禁止只追加。

### 5.7 接口（[api/kb_routes.py](../app/api/kb_routes.py)）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/kb` | 新建知识库 |
| GET | `/api/kb` | 知识库列表 |
| DELETE | `/api/kb/{kb_id}` | 删除知识库及索引 |
| POST | `/api/kb/{kb_id}/documents` | 上传文档并触发索引 |
| GET | `/api/kb/{kb_id}/documents` | 文档列表与状态 |
| POST | `/api/kb/{kb_id}/search` | 调试检索，返回分数与来源 |

上传接口返回 `task_id`，索引状态通过文档列表轮询；第一版不做 WebSocket。

---

## 6. 关键算法细节

### 6.1 混合评分

```text
norm(x) = (x - min) / (max - min)        # 单候选或 max == min 时记 1.0
score   = w_vec * norm(vec_sim) + w_bm25 * norm(bm25_score)
```

约束：`w_vec + w_bm25 = 1`，默认各 0.5；权重来自 `.env`，启动时校验和为 1。

### 6.2 去重

```text
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

按分词集合计算，同文档且 `|page_a - page_b| <= 1` 时比较；超过 0.8 视为重复。

### 6.3 证据不足判定

满足任一条件即 `insufficient=True`：

- 无候选
- 最高分 < `MIN_SCORE`
- 最高分与第二名差距 < 0.02 且最高分 < 0.3（可选，标定后启用）

上层 Agent 收到 `insufficient` 后，才允许调用联网搜索工具。

---

## 7. 配置项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_MODEL` | 必填 | 向量模型 ID |
| `CHUNK_SIZE` | 600 | 切片长度 |
| `CHUNK_OVERLAP` | 100 | 重叠长度 |
| `RETRIEVAL_TOP_K` | 6 | 返回条数 |
| `VECTOR_WEIGHT` | 0.5 | 向量权重 |
| `BM25_WEIGHT` | 0.5 | 关键词权重 |
| `MIN_SCORE` | 0.15（待新增） | 拒答阈值 |
| `UPLOAD_DIR` | `./data/uploads` | 上传目录 |
| `INDEX_DIR` | `./data/indexes` | 索引目录 |
| `MAX_UPLOAD_MB` | 50 | 单文件上限 |

`MIN_SCORE` 尚未写入 `.env.example`，实现检索时补充。

---

## 8. 异常与边界

| 场景 | 处理 |
|---|---|
| 扫描版 PDF 无文字层 | 解析结果为空 → 文档标记 `no_text`，提示需 OCR |
| 加密 PDF | 尝试空密码，失败则标记 `encrypted` |
| 非 UTF-8 文本 | 回退 gbk，再失败用 replace 并记录警告 |
| 超长文档 | 切片数超过 5000 时告警，不阻断 |
| 索引损坏 | `load_index` 抛错，提示重建 |
| 维度不一致 | 拒绝加载，要求重建 |
| 并发写索引 | 同一 `kb_id` 加文件锁，串行构建 |
| 空查询 / 超长查询 | 空查询拒绝；超过 500 字截断并记录 |

---

## 9. 性能与容量

| 指标 | 目标 |
|---|---|
| 单文档（50 页 PDF）解析 + 切片 | < 10 秒 |
| 1000 切片向量化（批量 32） | 以模型接口耗时为准，失败可续跑 |
| 单次检索（1000 切片，FlatIP） | < 200 ms（不含 embedding 网络耗时） |
| 索引体积 | 约为 `切片数 × 维度 × 4 字节` |

容量上限：单知识库建议 ≤ 2 万切片；超过后考虑 IVF 或分库。

---

## 10. 测试与验收

### 10.1 不依赖 API 的测试

- 固定维度假向量替换 embedding，验证“切片 → 索引 → 检索”链路。
- 空 PDF、扫描版 PDF、加密 PDF、GBK 文本、超大文件各一条。
- 构建中断后重建，旧索引仍可用。
- 同文档重复上传，sha256 去重生效。

### 10.2 接入真实模型后的测试

- 真实 embedding 建索引，维度与 manifest 一致。
- 20 条标注查询：Recall@5 ≥ 0.8，MRR ≥ 0.6。
- 混合权重对比：0.5/0.5、0.7/0.3、0.3/0.7 各跑一次，记录 Recall。
- 拒答测试：5 条知识库外问题，`insufficient` 命中 ≥ 4 条。

### 10.3 验收清单

- [ ] 四种格式均可解析
- [ ] 题目类切片保持完整性
- [ ] 索引四件套齐全且可重载
- [ ] 检索结果含文件名、页码、分数
- [ ] 证据不足可识别
- [ ] 重复上传不重复建索引
- [ ] 索引导出与恢复可用

---

## 11. 实施顺序

1. `loader.py` + 清洗，用真实教材验证。
2. `splitter.py` + 题目边界，人工抽查切片。
3. 假 embedding 打通 `indexer` 与 `retriever`，先验证排序逻辑。
4. 接入真实 embedding，标定 `MIN_SCORE`。
5. 封装 `kb_service` 与知识库接口。
6. 补测试与评估脚本。

前两步不需要 API 密钥，可立即开始。

---

## 12. 待确认事项

1. 首批资料的学科与格式，是否需要支持 PPTX。
2. 题库是否包含图片题，若包含需先确定图片存储与引用方式。
3. 单知识库容量预期，决定是否需要 ANN 索引。
4. `MIN_SCORE` 是否按学科分别标定。
