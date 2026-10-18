"""内容过滤：去重 + LLM 语义判定。

所有判定都交给 LLM，唯一不动用模型的是「标题完全重复」这种确定性去重。
LLM 连续重试仍调不通时一律保留条目——宁可多收，不可漏课。
"""

import html
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from llm import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 硬编码策略

# 每次请求最多提交多少条待判定条目
BATCH_SIZE = 40

SYSTEM_PROMPT = """
你是播客 feed 的内容审核助手。待判定条目来自「CLEC 投資理財頻道」——一个投资理财教育频道，从 X(Twitter) 与 YouTube 同步音频。

【默认判定为课程内容】
频道的正式内容体系（以下都属于课程，必须保留）：
- 五位数字编号 + 「短篇」：日常答疑与案例（如「00693 短篇【...】」）。即使标题谈的是家庭关系、健康、人生感悟、心理状态，它仍是该频道的编号课程，必须保留。
- 五位编号 + 【标题】：每周长篇（如「00574【...】」）。
- 「專題」「投資第一堂課」「講座」「貼文」等栏目。
- 标题以「簡報資料」「講義」「短連結」「PCloud」等结尾的，只是附带了讲义链接，**仍然是正常课程，必须保留**。同理，含 emoji、含 t.co 短链、含「Follow 我們的 Facebook」等社媒尾巴的，也都必须保留。
- 编号写错（如「0042」而不是五位数）不影响判定，看标题正文实质内容。

【仅以下几类判为非课程】
只有同时满足「没有任何课程编号 / 栏目名」且「与投资理财教育实质无关」时，才判 course=false：
1. 纯生活分享：节日祝福、烟花视频、旅游、晒娃、纯情绪发泄。
2. 为其他账号或其他产品做的互推/宣传（如「投資翻譯官 提供好的教育影片，值得大家支持觀看」）。
3. 纯活动通知：直播即将开始、临时停播、抽奖、报名、开奖。
4. 无实质内容的单句闲聊或口水话。

【关键约束】
- 只要标题里出现三位以上连续数字的课程编号，一律 course=true。
- 只要提及具体的投资标的、配置比例、税务、退休、资产、现金流等理财主题，一律 course=true。
- 拿不准 → course=true。误杀课程比多收一条更严重。

只输出 JSON，不要解释文字，不要 markdown 代码块。
格式：{"results":[{"i":0,"course":true,"reason":"简短理由"}, ...]}
必须对输入里每一个 i 都输出一条结果。
""".strip()


# ---------------------------------------------------------------- 标题归一化


def normalize_title(title: str) -> str:
    """去掉日期、括号符号与所有空白，用于重复检测。"""
    t = html.unescape(title or "")
    # 去掉「日期：2026年1月1日」「2026年1月1日」「2026年2026年1月7日」等
    # 注意：前缀「日期：」必须整体匹配，否则会残留单字（早期的字符类写法就会留下「日」）
    t = re.sub(r"(?:日期\s*[:：]\s*)?\s*20\d{2}\s*年"
               r"(?:\s*20\d{2}\s*年\s*)?\s*\d{1,2}\s*月\s*\d{1,2}\s*日", "", t)
    t = re.sub(r"(?:日期\s*[:：]\s*)?$", "", t)
    # 去掉各种括号与标点
    t = re.sub(r"[\[\]【】〖〗「」『』()（）]", "", t)
    t = re.sub(r"[\s\u3000\xa0]+", "", t)
    return t.strip()


# ---------------------------------------------------------------- 去重


class Deduper:
    """按归一化标题去重，用于拦截同一条推文被重复抓取。"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._seen: Dict[str, str] = {}

    def is_duplicate(self, title: str) -> Optional[str]:
        if not self.enabled:
            return None
        key = normalize_title(title)
        if not key:
            return None
        return self._seen.get(key)

    def remember(self, title: str, ident: str):
        if not self.enabled:
            return
        key = normalize_title(title)
        if key:
            self._seen.setdefault(key, ident)


# ---------------------------------------------------------------- LLM


class LLMClassifier:
    """把「是否为课程内容」的判定逻辑架在通用 LLMClient 之上。

    本类只负责：拼提示词、切批、把模型输出映射回条目 id、以及调用失败时保留条目。
    重试、超时、JSON 解析与容错都交给 llm.LLMClient。
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient.from_env()
        self.batch_size = BATCH_SIZE
        self.system_prompt = SYSTEM_PROMPT

    @property
    def enabled(self) -> bool:
        return self.client.available

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def usage(self):
        return self.client.usage

    def classify(self, items: List[dict]) -> Dict[str, Tuple[bool, str]]:
        """items: [{"id": str, "title": str}] -> {id: (is_course, reason)}"""
        if not items:
            return {}

        # 没有密钥：直接全部保留
        if not self.enabled:
            logger.warning(
                f"LLM API key not set; keeping all {len(items)} item(s) unfiltered."
            )
            return {it["id"]: (True, "llm unavailable: no api key") for it in items}

        logger.info(f"Classifying {len(items)} item(s) with LLM ({self.model})...")

        try:
            payloads = self.client.batch_json(
                items,
                self._build_messages,
                batch_size=self.batch_size,
                expect=dict,
            )
        except Exception as e:
            # 重试几次都调不通：全部保留
            logger.error(
                f"LLM classification failed entirely ({type(e).__name__}: {e}); "
                f"keeping all {len(items)} item(s)."
            )
            return {
                it["id"]: (True, f"llm error: {type(e).__name__}")
                for it in items
            }

        results: Dict[str, Tuple[bool, str]] = {}
        for chunk_start in range(0, len(items), self.batch_size):
            chunk = items[chunk_start : chunk_start + self.batch_size]
            # batch_json 约定返回与输入等长，但外部注入/异常时可能不一致
            payload = (
                payloads[chunk_start]
                if isinstance(payloads, list) and chunk_start < len(payloads)
                else None
            )
            verdicts = self._parse_results(payload, len(chunk))
            for i, it in enumerate(chunk):
                verdict = verdicts.get(i)
                if verdict is None:
                    # 模型漏判这一条：保守保留
                    logger.warning(
                        f"LLM gave no verdict for {it['id']} ({it['title'][:40]}); keeping it."
                    )
                    verdict = (True, "llm: no verdict")
                results[it["id"]] = verdict
        return results

    def _build_messages(self, chunk: List[dict]) -> List[dict]:
        payload_items = [{"i": i, "title": it["title"]} for i, it in enumerate(chunk)]
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": "请逐条判定下列条目是否为课程内容，返回 JSON：\n"
                + json.dumps(payload_items, ensure_ascii=False, indent=1),
            },
        ]

    @staticmethod
    def _parse_results(payload: Optional[dict], expected: int) -> Dict[int, Tuple[bool, str]]:
        """把 {"results":[{"i":0,"course":true,"reason":"..."}]} 转成 {0: (True, "...")}。"""
        if not isinstance(payload, dict):
            return {}
        entries = payload.get("results")
        if not isinstance(entries, list):
            logger.warning(f"LLM payload missing 'results' list: {str(payload)[:200]}")
            return {}

        out: Dict[int, Tuple[bool, str]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("i"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < expected):
                continue
            course = entry.get("course")
            if isinstance(course, str):
                course = course.strip().lower() in ("true", "yes", "1", "是")
            if not isinstance(course, bool):
                continue
            out[idx] = (course, str(entry.get("reason") or "").strip()[:120])
        return out


# ---------------------------------------------------------------- 门面


class ContentFilter:
    """把去重与 LLM 判定串起来的统一入口。"""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        dedup_enabled: bool = True,
    ):
        self.deduper = Deduper(dedup_enabled)
        self.llm = LLMClassifier(client=client)

    def seed_titles(self, titles: List[str]):
        """把已在 feed 里的标题灌进去重器，避免历史条目重复计入。"""
        for t in titles:
            self.deduper.remember(t, "existing")

    def filter_items(self, items: List[dict]) -> List[dict]:
        """items: [{"id","title"}]，返回其中判定为课程内容的条目（保持原顺序）。

        被去重拦下的条目直接删除；其余条目必须拿到 LLM 结论——
        拿不到（漏判 / 无密钥 / 服务失败）时一律保留。
        """
        pending: List[dict] = []
        dropped_by_dedup: set = set()

        for it in items:
            ident = it["id"]
            title = it.get("title") or ""

            dup_of = self.deduper.is_duplicate(title)
            if dup_of is not None:
                logger.info(f"[dedup] Skipping {ident} — same title as {dup_of}: {title[:60]}")
                dropped_by_dedup.add(ident)
                continue

            pending.append(it)
            self.deduper.remember(title, ident)

        verdicts: Dict[str, Tuple[bool, str]] = {}
        if pending:
            verdicts.update(self.llm.classify(pending))

        kept: List[dict] = []
        for it in items:
            ident = it["id"]
            if ident in dropped_by_dedup:
                continue

            verdict = verdicts.get(ident)
            if verdict is None:
                # 理论上不该发生（classify 已对每个 pending 给结论），
                # 但绝不能因此静默丢弃条目
                logger.warning(f"[keep] {ident} — no verdict obtained; keeping anyway")
                kept.append(it)
                continue

            is_course, reason = verdict
            if is_course:
                logger.info(f"[keep] {ident} — {reason or 'n/a'} | {(it.get('title') or '')[:70]}")
                kept.append(it)
            else:
                logger.info(
                    f"[drop] {ident} — {reason or 'n/a'} | {(it.get('title') or '')[:70]}"
                )

        logger.info(f"Content filter: kept {len(kept)}/{len(items)} item(s).")
        return kept
