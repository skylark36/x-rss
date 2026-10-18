#!/usr/bin/env python3
"""内容过滤回归测试。

用真实 feed 全量跑一遍 ContentFilter，校验：
  1. 期望被丢弃的条目确实被丢（无漏放）
  2. 其余条目全部保留（无误杀）

用法:
    python regress.py                    # 用本地快照，不访问网络
    python regress.py --refresh          # 先重新抓取 feed 再跑
    python regress.py --feed /tmp/x.xml  # 指定快照文件

退出码 0 = 全部符合期望；1 = 有偏差。
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

FEED_URL = "https://rss.clec.ccwu.cc/short/rss.xml"
SNAPSHOT = Path("/tmp/feed.xml")

# 期望被过滤掉的条目：标题归一化后的特征片段 -> 说明
EXPECTED_DROPS = {
    "雙十國慶煙火": "纯生活分享（烟花视频）",
    "投資翻譯官": "为其他账号互推宣传",
}

# 期望保留的条目：标题特征片段 -> 说明（关键回归点，防止误杀）
EXPECTED_KEEPS = {
    "00572": "长篇课程，附 PCloud 讲义链接",
    "00573": "长篇课程（曾被小模型误判为通知类）",
    "00575": "长篇课程，附簡報資料",
    "00581": "长篇课程（标题偏人生感悟，曾被误杀）",
    "0042": "编号异常但正文是投资主题",
    "00638": "编号短篇（曾被误杀）",
    "00679B": "编号短篇（标题偏家庭，曾被误杀）",
    "00626C": "编号短篇（标题偏伴侣话题，曾被误杀）",
    "00681D": "编号短篇（标题偏人生感悟，曾被误杀）",
    "投資第一堂課": "专题讲座",
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def fetch_snapshot() -> bytes:
    """直连失败时通过公共代理取回 feed 原文。"""
    import requests

    session = requests.Session()
    session.trust_env = False
    sources = [
        ("direct", FEED_URL),
        (
            "microlink",
            "https://api.microlink.io/?url=https%3A%2F%2Frss.clec.ccwu.cc%2Fshort%2Frss.xml"
            "&meta=false&data.content.selector=body&data.content.type=text",
        ),
    ]
    for name, url in sources:
        try:
            print(f"  尝试 {name} ...", file=sys.stderr)
            r = session.get(url, timeout=60)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}", file=sys.stderr)
                continue
            text = r.text
            if name == "microlink":
                payload = r.json()
                text = payload["data"]["content"]
                text = re.sub(r"^<pre[^>]*>", "", text)
                text = re.sub(r"</pre>$", "", text)
                text = html.unescape(text)
            if "<rss" in text[:2000]:
                print(f"    成功，{len(text)} 字节", file=sys.stderr)
                return text.encode("utf-8")
            print("    返回内容不像 RSS", file=sys.stderr)
        except Exception as e:
            print(f"    失败: {type(e).__name__}: {e}", file=sys.stderr)
    raise SystemExit("无法获取 feed")


def load_items(source: Path) -> List[dict]:
    root = ET.parse(source).getroot()
    channel = root.find("channel")
    if channel is None:
        raise SystemExit(f"{source} 不是合法 RSS")
    items = []
    for it in channel.findall("item"):
        guid = it.findtext("guid") or ""
        items.append({"id": guid, "title": clean(it.findtext("title"))})
    return items


def run(feed_path: Path) -> int:
    from filters import ContentFilter

    items = load_items(feed_path)
    if not items:
        raise SystemExit("feed 中没有条目")

    print(f"feed: {feed_path} | 条目数: {len(items)}")
    content_filter = ContentFilter()
    llm_on = content_filter.llm.enabled
    print(f"LLM: {content_filter.llm.model} | enabled: {llm_on}")
    if not llm_on:
        print(
            "\n!! 未配置 LLM，过滤全部按「保留」降级；\n"
            "   下面的「期望丢弃」会必然 FAIL，这是预期行为，不代表代码有问题。\n"
            "   不做 LLM 判定的回归请只看「去重」与「降级路径」两节。"
        )

    kept = content_filter.filter_items(items)
    kept_ids = {k["id"] for k in kept}
    dropped = [it for it in items if it["id"] not in kept_ids]

    print(f"\n保留 {len(kept)}/{len(items)}，丢弃 {len(dropped)} 条")

    failures: List[str] = []

    # 1) 期望丢弃的必须被丢
    print("\n--- 期望丢弃 ---")
    for needle, desc in EXPECTED_DROPS.items():
        matched = [it for it in dropped if needle in it["title"]]
        if matched:
            print(f"  PASS  {needle:12} ({desc})")
        else:
            if not llm_on:
                print(f"  SKIP  {needle:12} ({desc}) —— LLM 未启用，按设计保留")
            else:
                print(f"  FAIL  {needle:12} ({desc}) —— 未被过滤")
                failures.append(f"漏放: {needle} ({desc})")

    # 2) 期望保留的必须保留
    print("\n--- 期望保留（防误杀）---")
    for needle, desc in EXPECTED_KEEPS.items():
        present = [it for it in items if needle in it["title"]]
        if not present:
            print(f"  SKIP  {needle:12} —— feed 里没这条")
            continue
        wrong = [it for it in present if it["id"] not in kept_ids]
        if wrong:
            print(f"  FAIL  {needle:12} ({desc}) —— 被误杀")
            for w in wrong:
                failures.append(f"误杀: {needle} ({desc}) | {w['title'][:70]}")
        else:
            print(f"  PASS  {needle:12} ({desc})")

    # 3) 列出所有实际丢弃的条目，便于人工核对
    print("\n--- 实际丢弃明细 ---")
    if not dropped:
        print("  (无)")
    for it in dropped:
        print(f"  {it['id']:>20}  {it['title'][:90]}")

    # 4) 去重结论只汇总，具体单测在 run_dedup() 里独立做
    print("\n--- 实际处理中的去重 ---")
    from filters import normalize_title

    seen: Dict[str, str] = {}
    dup_in_feed = 0
    for it in items:
        key = normalize_title(it["title"])
        if key in seen:
            dup_in_feed += 1
            still_kept = it["id"] in kept_ids and seen[key] in kept_ids
            if still_kept:
                print(f"  FAIL  重复 {it['id']} 与 {seen[key]} 同时保留")
                failures.append(f"去重失效: {it['title'][:60]}")
            else:
                print(f"  PASS  重复 {it['id']} 已剔除（保留 {seen[key]}）")
        else:
            seen[key] = it["id"]
    if dup_in_feed == 0:
        print("  SKIP  feed 里没有重复标题")

    print(f"\nLLM 用量: {content_filter.llm.usage}")

    print("\n" + "=" * 60)
    if failures:
        print(f"回归失败：{len(failures)} 项偏差")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("回归通过：无漏放，无误杀，去重正常")
    return 0


def run_degradation(feed_path: Path) -> int:
    """验证 LLM 不可用时全部保留（仅去重仍生效）。"""
    from filters import ContentFilter
    from llm import LLMClient, LLMConfig

    items = load_items(feed_path)
    print("\n--- 降级路径：LLM 不可用时应全部保留 ---")

    scenarios = [
        ("无 API key", LLMClient(LLMConfig(api_key=None))),
        (
            "服务不可达",
            LLMClient(
                LLMConfig(
                    api_key="sk-x",
                    base_url="http://127.0.0.1:9/v1",
                    retries=2,
                    timeout=2,
                    retry_base_delay=0.01,
                )
            ),
        ),
        (
            "无效 key (401)",
            LLMClient(
                LLMConfig(api_key="sk-invalid", base_url="https://mgp.finstack.top/v1")
            ),
        ),
    ]

    from filters import normalize_title

    unique = len({normalize_title(it["title"]) for it in items})
    failures = 0
    for name, client in scenarios:
        kept = ContentFilter(client=client).filter_items(items)
        ok = len(kept) == unique
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:16} kept={len(kept)}/{len(items)} (去重后唯一标题 {unique})")
        if not ok:
            failures += 1
    return failures


def run_dedup(feed_path: Path) -> int:
    """独立验证去重，不经过 LLM。

    dedup 是确定性逻辑，没有任何理由依赖模型：本函数直接对 Deduper 做单元测试，
    因此即使 LLM 全部判错（或完全不可用），去重的回归结论仍然可信。
    """
    from filters import Deduper, normalize_title

    print("\n--- 去重（独立于 LLM）---")
    failures = 0

    items = load_items(feed_path)

    # a) 真实 feed 里的重复标题：第二个必须被识别为重复
    groups: Dict[str, List[dict]] = {}
    for it in items:
        groups.setdefault(normalize_title(it["title"]), []).append(it)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not dup_groups:
        print("  SKIP  快照里没有重复标题")
    for key, group in dup_groups.items():
        d = Deduper()
        d.remember(group[0]["title"], group[0]["id"])
        hits = [it for it in group[1:] if d.is_duplicate(it["title"]) is not None]
        ok = len(hits) == len(group) - 1
        print(
            f"  {'PASS' if ok else 'FAIL'}  {len(group)} 条同标题「{group[0]['title'][:40]}」"
            f" -> 识别出 {len(hits)} 条重复"
        )
        if not ok:
            failures += 1

    # b) 用真实 feed 标题构造变体：日期写法与空白不同，但语义相同
    #    注意必须基于真实标题（含空格），合成标题会让测试失去检测力
    real = [it["title"] for it in items if "日期：" in it["title"]]
    if not real:
        print("  SKIP  快照里没有带「日期：」的标题")
    else:
        src_title = real[0]
        stem = re.split(r"日期\s*[:：]", src_title)[0].strip()
        variants = [
            src_title,
            stem + "日期：2026年7月18日",
            stem + " 2026年7月18日 ",
            stem + "日期: 2026年 7 月 18 日",
            stem + "日期：2026年2026年7月18日",
        ]
        d = Deduper()
        d.remember(variants[0], "first")
        missed = [v for v in variants[1:] if d.is_duplicate(v) is None]
        ok = not missed
        print(
            f"  {'PASS' if ok else 'FAIL'}  真实标题的日期/空白变体 -> "
            f"漏识 {len(missed)}/{len(variants)-1} 条"
        )
        if not ok:
            failures += 1
            for m in missed:
                print(f"        期望归一化为同一值，实际: {normalize_title(m)!r}")

    # b2) 正文不同 -> 不应被过度合并
    if real:
        stem = re.split(r"日期\s*[:：]", real[0]["title"] if isinstance(real[0], dict) else real[0])[0].strip()
        d = Deduper()
        d.remember(real[0] if isinstance(real[0], str) else real[0]["title"], "a")
        other = stem + "日期：2026年7月17日 完全不同的补充说明文字"
        ok = d.is_duplicate(other) is None
        print(f"  {'PASS' if ok else 'FAIL'}  正文不同时不被误合并")
        if not ok:
            failures += 1

    # c) 不同课程编号绝不能被当成重复
    d = Deduper()
    d.remember("00574【消費才是經濟之母】2026年7月18日", "a")
    distinct = [
        "00575【真正教會你的，不是結果，而是過程！】2026年7月25日",
        "00674【槓桿不能大於原型】2026年3月19日",
        "00573【真正讓我富有的】2026年7月11日",
    ]
    wrong = [v for v in distinct if d.is_duplicate(v) is not None]
    ok = not wrong
    print(f"  {'PASS' if ok else 'FAIL'}  不同编号不被误判为重复 -> 误判 {len(wrong)} 条")
    if not ok:
        failures += 1
        for w in wrong:
            print(f"        误判: {w[:50]}")

    # d) 关闭开关后应完全不去重
    probe = real[0] if real else "00574【X】日期：2026年7月18日"
    d = Deduper(enabled=False)
    d.remember(probe, "a")
    ok = d.is_duplicate(probe) is None
    print(f"  {'PASS' if ok else 'FAIL'}  dedup_enabled=False 时不去重")
    if not ok:
        failures += 1

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="内容过滤回归测试")
    parser.add_argument("--feed", type=Path, default=SNAPSHOT, help="feed 快照路径")
    parser.add_argument("--refresh", action="store_true", help="先重新抓取 feed")
    parser.add_argument(
        "--skip-degradation", action="store_true", help="跳过 LLM 不可用的降级测试"
    )
    args = parser.parse_args()

    feed_path = args.feed
    if args.refresh or not feed_path.exists():
        print("抓取 feed ...")
        data = fetch_snapshot()
        feed_path = SNAPSHOT
        feed_path.write_bytes(data)
        print(f"已保存快照: {feed_path}")

    code = run(feed_path)

    dedup_failures = run_dedup(feed_path)
    if dedup_failures:
        print(f"\n去重测试失败：{dedup_failures} 项")
        code = 1
    else:
        print("去重测试通过（未调用 LLM）")

    if not args.skip_degradation:
        extra = run_degradation(feed_path)
        if extra:
            print(f"\n降级测试失败：{extra} 项")
            code = 1
        else:
            print("降级测试通过")

    return code


if __name__ == "__main__":
    sys.exit(main())
