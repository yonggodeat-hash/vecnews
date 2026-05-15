"""
Phase 0 小实验：对 50 条金标准标注运行 Reward 引擎
验证评分体系的有效性
"""
import json
import re
import os
import sys
from collections import defaultdict
import statistics

# 添加 newsdisco_reward 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from newsdisco_reward.annotation_schema import (
    NewsAnnotation, NewsMeta, SourceAnnotation, ClaimAnnotation,
    RhetoricAnnotation, AdjectiveAnnotation, NewsValueAnnotation,
    NeutralityAnnotation, NewsGrade,
    SourceType, ClaimType, VerificationStatus, RhetoricType,
    Perspective, ConflictOfInterest, Market, EventType,
    SOURCE_CREDIBILITY_BASE,
)
from newsdisco_reward.reward_engine import FinanceNewsRewardEngine


# ============================================================
# 字段映射
# ============================================================

SOURCE_TYPE_MAP = {
    "监管机构/交易所/法院/官方统计": SourceType.REGULATOR,
    "监管机构/交易所/法院/官方统计部门": SourceType.REGULATOR,
    "上市公司公告/财报/招股书/审计报告": SourceType.LISTED_COMPANY,
    "专业财经媒体": SourceType.PROFESSIONAL_MEDIA,
    "券商研报/评级机构": SourceType.BROKER_RESEARCH,
    "券商研报/评级机构/咨询机构": SourceType.BROKER_RESEARCH,
    "公司新闻稿/董秘回复/投资者关系": SourceType.COMPANY_PR,
    "公司新闻稿/董秘回复/投资者关系材料": SourceType.COMPANY_PR,
    "自媒体财经号": SourceType.SELF_MEDIA,
    "个人社交媒体发言": SourceType.PERSONAL_SOCIAL,
    "匿名爆料/群聊截图/传闻": SourceType.ANONYMOUS_RUMOR,
}

CLAIM_TYPE_MAP = {
    "财务事实": ClaimType.FINANCIAL_FACT,
    "公司解释": ClaimType.COMPANY_EXPLANATION,
    "市场分析事实": ClaimType.MARKET_ANALYSIS,
    "监管事实": ClaimType.REGULATORY_FACT,
    "预测": ClaimType.PREDICTION,
    "观点": ClaimType.OPINION,
    "传言": ClaimType.RUMOR,
}

VERIFICATION_MAP = {
    "confirmed": VerificationStatus.CONFIRMED,
    "cross_verified": VerificationStatus.CROSS_VERIFIED,
    "attributed": VerificationStatus.ATTRIBUTED,
    "partially_verified": VerificationStatus.PARTIALLY_VERIFIED,
    "unverified": VerificationStatus.UNVERIFIED,
    "disputed": VerificationStatus.DISPUTED,
    "fabricated": VerificationStatus.FABRICATED,
}

CONFLICT_MAP = {
    True: ConflictOfInterest.MEDIUM,
    False: ConflictOfInterest.NONE,
    "none": ConflictOfInterest.NONE,
    "low": ConflictOfInterest.LOW,
    "medium": ConflictOfInterest.MEDIUM,
    "high": ConflictOfInterest.HIGH,
}

PERSPECTIVE_MAP = {
    "投资者": Perspective.INVESTOR,
    "公司管理层": Perspective.COMPANY,
    "监管者": Perspective.REGULATOR,
    "消费者": Perspective.CONSUMER,
    "员工": Perspective.EMPLOYEE,
    "债权人": Perspective.CREDITOR,
    "供应商": Perspective.SUPPLIER,
    "公众": Perspective.PUBLIC,
    "全人类": Perspective.UNIVERSAL,
}

MARKET_MAP = {
    "A股": Market.A_SHARE,
    "港股": Market.HK,
    "美股": Market.US,
    "债券": Market.BOND,
    "商品": Market.COMMODITY,
    "外汇": Market.FOREX,
    "宏观": Market.MACRO,
    "产业": Market.INDUSTRY,
}

EVENT_TYPE_MAP = {
    "财报": EventType.EARNINGS,
    "并购": EventType.MA,
    "监管": EventType.REGULATORY,
    "诉讼": EventType.LITIGATION,
    "人事": EventType.PERSONNEL,
    "股价异动": EventType.PRICE_MOVE,
    "政策": EventType.POLICY,
    "行业数据": EventType.INDUSTRY_DATA,
}


def extract_json_from_md(filepath: str) -> dict:
    """从 markdown 文件中提取 JSON 块"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找 ```json ... ``` 代码块
    pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)

    if not matches:
        raise ValueError(f"No JSON block found in {filepath}")

    return json.loads(matches[0])


def convert_to_annotation(data: dict) -> NewsAnnotation:
    """将标注字典转为 NewsAnnotation"""

    # Meta
    meta = NewsMeta(
        news_id=data.get("news_id", "unknown"),
        title=data.get("title", ""),
        body_text=data.get("content", data.get("body_text", "")),
        publish_time=data.get("publish_time", ""),
        event_time=data.get("event_time", ""),
        market=MARKET_MAP.get(data.get("market", "A股"), Market.A_SHARE),
        companies=data.get("companies", []),
        event_type=EVENT_TYPE_MAP.get(data.get("event_type", "财报"), EventType.EARNINGS),
    )

    # Sources
    sources = []
    for s in data.get("sources", []):
        src_type_str = s.get("source_type", "专业财经媒体")
        src_type = SOURCE_TYPE_MAP.get(src_type_str, SourceType.PROFESSIONAL_MEDIA)

        has_conflict = s.get("has_conflict_of_interest", False)
        if isinstance(has_conflict, bool):
            conflict = ConflictOfInterest.MEDIUM if has_conflict else ConflictOfInterest.NONE
        else:
            conflict = CONFLICT_MAP.get(has_conflict, ConflictOfInterest.NONE)

        claim_type_str = s.get("claim_type", "事实")
        claim_type = CLAIM_TYPE_MAP.get(claim_type_str, ClaimType.FINANCIAL_FACT)

        sources.append(SourceAnnotation(
            source_name=s.get("source_name", "未知信源"),
            source_type=src_type,
            credibility_base=SOURCE_CREDIBILITY_BASE.get(src_type, 0.5),
            is_primary_source=s.get("is_primary_source", False),
            is_traceable=s.get("is_traceable", False),
            conflict_of_interest=conflict,
            source_claim_type=claim_type,
        ))

    # Claims
    claims = []
    for c in data.get("claims", []):
        claim_type_str = c.get("claim_type", "财务事实")
        claim_type = CLAIM_TYPE_MAP.get(claim_type_str, ClaimType.FINANCIAL_FACT)

        verify_str = c.get("verification_status", "attributed")
        verify_status = VERIFICATION_MAP.get(verify_str, VerificationStatus.ATTRIBUTED)

        evidence = c.get("evidence_sources", [])
        independent_count = len(set(evidence))

        claims.append(ClaimAnnotation(
            claim_id=c.get("claim_id", "cx"),
            claim_text=c.get("claim_text", ""),
            claim_type=claim_type,
            evidence_sources=evidence,
            independent_sources_count=max(1, independent_count),
            verification_status=verify_status,
            confidence=c.get("confidence", 0.5),
            are_sources_independent=(independent_count > 1),
        ))

    # Rhetoric
    rhetoric_data = data.get("rhetoric", {})
    adj_list = rhetoric_data.get("emotional_adjectives", [])
    unsupported = rhetoric_data.get("unsupported_adverbs", [])
    all_emotional = adj_list + unsupported

    adj_annotations = []
    for word in all_emotional:
        if isinstance(word, str):
            adj_annotations.append(AdjectiveAnnotation(
                word=word,
                normalized=word,
                type=RhetoricType.EMOTIONAL_RHETORIC,
                is_supported_by_fact=False,
                penalty=-2.0,
            ))
        elif isinstance(word, dict):
            adj_annotations.append(AdjectiveAnnotation(
                word=word.get("word", ""),
                normalized=word.get("normalized", ""),
                type=RhetoricType.EMOTIONAL_RHETORIC,
                is_supported_by_fact=False,
                penalty=float(word.get("penalty", -2)),
            ))

    # Count factual sentences from claims
    total_factual = max(1, len(claims))
    total_emotional = len(all_emotional)

    rhetoric = RhetoricAnnotation(
        adjectives_adverbs=adj_annotations,
        total_factual_sentences=total_factual,
        total_emotional_words=total_emotional,
    )

    # News Value
    nv = data.get("news_value", {})
    news_value = NewsValueAnnotation(
        timeliness=float(nv.get("timeliness", 3)),
        importance=float(nv.get("importance", 3)),
        prominence=float(nv.get("prominence", 3)),
        proximity=float(nv.get("proximity", 3)),
        human_interest=float(nv.get("human_interest", 2)),
    )

    # Neutrality
    neu = data.get("neutrality", {})
    dominant = PERSPECTIVE_MAP.get(neu.get("dominant_perspective", "公众"), Perspective.PUBLIC)

    missing_raw = neu.get("missing_perspectives", [])
    missing = []
    for m in missing_raw:
        p = PERSPECTIVE_MAP.get(m)
        if p:
            missing.append(p)

    neutrality = NeutralityAnnotation(
        dominant_perspective=dominant,
        missing_perspectives=missing,
        is_single_interest_framing=neu.get("is_single_interest_framing", False),
        has_counterparty_view=not bool(missing),
        fact_opinion_separated=neu.get("separates_fact_and_opinion", True),
        uncertainty_disclosed=neu.get("discloses_uncertainty", neu.get("uncertainty_disclosed", False)),
        transaction_inducement=neu.get("has_transaction_inducement", False),
    )

    return NewsAnnotation(
        meta=meta,
        sources=sources,
        claims=claims,
        rhetoric=rhetoric,
        news_value=news_value,
        neutrality=neutrality,
    )


def run_experiment():
    """主实验：对 50 条标注运行 Reward 引擎"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_dir = os.path.join(base_dir, "data", "annotations")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    engine = FinanceNewsRewardEngine()

    results = []
    errors = []

    for i in range(1, 51):
        filename = f"finance_{i:06d}.md"
        filepath = os.path.join(annotation_dir, filename)

        if not os.path.exists(filepath):
            errors.append(f"{filename}: file not found")
            continue

        try:
            data = extract_json_from_md(filepath)
            annotation = convert_to_annotation(data)
            result = engine.evaluate(annotation)
            results.append(result)
        except Exception as e:
            errors.append(f"{filename}: {e}")

    # ============================================================
    # 统计分析
    # ============================================================

    if not results:
        print("ERROR: No valid results!")
        for e in errors:
            print(f"  {e}")
        return

    scores = [r.total_score for r in results]
    grades = [r.grade.value for r in results]

    dim_scores = {
        "信源可信度": [r.source_credibility_score for r in results],
        "交叉印证": [r.cross_verification_score for r in results],
        "新闻价值": [r.news_value_score for r in results],
        "客观公正": [r.neutrality_score for r in results],
        "语言得分": [r.language_score for r in results],
    }

    print("=" * 70)
    print("📊 NewsDisco Phase 0 小实验 — Reward 引擎评估报告")
    print("=" * 70)
    print(f"\n标注总数: {len(results)} / 50")
    print(f"解析失败: {len(errors)}")

    print(f"\n## 总体统计\n")
    print(f"  平均分:   {statistics.mean(scores):.1f} / 100")
    print(f"  中位数:   {statistics.median(scores):.1f} / 100")
    print(f"  标准差:   {statistics.stdev(scores):.1f}" if len(scores) > 1 else "  标准差:   N/A")
    print(f"  最高分:   {max(scores):.1f} / 100")
    print(f"  最低分:   {min(scores):.1f} / 100")

    print(f"\n## 等级分布\n")
    grade_counts = defaultdict(int)
    for g in grades:
        grade_counts[g] += 1
    for g in ["S", "A", "B", "C", "D"]:
        count = grade_counts.get(g, 0)
        bar = "█" * (count)
        print(f"  {g}级: {count:3d} 篇  {bar}")

    print(f"\n## 各维度统计 (平均分/满分)\n")
    for dim_name, dim_vals in dim_scores.items():
        max_vals = {"信源可信度": 20, "交叉印证": 25, "新闻价值": 25, "客观公正": 20, "语言得分": 10}
        mx = max_vals[dim_name]
        avg = statistics.mean(dim_vals)
        pct = avg / mx * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {dim_name:8s}: {avg:5.1f}/{mx} ({pct:5.1f}%) {bar}")

    # Top 5 & Bottom 5
    sorted_results = sorted(results, key=lambda r: r.total_score, reverse=True)

    print(f"\n## Top 5 最高分\n")
    for rank, r in enumerate(sorted_results[:5], 1):
        print(f"  {rank}. [{r.grade.value}] {r.meta.title[:50]}... → {r.total_score:.1f}分")
        print(f"     优点: {'; '.join(r.strengths[:2])}")

    print(f"\n## Bottom 5 最低分\n")
    for rank, r in enumerate(sorted_results[-5:], 1):
        print(f"  {rank}. [{r.grade.value}] {r.meta.title[:50]}... → {r.total_score:.1f}分")
        print(f"     缺陷: {'; '.join(r.weaknesses[:2])}")

    # 硬性扣分统计
    penalties = [r.hard_penalty for r in results if r.hard_penalty < 0]
    if penalties:
        print(f"\n## 硬性扣分\n")
        print(f"  触发扣分的文章: {len(penalties)} / {len(results)}")
        print(f"  平均扣分: {statistics.mean(penalties):.1f}")
        print(f"  最大扣分: {min(penalties):.1f}")

    # 建议行动分布
    actions = defaultdict(int)
    for r in results:
        actions[r.recommended_action] += 1
    print(f"\n## 建议行动\n")
    for action, count in actions.items():
        print(f"  {action}: {count} 篇")

    # Error report
    if errors:
        print(f"\n## 解析错误 ({len(errors)})\n")
        for e in errors:
            print(f"  ❌ {e}")

    print("\n" + "=" * 70)
    print("实验完成 ✅")
    print("=" * 70)

    # 输出 JSON 结果文件
    output = {
        "experiment": "Phase 0 Reward Validation",
        "n_annotations": len(results),
        "n_errors": len(errors),
        "statistics": {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "max": max(scores),
            "min": min(scores),
        },
        "grade_distribution": dict(grade_counts),
        "dimension_averages": {k: statistics.mean(v) for k, v in dim_scores.items()},
        "action_distribution": dict(actions),
        "top5": [{"id": r.meta.news_id, "title": r.meta.title, "score": r.total_score, "grade": r.grade.value}
                 for r in sorted_results[:5]],
        "bottom5": [{"id": r.meta.news_id, "title": r.meta.title, "score": r.total_score, "grade": r.grade.value}
                    for r in sorted_results[-5:]],
        "all_scores": [{"id": r.meta.news_id, "title": r.meta.title, "score": r.total_score,
                        "grade": r.grade.value, "dimensions": {
                            "source": r.source_credibility_score,
                            "cross_verify": r.cross_verification_score,
                            "news_value": r.news_value_score,
                            "neutrality": r.neutrality_score,
                            "language": r.language_score,
                            "penalty": r.hard_penalty,
                        }} for r in sorted_results],
    }

    output_path = os.path.join(results_dir, "phase0_experiment_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")

    return results, errors


if __name__ == "__main__":
    run_experiment()
