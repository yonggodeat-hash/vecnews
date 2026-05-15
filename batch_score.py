"""
批量评分脚本：加载 vecnews-main 下所有 50 条标注，运行 reward 系统，输出汇总报告
"""
import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加 reward 系统路径
sys.path.insert(0, str(Path(__file__).parent))
from news_reward.reward_system import (
    NewsAnnotation, SourceAnnotation, ClaimAnnotation,
    RhetoricAnnotation, NewsValueAnnotation, NeutralityAnnotation,
    SourceType, ClaimType, VerificationStatus, Perspective,
    compute_full_reward, to_eval_result,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_json_from_md(filepath: Path) -> dict:
    """从 markdown 文件中提取 JSON 标注"""
    text = filepath.read_text(encoding='utf-8')

    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # 尝试提取裸 JSON
    json_match = re.search(r'\{[\s\S]*"news_id"[\s\S]*\}', text)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"Cannot parse JSON from {filepath}")


def parse_annotation(data: dict) -> NewsAnnotation:
    """将 JSON dict 转换为 NewsAnnotation"""

    # 解析信源
    sources = []
    for s in data.get('sources', []):
        source_type_str = s.get('source_type', '专业财经媒体')
        try:
            source_type = SourceType(source_type_str)
        except ValueError:
            source_type = SourceType.PROFESSIONAL_MEDIA

        sources.append(SourceAnnotation(
            source_name=s.get('source_name', ''),
            source_type=source_type,
            credibility_base=s.get('credibility_base', 0.8),
            is_primary_source=s.get('is_primary_source', False),
            is_traceable=s.get('is_traceable', True),
            has_conflict_of_interest=s.get('has_conflict_of_interest', False),
            conflict_of_interest_detail=s.get('conflict_of_interest_detail', ''),
            claim_type=s.get('claim_type', '事实'),
        ))

    # 解析事实主张
    claims = []
    for c in data.get('claims', []):
        vs_str = c.get('verification_status', 'confirmed')
        try:
            vs = VerificationStatus(vs_str)
        except ValueError:
            vs = VerificationStatus.ATTRIBUTED

        ct_str = c.get('claim_type', '财务事实')
        try:
            ct = ClaimType(ct_str)
        except ValueError:
            ct = ClaimType.FINANCIAL_FACT

        claims.append(ClaimAnnotation(
            claim_id=c.get('claim_id', ''),
            claim_text=c.get('claim_text', ''),
            claim_type=ct,
            evidence_sources=c.get('evidence_sources', c.get('evidence', [])),
            verification_status=vs,
            confidence=c.get('confidence', 0.5),
        ))

    # 解析修辞
    rhetoric_data = data.get('rhetoric', {})
    rhetoric = RhetoricAnnotation(
        emotional_adjectives=rhetoric_data.get('emotional_adjectives', []),
        unsupported_adverbs=rhetoric_data.get('unsupported_adverbs', []),
    )

    # 解析新闻价值
    nv_data = data.get('news_value', {})
    news_value = NewsValueAnnotation(
        timeliness=nv_data.get('timeliness', 3),
        importance=nv_data.get('importance', 3),
        prominence=nv_data.get('prominence', 3),
        proximity=nv_data.get('proximity', 3),
        human_interest=nv_data.get('human_interest', 2),
    )

    # 解析客观公正
    neut_data = data.get('neutrality', {})
    dom_persp_str = neut_data.get('dominant_perspective', '投资者')
    try:
        dom_persp = Perspective(dom_persp_str)
    except ValueError:
        dom_persp = Perspective.ALL_HUMANITY

    missing_strs = neut_data.get('missing_perspectives', [])
    missing_persps = []
    for mp in missing_strs:
        try:
            missing_persps.append(Perspective(mp))
        except ValueError:
            pass

    neutrality = NeutralityAnnotation(
        dominant_perspective=dom_persp,
        missing_perspectives=missing_persps,
        is_single_interest_framing=neut_data.get('is_single_interest_framing', False),
        has_counterparty_view=neut_data.get('has_counterparty_view', True),
        discloses_uncertainty=neut_data.get('discloses_uncertainty', True),
        separates_fact_and_opinion=neut_data.get('separates_fact_and_opinion',
                                                  neut_data.get('fact_opinion_separated', True)),
        has_transaction_inducement=neut_data.get('has_transaction_inducement',
                                                  neut_data.get('transaction_inducement', False)),
    )

    return NewsAnnotation(
        news_id=data.get('news_id', 'unknown'),
        title=data.get('title', ''),
        content=data.get('content', data.get('body_text', '')),
        publish_time=data.get('publish_time', ''),
        event_time=data.get('event_time', ''),
        market=data.get('market', ''),
        companies=data.get('companies', []),
        event_type=data.get('event_type', ''),
        sources=sources,
        claims=claims,
        rhetoric=rhetoric,
        news_value=news_value,
        neutrality=neutrality,
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 找所有标注文件
    files = sorted(ANNOTATION_DIR.glob('finance_*.md'))
    print(f"找到 {len(files)} 个标注文件\n")

    results = []
    scores_by_grade = defaultdict(list)
    dimension_scores = {
        'source': [],
        'cross_verify': [],
        'language': [],
        'news_value': [],
        'neutrality': [],
    }

    for f in files:
        try:
            data = extract_json_from_md(f)
            annotation = parse_annotation(data)
            result = compute_full_reward(annotation)
            eval_out = to_eval_result(annotation)

            results.append({
                'news_id': result.news_id,
                'title': result.title[:60],
                'score': result.reward_score,
                'grade': result.grade,
                'rhetoric_rate': result.rhetoric.rhetoric_pollution_rate,
                'n_sources': len(result.sources),
                'n_claims': len(result.claims),
                'strengths': result.main_strengths[:2],
                'weaknesses': result.main_weaknesses[:2],
            })

            scores_by_grade[result.grade].append(result.reward_score)
            dimension_scores['source'].append(eval_out['details']['dimensions']['source_credibility'])
            dimension_scores['news_value'].append(result.news_value.timeliness +
                result.news_value.importance + result.news_value.prominence +
                result.news_value.proximity + result.news_value.human_interest)
            dimension_scores['language'].append(result.rhetoric.language_score)
            dimension_scores['neutrality'].append(result.neutrality.neutrality_score)

        except Exception as e:
            print(f"⚠️ 解析失败: {f.name} — {e}")

    # 排序
    results.sort(key=lambda x: x['score'], reverse=True)

    # 输出汇总
    print("=" * 70)
    print("📊 财经新闻 Reward 批量评分报告")
    print("=" * 70)

    # 总览
    scores = [r['score'] for r in results]
    print(f"\n## 总览")
    print(f"- 标注总数: {len(results)}")
    print(f"- 平均分: {sum(scores)/len(scores):.1f}/100")
    print(f"- 最高分: {max(scores):.1f}")
    print(f"- 最低分: {min(scores):.1f}")
    print(f"- 中位数: {sorted(scores)[len(scores)//2]:.1f}")

    print(f"\n## 评级分布")
    for grade in ['A', 'B', 'C', 'D']:
        count = len(scores_by_grade.get(grade, []))
        bar = '█' * count
        print(f"  {grade} 级: {count:2d} 篇 {bar}")

    print(f"\n## Top 10 高质量新闻")
    print(f"{'ID':<22} {'标题':<42} {'得分':>6} {'评级':>4}")
    print("-" * 76)
    for r in results[:10]:
        print(f"{r['news_id']:<22} {r['title'][:40]:<42} {r['score']:>6.1f} {r['grade']:>4}")

    print(f"\n## Bottom 10 低质量新闻")
    for r in results[-10:]:
        print(f"{r['news_id']:<22} {r['title'][:40]:<42} {r['score']:>6.1f} {r['grade']:>4}")
        if r['weaknesses']:
            for w in r['weaknesses'][:1]:
                print(f"  ⚠️ {w}")

    # 维度统计
    print(f"\n## 各维度平均分")
    for dim, vals in dimension_scores.items():
        if vals:
            print(f"  {dim}: {sum(vals)/len(vals):.1f}")

    # 修辞污染率
    rhetoric_rates = [r['rhetoric_rate'] for r in results]
    clean = sum(1 for r in rhetoric_rates if r <= 0.1)
    print(f"\n## 修辞质量")
    print(f"  修辞污染率 ≤ 10%: {clean}/{len(results)} 篇 ({clean/len(results)*100:.0f}%)")
    print(f"  平均污染率: {sum(rhetoric_rates)/len(rhetoric_rates)*100:.1f}%")

    # 输出详细 JSON
    output_path = RESULTS_DIR / 'batch_scoring_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细结果已保存到: {output_path}")

    return results


if __name__ == '__main__':
    main()
