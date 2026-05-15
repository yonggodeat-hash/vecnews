"""
真实 LLM-Judge 校准：DeepSeek API
"""
import json, re, math, time, subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
API_KEY_ENV = "DEEPSEEK_API_KEY"
API_URL = "https://api.deepseek.com/chat/completions"

def avg(v): return sum(v)/len(v) if v else 0.0
def stdv(v):
    m = avg(v)
    return math.sqrt(sum((x-m)**2 for x in v)/len(v)) if v else 0.0

def spearman_rho(x, y):
    n = len(x)
    if n < 3: return 0.0
    def rank(v):
        idx = sorted(enumerate(v), key=lambda t: t[1])
        ranks = [0]*n; i = 0
        while i < n:
            j = i
            while j < n and idx[j][1] == idx[i][1]: j += 1
            ar = (i+j-1)/2.0 + 1
            for k in range(i,j): ranks[idx[k][0]] = ar
            i = j
        return ranks
    rx, ry = rank(x), rank(y)
    d2 = sum((rx[i]-ry[i])**2 for i in range(n))
    return 1 - 6*d2/(n*(n*n-1))

def extract_json_from_md(fp):
    t = Path(fp).read_text(encoding='utf-8')
    m = re.search(r'```json\s*\n(.*?)\n```', t, re.DOTALL)
    if m: return json.loads(m.group(1))
    m = re.search(r'\{[\s\S]*"news_id"[\s\S]*\}', t)
    if m: return json.loads(m.group(0))
    return None

SYSTEM_PROMPT = """你是资深财经新闻质量评审员。对新闻0-100综合评分。

## 评分锚定（重要！）
- 90-100(A): 信源权威+多信源交叉印证+价值极高+语言完美，极罕见（<5%）
- 75-89(B): 信源可靠，有一定交叉印证，价值高但有小缺陷
- 55-74(C): 大多数财经新闻。信源可接受但有瑕疵，交叉印证不足，价值中等
- 40-54(D): 信源弱/不可追溯，单信源无印证，存在明显质量问题
- <40(F): 严重问题：编造/无信源重大断言/标题党/交易诱导

## 评分维度
1. 信源可信度(0-20)：监管/交易所/法院(0.95) > 财报公告(0.90) > 专业媒体(0.80) > 券商研报(0.70) > 公司PR(0.65) > 自媒体(0.45) > 个人社交(0.30) > 匿名(0.10)。不可追溯=大幅扣分。
2. 交叉印证(0-25)：无信源=0，单一高可信源=10-12，2个独立信源=15-18，3+独立信源=20-25。注意：一个公告被多家媒体引用不算交叉印证！
3. 新闻价值(0-25)：时新性/重要性/显著性/接近性/趣味性各0-5
4. 客观公正(0-20)：区分事实观点、多方视角、披露不确定性、无交易诱导
5. 语言克制(0-10)：情绪词扣分，法律术语不扣

## 硬性扣分
编造-100 | 无源重大断言-50 | 传言当事实-40 | 诱导交易-30 | 标题党-20 | 缺数据口径-15 | 混淆事实观点-15

## 参考示例
例1：一篇仅有"某分析师认为"而无其他信源的股市评论 → 45-52分(D)
例2：一篇引用公司公告+交易所数据+独立行业分析师的财报新闻 → 78-85分(B)
例3：一篇来自监管机构官方发布、有多个独立专家解读、无情绪渲染的政策新闻 → 88-95分(A)

输出严格JSON：{"score": <整数>, "grade": "<A/B/C/D>", "reason": "<20字>"}"""

FEWSHOT_EXAMPLES = """
【已评分示例1】标题：开盘：创指高开1.23% 快手概念涨幅居前
内容摘要：三大指数集体高开，沪指报4256点，深成指16202点。消息面涵盖特朗普访华、腾讯Q1财报等。
评分：{"score": 48, "grade": "D", "reason": "信源混杂不可追溯，无交叉印证，纯信息罗列无分析"}
---
【已评分示例2】标题：破除地方保护'隐性壁垒'，国家启动八个月专项整治
内容摘要：市场监管总局宣布5-12月全国开展专项行动。锁定四类堵点：行政许可隐性门槛、限制要素流动、歧视性待遇、政府采购隐性壁垒。
评分：{"score": 75, "grade": "B", "reason": "信源权威（总局+司长），事实清晰，价值高，语言克制，缺多信源交叉印证"}
---
【已评分示例3】标题：独家｜开市客北京称正与开市客中国谈判
内容摘要：开市客北京相关负责人独家回应称正与开市客中国谈判，有望和解。
评分：{"score": 35, "grade": "D", "reason": "独家单信源不可追溯，无交叉印证，无官方公告佐证"}
---
请参考以上示例的评分尺度，对以下新闻评分："""

def call_deepseek(title, content):
    """通过 curl 调用 DeepSeek API"""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Missing {API_KEY_ENV}. Set it before running this script.")

    user_msg = FEWSHOT_EXAMPLES + f"\n【标题】{title}\n\n【全文】{content[:3000]}"

    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    })

    try:
        result = subprocess.run([
            'curl', '-s', API_URL,
            '-H', f'Authorization: Bearer {api_key}',
            '-H', 'Content-Type: application/json',
            '-d', payload
        ], capture_output=True, text=True, timeout=45)

        if result.returncode != 0:
            return {"score": 50, "grade": "C", "reason": f"curl error: {result.stderr[:80]}"}

        resp = json.loads(result.stdout)
        if 'choices' not in resp:
            return {"score": 50, "grade": "C", "reason": f"API error: {str(resp)[:80]}"}

        msg = resp['choices'][0]['message']['content']
        # 尝试解析 JSON
        try:
            return json.loads(msg)
        except:
            # 尝试从文本中提取
            m = re.search(r'"score"\s*:\s*(\d+)', msg)
            if m: return {"score": int(m.group(1)), "grade": "C", "reason": "parsed"}
            return {"score": 50, "grade": "C", "reason": f"parse fail: {msg[:80]}"}

    except subprocess.TimeoutExpired:
        return {"score": 50, "grade": "C", "reason": "timeout"}
    except Exception as e:
        return {"score": 50, "grade": "C", "reason": str(e)[:80]}

def compute_gold_score(data):
    from news_reward.reward_system import (
        NewsAnnotation, SourceAnnotation, ClaimAnnotation,
        RhetoricAnnotation, NewsValueAnnotation, NeutralityAnnotation,
        SourceType, ClaimType, VerificationStatus, Perspective,
        compute_full_reward,
    )
    sources = []
    for s in data.get('sources', []):
        try: st = SourceType(s.get('source_type', '专业财经媒体'))
        except: st = SourceType.PROFESSIONAL_MEDIA
        sources.append(SourceAnnotation(
            source_name=s.get('source_name',''), source_type=st,
            credibility_base=s.get('credibility_base',0.8),
            is_primary_source=s.get('is_primary_source',False),
            is_traceable=s.get('is_traceable',True),
            has_conflict_of_interest=s.get('has_conflict_of_interest',False),
            conflict_of_interest_detail=s.get('conflict_of_interest_detail',''),
            claim_type=s.get('claim_type','事实')))

    claims = []
    for c in data.get('claims', []):
        try: vs = VerificationStatus(c.get('verification_status','confirmed'))
        except: vs = VerificationStatus.ATTRIBUTED
        try: ct = ClaimType(c.get('claim_type','财务事实'))
        except: ct = ClaimType.FINANCIAL_FACT
        claims.append(ClaimAnnotation(
            claim_id=c.get('claim_id',''), claim_text=c.get('claim_text',''),
            claim_type=ct,
            evidence_sources=c.get('evidence_sources',c.get('evidence',[])),
            verification_status=vs, confidence=c.get('confidence',0.5)))

    rd = data.get('rhetoric', {})
    rhetoric = RhetoricAnnotation(
        emotional_adjectives=rd.get('emotional_adjectives',[]),
        unsupported_adverbs=rd.get('unsupported_adverbs',[]))

    nv = data.get('news_value', {})
    news_value = NewsValueAnnotation(
        timeliness=nv.get('timeliness',3), importance=nv.get('importance',3),
        prominence=nv.get('prominence',3), proximity=nv.get('proximity',3),
        human_interest=nv.get('human_interest',2))

    nd = data.get('neutrality', {})
    try: dp = Perspective(nd.get('dominant_perspective','投资者'))
    except: dp = Perspective.ALL_HUMANITY
    mps = []
    for mp in nd.get('missing_perspectives',[]):
        try: mps.append(Perspective(mp))
        except: pass
    neutrality = NeutralityAnnotation(
        dominant_perspective=dp, missing_perspectives=mps,
        is_single_interest_framing=nd.get('is_single_interest_framing',False),
        has_counterparty_view=nd.get('has_counterparty_view',True),
        discloses_uncertainty=nd.get('discloses_uncertainty',True),
        separates_fact_and_opinion=nd.get('separates_fact_and_opinion',nd.get('fact_opinion_separated',True)),
        has_transaction_inducement=nd.get('has_transaction_inducement',nd.get('transaction_inducement',False)))

    ann = NewsAnnotation(
        news_id=data.get('news_id',''), title=data.get('title',''),
        content=data.get('content',data.get('body_text','')),
        publish_time=data.get('publish_time',''), event_time=data.get('event_time',''),
        market=data.get('market',''), companies=data.get('companies',[]),
        event_type=data.get('event_type',''), sources=sources, claims=claims,
        rhetoric=rhetoric, news_value=news_value, neutrality=neutrality)

    res = compute_full_reward(ann)
    return res.reward_score, res.grade

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 是否全量
    full_run = '--full' in sys.argv
    n_samples = 50 if full_run else 15

    print("=" * 60)
    print(f"🔧 真实 LLM-Judge 校准 (DeepSeek, {n_samples}条)")
    print("=" * 60)

    # 加载
    files = sorted(ANNOTATION_DIR.glob('finance_*.md'))
    print(f"\n[1/3] 加载标注...")
    gold_data = []
    for f in files:
        d = extract_json_from_md(f)
        if d:
            gs, grade = compute_gold_score(d)
            d['_gold_score'] = gs
            d['_gold_grade'] = grade
            gold_data.append(d)

    print(f"  {len(gold_data)} 条, Gold avg={avg([d['_gold_score'] for d in gold_data]):.1f}")

    # 跑 API
    sample = gold_data[:n_samples]
    print(f"\n[2/3] DeepSeek API 评分 ({len(sample)}条)...\n")

    results = []
    for i, d in enumerate(sample):
        nid = d.get('news_id', '?')
        title = d.get('title', '')
        gs = d['_gold_score']

        print(f"  [{i+1:2d}/{len(sample)}] {nid}: {title[:45]}...", flush=True)
        judge = call_deepseek(title, d.get('content',''))
        js = float(judge.get('score', 50))
        reason = judge.get('reason','')[:50]
        delta = js - gs
        bar = '+' * max(0,int(delta)) + '-' * max(0,int(-delta))
        print(f"      gold={gs:.0f} judge={js:.0f} {bar} Δ={delta:+.0f}  {reason}")

        results.append({'news_id': nid, 'title': title[:50],
                         'gold_score': gs, 'judge_score': js,
                         'judge_grade': judge.get('grade','?'),
                         'delta': delta, 'reason': judge.get('reason','')})
        time.sleep(0.3)

    # 统计
    print(f"\n[3/3] 统计...")
    gs = [r['gold_score'] for r in results]
    js = [r['judge_score'] for r in results]
    rho = spearman_rho(gs, js)
    deltas = [r['delta'] for r in results]

    print(f"\n{'='*60}")
    print("📊 真实校准报告 (DeepSeek-V3)")
    print(f"{'='*60}")
    print(f"  样本数: {len(results)}")
    print(f"  Spearman ρ: {rho:.4f}")
    print(f"  通过 (ρ > 0.7): {'✅' if rho > 0.7 else '❌'}")
    print(f"  平均误差: {avg(deltas):.2f}")
    print(f"  RMSE: {math.sqrt(avg([d**2 for d in deltas])):.2f}")
    print(f"  Gold 均值: {avg(gs):.1f}  Judge 均值: {avg(js):.1f}")

    # 逐条
    print(f"\n  Top 5 偏差:")
    sorted_r = sorted(results, key=lambda r: abs(r['delta']), reverse=True)
    for r in sorted_r[:5]:
        print(f"  {r['news_id']}: gold={r['gold_score']:.0f} judge={r['judge_score']:.0f} Δ={r['delta']:+.0f}")

    # 保存
    report = {
        'model': 'deepseek-chat',
        'n_samples': len(results),
        'spearman_rho': rho,
        'passed': rho > 0.7,
        'mean_delta': avg(deltas),
        'rmse': math.sqrt(avg([d**2 for d in deltas])),
        'gold_avg': avg(gs),
        'judge_avg': avg(js),
        'results': results
    }
    path = RESULTS_DIR / ('real_calibration_full.json' if full_run else 'real_calibration_15.json')
    with open(path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ → {path.name}")

if __name__ == '__main__':
    main()
