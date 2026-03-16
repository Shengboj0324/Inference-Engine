import sys
import importlib.util
import time
import csv
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import numpy as np

# Run from repo root: python benchmark_suite.py /path/to/repo
repo_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.').resolve()
sys.path.insert(0, str(repo_root))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


pq_mod = load_module('pq_mod', repo_root / 'app/scraping/priority_queue.py')
rank_mod = load_module('rank_mod', repo_root / 'app/intelligence/action_ranker.py')
res_mod = load_module('res_mod', repo_root / 'app/scraping/reservoir_sampling.py')

from app.domain.normalized_models import NormalizedObservation, SentimentPolarity, ContentQuality
from app.domain.inference_models import SignalInference, SignalPrediction, SignalType
from app.core.models import SourcePlatform, MediaType

PriorityQueue, CrawlItem, PriorityLevel = pq_mod.PriorityQueue, pq_mod.CrawlItem, pq_mod.PriorityLevel
ActionRanker = rank_mod.ActionRanker
ReservoirSampler = res_mod.ReservoirSampler

rng = np.random.default_rng(123)
priority_levels = list(PriorityLevel)
signal_types = [
    SignalType.ALTERNATIVE_SEEKING,
    SignalType.COMPETITOR_MENTION,
    SignalType.COMPLAINT,
    SignalType.FEATURE_REQUEST,
    SignalType.CHURN_RISK,
]


def make_obs_inf(i: int):
    obs_id = uuid4()
    user_id = uuid4()
    now = datetime.now(timezone.utc)
    obs = NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id=str(i),
        source_url=f'https://reddit.com/{i}',
        author='user',
        title='Need better alternative to current tool',
        normalized_text='Need better alternative to current tool because pricing is too high',
        media_type=MediaType.TEXT,
        published_at=now - timedelta(hours=1),
        fetched_at=now,
        sentiment=SentimentPolarity.NEGATIVE,
        quality=ContentQuality.HIGH,
        quality_score=0.9,
        completeness_score=0.9,
        engagement_velocity=float(rng.random() * 20),
        virality_score=float(rng.random()),
        id=obs_id,
    )
    sig = signal_types[int(rng.integers(0, len(signal_types)))]
    prob = float(0.5 + rng.random() * 0.5)
    pred = SignalPrediction(signal_type=sig, probability=prob)
    inf = SignalInference(
        normalized_observation_id=obs_id,
        user_id=user_id,
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        model_name='benchmark-model',
        model_version='1.0',
        inference_method='synthetic',
    )
    return obs, inf


def benchmark_priority_queue(rows: list[list[object]]):
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        items = [
            CrawlItem(
                priority_score=0.5,
                url=f'https://example.com/{i}',
                priority_level=priority_levels[int(rng.integers(0, len(priority_levels)))],
                estimated_freshness=float(rng.random()),
                estimated_relevance=float(rng.random()),
                engagement_score=float(rng.random()),
                created_at=datetime.utcnow() - timedelta(seconds=int(rng.integers(0, 3600))),
            )
            for i in range(n)
        ]
        pq = PriorityQueue(enable_deduplication=False)
        for item in items:
            pq.push(item)
        while not pq.is_empty():
            pq.pop()
        for trial in range(1, 5):
            pq = PriorityQueue(enable_deduplication=False)
            t0 = time.perf_counter()
            for item in items:
                pq.push(item)
            while not pq.is_empty():
                pq.pop()
            rows.append(['priority_queue_push_pop', n, trial, time.perf_counter() - t0])


def benchmark_action_ranker(rows: list[list[object]]):
    ranker = ActionRanker()
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        obs_dict = {}
        inferences = []
        for i in range(n):
            obs, inf = make_obs_inf(i)
            obs_dict[str(obs.id)] = obs
            inferences.append(inf)
        ranker.rank_batch(inferences, obs_dict)  # warmup
        for trial in range(1, 5):
            t0 = time.perf_counter()
            ranker.rank_batch(inferences, obs_dict)
            rows.append(['action_ranker_rank_batch', n, trial, time.perf_counter() - t0])


def benchmark_reservoir(rows: list[list[object]]):
    for n in [10000, 20000, 40000, 80000, 120000, 160000]:
        data = list(range(n))
        sampler = ReservoirSampler(reservoir_size=1024, enable_weighted=False, random_seed=42)
        for x in data:
            sampler.add(x)
        for trial in range(1, 4):
            sampler = ReservoirSampler(reservoir_size=1024, enable_weighted=False, random_seed=42)
            t0 = time.perf_counter()
            for x in data:
                sampler.add(x)
            rows.append(['reservoir_uniform_add', n, trial, time.perf_counter() - t0])


def main():
    rows: list[list[object]] = []
    benchmark_priority_queue(rows)
    benchmark_action_ranker(rows)
    benchmark_reservoir(rows)

    out = Path('benchmark_results.csv')
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['benchmark', 'n', 'trial', 'seconds'])
        writer.writerows(rows)
    print(f'Wrote {out.resolve()} with {len(rows)} measurements')


if __name__ == '__main__':
    main()
