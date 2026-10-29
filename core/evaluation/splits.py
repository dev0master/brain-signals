"""تقسيم البيانات إلى تدريب واختبار دون تسريب.

الخطأ الأشهر في مشاريع EEG (ومنها rnn_brain_reading): تقسيم العيّنات عشوائياً.
اللحظتان المتجاورتان في التسجيل شبه متطابقتين، فإذا وقعت واحدة في التدريب
والأخرى في الاختبار "يحفظ" النموذج الإجابة وتخرج دقة عالية مزيّفة.

الحل: نقسّم حسب "مجموعة" كاملة — شخص، أو جلسة، أو ملف تسجيل، أو نوبة —
بحيث لا تظهر المجموعة نفسها في الطرفين أبداً.
"""
from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


class LeakageError(AssertionError):
    """مجموعة ظهرت في التدريب والاختبار معاً."""


def assert_no_leak(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    shared = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if shared:
        raise LeakageError(f"تسريب: المجموعات {sorted(shared)[:5]} في التدريب والاختبار معاً")


def group_splits(groups: np.ndarray, n_splits: int | None = None
                 ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """تقسيمات بحسب المجموعات، مع التحقق من عدم التسريب في كل تقسيم.

    n_splits=None ← اترك مجموعة واحدة للاختبار في كل مرة (leave-one-group-out).
    """
    groups = np.asarray(groups)
    dummy = np.zeros(len(groups))
    splitter = LeaveOneGroupOut() if n_splits is None else GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(dummy, groups=groups):
        assert_no_leak(groups, train_idx, test_idx)
        yield train_idx, test_idx
