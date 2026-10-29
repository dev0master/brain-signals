"""فحص جودة كل قناة قبل الثقة بأي نتيجة.

قطب منفصل عن الجلد أو تالف يعطي إشارة مسطّحة أو مشبعة، وأي "تفسير" لها
لا معنى له. الواجهة تعرض هذه الحالة بجانب كل قناة.
"""
import numpy as np

# حدود تقريبية بالميكروفولت لـ EEG فروة الرأس بعد إزالة الانحراف البطيء
FLAT_STD_UV = 0.5        # أقل من هذا: القطب غالباً منفصل
NOISY_STD_UV = 150.0     # أكثر من هذا: حركة أو تلامس سيئ
SATURATED_ABS_UV = 1000  # أكثر من هذا: المضخّم وصل حدّه


def channel_quality(window: np.ndarray) -> list[str]:
    """حالة كل قناة في نافذة (قنوات، عيّنات): good | flat | noisy | saturated."""
    centered = window - window.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    peak = np.abs(centered).max(axis=1)
    status = []
    for s, p in zip(std, peak):
        if p > SATURATED_ABS_UV:
            status.append("saturated")
        elif s < FLAT_STD_UV:
            status.append("flat")
        elif s > NOISY_STD_UV:
            status.append("noisy")
        else:
            status.append("good")
    return status
