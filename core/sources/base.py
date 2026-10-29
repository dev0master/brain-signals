"""الواجهة الموحّدة لكل مصادر الإشارة.

أي مصدر — تسجيل محفوظ يُعاد تشغيله، أو محاكي، أو جهاز حقيقي لاحقاً — يقدّم
الشيئين نفسيهما: معلومات القنوات، ودالة تعيد العيّنات الجديدة منذ آخر قراءة.
بقية النظام لا يعرف ولا يهمه من أين تأتي الإشارة.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SourceInfo:
    fs: float                 # معدل العيّنات: كم قراءة في الثانية لكل قناة
    ch_names: list[str]       # أسماء القنوات (مواضع الأقطاب على الرأس)
    kind: str                 # "replay" | "synthetic" | "device"
    description: str = ""
    # أحداث معروفة مسبقاً في التسجيل (مثل بداية ونهاية نوبة صرع بالثواني)
    # تُعرض في الواجهة للمقارنة فقط ولا يراها النموذج
    events: list[dict] = field(default_factory=list)


class SignalSource(ABC):
    info: SourceInfo

    def start(self) -> None:
        """يبدأ القراءة (للأجهزة: يفتح الاتصال)."""

    def stop(self) -> None:
        """يوقف القراءة ويحرّر الموارد."""

    @abstractmethod
    def read_chunk(self) -> np.ndarray:
        """العيّنات الجديدة منذ آخر استدعاء.

        الشكل (عدد القنوات، عدد العيّنات الجديدة)، والوحدة ميكروفولت (µV).
        قد يكون عدد العيّنات صفراً إن استُدعيت الدالة بسرعة.
        """

    @property
    def position_sec(self) -> float:
        """الموضع الحالي في التسجيل بالثواني (للمصادر التي لها بداية ونهاية)."""
        return 0.0
