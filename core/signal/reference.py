"""إعادة المرجعية: إزالة ما هو مشترك بين القنوات لإبراز النشاط المحلي.

كل قطب يقيس فرق الجهد بينه وبين قطب مرجعي. التشويش الذي يصيب كل القنوات معاً
(حركة، كهرباء) يمكن طرحه:

- CAR: نطرح من كل قناة متوسط كل القنوات.
- Laplacian السطحي: نطرح من كل قناة متوسط جيرانها القريبين فقط، فيبقى ما يخص
  تلك البقعة من الرأس. هذا مهم للكتابة بالدماغ لأن تخيّل حركة اليد يظهر محلياً
  فوق منطقة الحركة (C3 لليد اليمنى وC4 لليسرى).

ملاحظة: مشروع Data1a.ipynb كتب "Laplacian" على شكل np.eye(59) * -1 — وهذا
يضرب الإشارة في -1 فقط ولا يطرح أي جيران. التنفيذ الصحيح أدناه.
"""
import numpy as np


def car(data: np.ndarray) -> np.ndarray:
    """المرجع المتوسط المشترك. data بالشكل (قنوات، عيّنات) أو (نوافذ، قنوات، عيّنات)."""
    return data - data.mean(axis=-2, keepdims=True)


def laplacian_matrix(ch_names: list[str], neighbors: dict[str, list[str]]) -> np.ndarray:
    """مصفوفة تحويل: كل قناة = نفسها − متوسط جيرانها الموجودين.

    القناة التي لا جيران لها في القائمة تبقى كما هي.
    """
    idx = {name: i for i, name in enumerate(ch_names)}
    n = len(ch_names)
    m = np.eye(n)
    for name, nbrs in neighbors.items():
        if name not in idx:
            continue
        present = [idx[x] for x in nbrs if x in idx]
        if present:
            m[idx[name], present] = -1.0 / len(present)
    return m


def apply_matrix(m: np.ndarray, data: np.ndarray) -> np.ndarray:
    """يطبّق مصفوفة قنوات على (قنوات، عيّنات) أو (نوافذ، قنوات، عيّنات)."""
    return np.einsum("ij,...jt->...it", m, data)


def neighbors_from_montage(ch_names: list[str], k: int = 4,
                           montage: str | None = None) -> dict[str, list[str]]:
    """أقرب k أقطاب لكل قناة حسب مواضعها القياسية على الرأس."""
    pos = make_positions(ch_names, montage)
    names = [n for n in ch_names if n in pos]
    xyz = np.array([pos[n] for n in names])
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    out = {}
    for i, name in enumerate(names):
        order = np.argsort(dist[i])
        out[name] = [names[j] for j in order[1 : k + 1]]
    return out


def make_positions(ch_names: list[str], montage: str | None = None) -> dict[str, np.ndarray]:
    import mne

    if montage is None:
        # MNE 1.13 أعاد تسمية standard_1005 إلى colin27_1005 (نفس المواضع)
        builtin = mne.channels.get_builtin_montages()
        montage = "colin27_1005" if "colin27_1005" in builtin else "standard_1005"
    mpos = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in mpos.items()}
    return {n: lower[n.lower()] for n in ch_names if n.lower() in lower}
