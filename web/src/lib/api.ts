// طلبات HTTP للخادم (أثناء التطوير يحوّلها vite إلى المنفذ 8770)

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `خطأ من الخادم (${res.status})`);
  }
  return res.json() as Promise<T>;
}
