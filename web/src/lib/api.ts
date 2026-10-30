// طلبات HTTP للخادم (أثناء التطوير يحوّلها vite إلى المنفذ 8770)

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `خطأ من الخادم (${res.status})`);
  }
  return res.json() as Promise<T>;
}

/** POST/PUT/DELETE بجسم JSON — رسالة الخطأ من الخادم كما هي (عربية ومفهومة). */
export async function sendJson<T>(method: "POST" | "PUT" | "DELETE", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `خطأ (${res.status})`);
  return data as T;
}
