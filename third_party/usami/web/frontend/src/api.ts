export class ApiError extends Error { constructor(public status: number, message: string) { super(message); } }

export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map((e: { loc: string[]; msg: string }) => `${e.loc.join('.')}: ${e.msg}`).join('; ')
      : detail?.message ?? `HTTP ${response.status}`;
    throw new ApiError(response.status, message);
  }
  return data as T;
}

export async function download(kind: 'world' | 'trace') {
  const response = await fetch(`/api/export/${kind}`, { credentials: 'same-origin' });
  if (!response.ok) throw new Error(`Export failed: HTTP ${response.status}`);
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url; link.download = `usami-${kind}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
