import { expect, test } from "@playwright/test";

test("البث المباشر يتصل ويعرض الإشارة والنطاقات", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("متصل", { exact: true })).toBeVisible();
  await expect(page.locator(".signal-wrap canvas")).toBeVisible();
  // النطاقات تحتاج ثانيتين من الإشارة
  await expect(page.locator(".band-row")).toHaveCount(5);
});

test("فحص الصرع: الخط الزمني يقارن الإنذارات بالنوبات الحقيقية", async ({ page }) => {
  await page.goto("/#/epilepsy");
  await expect(page.getByRole("heading", { name: "فحص الصرع" })).toBeVisible();
  const first = page.locator(".epi-item:not([disabled])").first();
  await expect(first).toBeVisible();
  // يُفتح أول تسجيل فيه نوبة تلقائياً
  await expect(page.locator(".timeline svg")).toBeVisible();
  await expect(page.locator(".tl-true").first()).toBeAttached();
  await expect(page.locator(".banner").filter({ hasText: "نوبة" }).first()).toBeVisible();
});

test("رابط «شاهد النوبة» يفتح البث المباشر مع كاشف الصرع", async ({ page }) => {
  await page.goto("/#/epilepsy");
  await page.getByRole("link", { name: /شاهد النوبة في البث المباشر/ }).click();
  await expect(page).toHaveURL(/source=edf/);
  await expect(page.getByText("كاشف الصرع", { exact: false })).toBeVisible();
  await expect(page.locator(".meter")).toBeVisible();
});

test("صفحة النماذج تعرض تقارير الصرع بمقاييس الأحداث", async ({ page }) => {
  await page.goto("/#/models");
  await expect(page.getByRole("heading", { name: /كشف الصرع — نموذج خاص بكل مريض/ })).toBeVisible();
  await expect(page.getByText("إنذارات كاذبة في الساعة").first()).toBeVisible();
});

test("الكتابة بالدماغ: المحاكاة تكتب بمحاولات حقيقية، واللوحة تستجيب للوحة المفاتيح", async ({ page }) => {
  await page.goto("/#/speller");
  await expect(page.getByRole("heading", { name: "الكتابة بالدماغ" })).toBeVisible();
  await page.getByRole("combobox").nth(1).selectOption({ label: "سريع" });
  await page.getByRole("button", { name: /ابدأ/ }).click();
  await expect(page.getByText(/الخطوة \d+ من \d+/)).toBeVisible();
  await expect(page.getByText("أراد المستخدم:")).toBeVisible();
  await expect(page.locator(".prob-row")).toHaveCount(4);

  await page.getByRole("button", { name: "جرّب اللوحة بنفسك" }).click();
  await page.keyboard.press("Enter");           // ادخل المجموعة الأولى
  await page.keyboard.press("ArrowLeft");       // أ
  await page.keyboard.press("Enter");           // اكتبها
  await expect(page.locator(".typed")).toContainText("أ");
  await page.keyboard.press("Backspace");       // احذفها
  await expect(page.locator(".typed")).not.toContainText("أ");
});

test("جهازك: الإعداد والجاهزية، ثم البث المباشر من الجهاز المحاكى", async ({ page, request }) => {
  // الاختبار يغيّر إعداد الجهاز المحفوظ — نعيده كما كان في النهاية
  const before = await (await request.get("/api/device")).json();
  try {
    await page.goto("/#/device");
    await expect(page.getByRole("heading", { name: "جهازك" })).toBeVisible();
    await page.getByRole("combobox").first().selectOption("synthetic");
    await expect(page.locator(".channel input")).toHaveCount(16);
    await page.getByRole("button", { name: /احفظ الإعداد/ }).click();
    await expect(page.getByText("حُفظ", { exact: true })).toBeVisible();
    await expect(page.locator(".ready-card")).toHaveCount(4);
    await expect(page.locator(".ready-card").filter({ hasText: "الكتابة بالدماغ" })).toBeVisible();

    await page.goto("/#/?source=device");
    await expect(page.getByText("جهاز متصل", { exact: true })).toBeVisible();
    await expect(page.locator(".band-row")).toHaveCount(5);
    await expect(page.locator(".chip")).toHaveCount(16);
  } finally {
    if (before.config) await request.put("/api/device", { data: before.config });
    else await request.delete("/api/device");
  }
});

test("الحالة الذهنية: المقياس يتحرك مع الزمن ويعرض حدوده بصدق", async ({ page }) => {
  await page.goto("/#/state");
  await expect(page.getByRole("heading", { name: "الحالة الذهنية" })).toBeVisible();
  await expect(page.locator(".gauge svg")).toBeVisible();
  await expect(page.locator(".timeline svg")).toBeVisible();
  await page.getByRole("button", { name: /مسترخٍ ↔ تحت مهمة مجهِدة/ }).click();
  await expect(page.getByText(/لا يتبع شدة التوتر/).first()).toBeVisible();
  // الانتقال إلى منتصف الخط الزمني يحرّك الإبرة
  const tl = page.locator(".timeline svg");
  await tl.scrollIntoViewIfNeeded();
  const box = (await tl.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.8, box.y + box.height * 0.5);
  await expect(page.locator(".g-needle")).toBeVisible();
});
