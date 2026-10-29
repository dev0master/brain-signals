/** صفحة مؤقتة للمراحل التي لم تُبنَ بعد — توضح ما سيأتي فيها. */
export function ComingSoon({ title, phase, children }: { title: string; phase: string; children: React.ReactNode }) {
  return (
    <div className="page">
      <header className="page-head">
        <h1>{title}</h1>
        <span className="pill pill-info">قيد البناء — {phase}</span>
      </header>
      <section className="card">{children}</section>
    </div>
  );
}
