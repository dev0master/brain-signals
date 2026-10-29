import { Activity, ChartColumn, Gauge, HeartPulse, Keyboard } from "lucide-react";
import { NavLink, Route, Routes } from "react-router-dom";
import { EpilepsyPage } from "./pages/EpilepsyPage";
import { LivePage } from "./pages/LivePage";
import { MentalPage } from "./pages/MentalPage";
import { ModelsPage } from "./pages/ModelsPage";
import { SpellerPage } from "./pages/SpellerPage";

const NAV = [
  { to: "/", label: "البث المباشر", icon: Activity },
  { to: "/epilepsy", label: "فحص الصرع", icon: HeartPulse },
  { to: "/speller", label: "الكتابة بالدماغ", icon: Keyboard },
  { to: "/state", label: "الحالة الذهنية", icon: Gauge },
  { to: "/models", label: "النماذج ونتائجها", icon: ChartColumn },
];

export default function App() {
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <div className="brand-name ltr">NeuroLens</div>
          <div className="small muted">قراءة إشارات الدماغ وترجمتها</div>
        </div>
        <nav>
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
              <Icon size={18} aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        <p className="side-note small faint">أداة تعليمية وبحثية — ليست جهازاً طبياً ولا تقدّم تشخيصاً.</p>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<LivePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/epilepsy" element={<EpilepsyPage />} />
          <Route path="/speller" element={<SpellerPage />} />
          <Route path="/state" element={<MentalPage />} />
        </Routes>
      </main>
    </div>
  );
}
