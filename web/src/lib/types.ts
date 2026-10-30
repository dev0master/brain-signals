// أشكال الرسائل القادمة من خادم Python (server/session.py و server/main.py)

export type KnownEvent = { start: number; end: number; label?: string };

export type Hello = {
  type: "hello";
  fs: number;
  display_fs: number;
  ch_names: string[];
  kind: "replay" | "synthetic" | "device";
  description: string;
  events: KnownEvent[];
  bands: string[];
  duration_sec: number | null;
  /** اسم ملف EDF الذي تُحفظ فيه هذه الجلسة، إن طُلب التسجيل */
  recording?: string | null;
};

export type Frame = {
  type: "frame";
  t: number;
  samples: number[][];
  bands?: number[][];
  quality?: Quality[];
  recording_sec?: number;
  [analyzer: string]: unknown;
};

export type ServerError = { type: "error"; message: string };

export type Quality = "good" | "flat" | "noisy" | "saturated";

export type SourceItem = { id: string; label: string; kind: string; n_events?: number };

export type ModelReport = {
  model_id: string;
  created_at: string;
  title?: string;
  track?: string;
  dataset?: string;
  evaluation?: string;
  [key: string]: unknown;
};

// ---------- الصرع ----------

export type EpiRecording = {
  file: string;
  patient: string;
  n_seizures: number;
  analyzed: boolean;
  source_id: string;
};

export type DetectedEvent = { start: number; end: number; alarm?: number };

export type EventMetrics = {
  n_seizures: number;
  detected: number;
  false_alarms: number;
  hours: number;
  sensitivity: number | null;
  false_alarms_per_hour: number | null;
  mean_latency_sec: number | null;
};

export type ModelTimeline = {
  times: number[];
  probs: number[];
  detected: DetectedEvent[];
  threshold: number;
  metrics?: EventMetrics;
};

export type EpiResult = {
  file: string;
  patient?: string;
  duration_sec: number;
  seizures: KnownEvent[];
  patient_specific?: ModelTimeline;
  general?: ModelTimeline;
  source_id?: string | null;
  warnings?: string[];
};

export type SeizureLive = {
  prob: number;
  alarm: boolean;
  threshold: number;
  /** من جهازك (النموذج العام المجهّز لأقطابه) لا من تسجيل محفوظ */
  device?: boolean;
  note?: string | null;
};

// ---------- الكتابة بالدماغ ----------

export type SpellerItem = { kind: "group" | "char" | "word"; value: string | number; label?: string };

export type SpellerView = { text: string; group: number | null; cursor: number; items: SpellerItem[] };

export type SpellerCommand = "left" | "right" | "select" | "back";

export type CommandMeta = { command: SpellerCommand; class: string; imagine: string; action: string };

export type SpellerInfo = {
  groups: string[][];
  commands: CommandMeta[];
  class_ar: Record<string, string>;
  seconds_per_command: number;
  initial: SpellerView;
};

export type SpellingStats = {
  target: string;
  runs: number;
  completion_rate: number;
  median_commands: number;
  median_chars_per_minute: number | null;
  ideal_commands: number;
};

export type SpellerSubject = {
  subject: number;
  accuracy: number | null;
  pipeline_ar: string | null;
  mapping: Record<SpellerCommand, string> | null;
  spelling: SpellingStats | null;
  spelling_default_mapping: SpellingStats | null;
};

export type SimStep = {
  intended: SpellerCommand;
  decoded: SpellerCommand;
  executed: SpellerCommand | null;
  probs: number[];
  trial: number;
  view: SpellerView;
};

export type SimResult = {
  subject: number;
  target: string;
  finished: boolean;
  steps: SimStep[];
  metrics: {
    commands: number;
    command_accuracy: number | null;
    abstained: number;
    minutes: number;
    chars_per_minute: number | null;
    ideal_commands: number;
    seconds_per_command: number;
  };
};

// ---------- الحالة الذهنية ----------

export type MentalScale = {
  id: string;
  title: string;
  low: string;
  high: string;
  auc: number;
  balanced_accuracy: number;
  subjects_rising: number;
  without_calibration?: { auc: number; subjects_rising: number; balanced_accuracy: number };
  rating_agreement?: { spearman_pooled: number; mean_within_subject: number; n_trials: number } | null;
  validity_notes: string[];
  top_features: { feature: string; weight: number; direction: string }[];
  dataset: string;
  subjects: string[];
  per_subject: Record<string, { subject: string; calm_mean: number; task_mean: number; auc: number }>;
};

export type MentalSegment = { kind: "calibration" | "calm" | "task"; label: string; start: number; end: number };

export type MentalTimeline = {
  subject: string;
  segments: MentalSegment[];
  times: number[];
  calibrated: number[];
  raw: number[];
  sources?: { label: string; source_id: string }[];
};

export type MentalLive = {
  calibrating?: boolean;
  remaining_sec?: number;
  value?: number;
  truth?: string;
  low?: string;
  high?: string;
  /** من جهازك: قراءة لكل مقياس مجهّز لأقطابه */
  scales?: { id: string; title: string; low: string; high: string; value: number }[];
};

// ---------- جهازك ----------

export type Connection = "none" | "serial" | "bluetooth";

export type BoardInfo = {
  id: string;
  label: string;
  connection: Connection;
  fs: number;
  n_channels: number;
  default_names: string[];
};

export type DeviceConfig = { board: string; serial_port: string; mac_address: string; ch_names: string[] };

export type ReadinessStatus = "ready" | "needs_prepare" | "needs_calibration" | "unsupported";

export type Readiness = {
  title: string;
  track: "live" | "motor_imagery" | "mental_state" | "epilepsy";
  status: ReadinessStatus;
  reason: string | null;
  missing_recommended?: string[];
  calibrated_at?: string | null;
  accuracy?: number | null;
  pairs?: string[];
  n_pairs_total?: number;
  missing_electrodes?: string[];
  scales?: { scale: string; title: string; used: string[]; n_dataset_channels: number; supported: boolean; prepared: boolean }[];
};

export type PrepareJob = { status: "idle" | "running" | "done" | "error"; log: string[]; elapsed_sec: number | null };

export type DeviceState = {
  config: DeviceConfig | null;
  boards: BoardInfo[];
  standard_names: string[];
  readiness: Readiness[] | null;
  prepare: PrepareJob;
};

export type CommandInfo = { command: SpellerCommand; class: string; imagine: string; action: string; icon: string };

export type CalibrationResult = {
  accuracy: number;
  balanced_accuracy: number;
  chance: number;
  recall: Record<string, number>;
  commands: CommandInfo[];
  spelling: SpellingStats;
  n_trials: number;
};

export type CalibrationLive = {
  phase: "lead_in" | "prepare" | "imagine" | "rest" | "training" | "done" | "error";
  n_trials: number;
  collected: number;
  per_class: number;
  trial?: number;
  remaining_sec?: number;
  class?: string | null;
  imagine?: string | null;
  result?: CalibrationResult | null;
  error?: string | null;
};

export type SpellerLive = {
  phase: "lead_in" | "prepare" | "imagine" | "result";
  remaining_sec: number;
  view: SpellerView;
  last: { decoded: SpellerCommand; executed: SpellerCommand | null; confidence: number; probs: Record<SpellerCommand, number> } | null;
  count: number;
  minutes: number;
  commands: CommandInfo[];
  target?: string;
  done?: boolean;
  suggested?: SpellerCommand | null;
};
