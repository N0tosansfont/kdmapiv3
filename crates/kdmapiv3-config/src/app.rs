//! `eframe` / `egui` UI implementation for `kdmapiv3-config`.
//!
//! Styled after the classic OmniMIDI Configurator with a compact dark-mode layout:
//! - Top Status & Quick Actions Bar (`kdmapiv3.toml` path, Save & Apply, Reload, Reset Defaults, status badge)
//! - Engine & Performance Tab (Voice Limit, Interpolation, Buffer Size, Ring Buffer Capacity, Telemetry)
//! - SoundFonts Tab (Priority reorder, Enable/Disable, Remove, `rfd` file picker, Drag-and-Drop)
//! - Live Diagnostics / Telemetry Panel (2Hz non-blocking `kdmapiv3.log` tail with color-coded XRUN badge)

use std::path::{Path, PathBuf};
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, Vec2};

use crate::config_io::{
    discover_config_path, is_supported_soundfont, load_config_file, log_path_for_config,
    normalize_path_separators, save_config_atomic, EditorConfig, InterpolationChoice,
    SoundFontItem, BUFFER_FRAMES_MAX, BUFFER_FRAMES_MIN, BUFFER_FRAME_PRESETS,
    RENDER_THREADS_MAX, RENDER_THREADS_MIN, RING_CAPACITY_MAX, RING_CAPACITY_MIN,
    RING_CAPACITY_PRESETS, VOICE_LIMIT_MAX, VOICE_LIMIT_MIN,
};
use crate::telemetry_tail::TelemetryPoller;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Engine,
    SoundFonts,
}

#[derive(Debug, Clone)]
enum StatusNotice {
    Saved,
    Unsaved,
    LoadedDefaults,
    Error(String),
}

pub struct ConfiguratorApp {
    config_path: PathBuf,
    log_path: PathBuf,
    config: EditorConfig,
    saved_snapshot: EditorConfig,
    active_tab: ActiveTab,
    status: StatusNotice,
    telemetry: TelemetryPoller,
}

impl ConfiguratorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Classic compact dark-mode appearance by default.
        let mut visuals = egui::Visuals::dark();
        visuals.window_corner_radius = egui::CornerRadius::same(6);
        visuals.panel_fill = Color32::from_rgb(20, 22, 27);
        visuals.extreme_bg_color = Color32::from_rgb(14, 16, 20);
        cc.egui_ctx.set_visuals(visuals);

        let config_path = discover_config_path();
        let log_path = log_path_for_config(&config_path);

        let (config, status) = match load_config_file(&config_path) {
            Ok((cfg, true)) => (cfg, StatusNotice::Saved),
            Ok((cfg, false)) => (cfg, StatusNotice::LoadedDefaults),
            Err(err) => (EditorConfig::default(), StatusNotice::Error(err)),
        };

        let saved_snapshot = config.clone();

        Self {
            config_path,
            log_path,
            config,
            saved_snapshot,
            active_tab: ActiveTab::Engine,
            status,
            telemetry: TelemetryPoller::default(),
        }
    }

    fn has_unsaved_changes(&self) -> bool {
        self.config != self.saved_snapshot
    }

    fn sync_dirty_status(&mut self) {
        if self.has_unsaved_changes() {
            if !matches!(self.status, StatusNotice::Error(_)) {
                self.status = StatusNotice::Unsaved;
            }
        } else if matches!(self.status, StatusNotice::Unsaved) {
            self.status = StatusNotice::Saved;
        }
    }

    fn action_save(&mut self) {
        match save_config_atomic(&self.config_path, &self.config) {
            Ok(()) => {
                // Normalize in-memory paths after save so UI matches file cleanly.
                for sf in &mut self.config.soundfonts {
                    sf.path = normalize_path_separators(&sf.path);
                }
                self.saved_snapshot = self.config.clone();
                self.status = StatusNotice::Saved;
            }
            Err(err) => {
                self.status = StatusNotice::Error(err);
            }
        }
    }

    fn action_reload(&mut self) {
        match load_config_file(&self.config_path) {
            Ok((cfg, true)) => {
                self.config = cfg.clone();
                self.saved_snapshot = cfg;
                self.status = StatusNotice::Saved;
            }
            Ok((cfg, false)) => {
                self.config = cfg.clone();
                self.saved_snapshot = cfg;
                self.status = StatusNotice::LoadedDefaults;
            }
            Err(err) => {
                self.status = StatusNotice::Error(err);
            }
        }
        self.telemetry.force_refresh(&self.log_path);
    }

    fn action_reset_defaults(&mut self) {
        self.config = EditorConfig::default();
        self.status = StatusNotice::Unsaved;
    }

    fn action_pick_config_file(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Select kdmapiv3.toml")
            .add_filter("TOML Config", &["toml"]);
        if let Some(dir) = self.config_path.parent() {
            if dir.is_dir() {
                dialog = dialog.set_directory(dir);
            }
        }
        if let Some(chosen) = dialog.pick_file() {
            self.config_path = chosen;
            self.log_path = log_path_for_config(&self.config_path);
            self.action_reload();
        }
    }

    fn action_add_soundfonts_dialog(&mut self) {
        if let Some(files) = rfd::FileDialog::new()
            .set_title("Add SoundFont (.sf2 / .sfz)")
            .add_filter("SoundFonts (*.sf2, *.sfz)", &["sf2", "sfz", "SF2", "SFZ"])
            .pick_files()
        {
            self.add_soundfont_paths(files);
        }
    }

    fn add_soundfont_paths(&mut self, paths: Vec<PathBuf>) {
        let mut added_any = false;
        for path in paths {
            if !is_supported_soundfont(&path) {
                continue;
            }
            let normalized = normalize_path_separators(&path.to_string_lossy());
            if normalized.is_empty() {
                continue;
            }
            if !self
                .config
                .soundfonts
                .iter()
                .any(|existing| existing.path.eq_ignore_ascii_case(&normalized))
            {
                self.config.soundfonts.push(SoundFontItem {
                    path: normalized,
                    enabled: true,
                });
                added_any = true;
            }
        }
        if added_any {
            self.sync_dirty_status();
        }
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() {
            self.add_soundfont_paths(dropped);
            self.active_tab = ActiveTab::SoundFonts;
        }
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_status_bar")
            .exact_height(68.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("kdmapiv3 Configurator")
                            .strong()
                            .size(16.0)
                            .color(Color32::from_rgb(125, 207, 255)),
                    );

                    ui.separator();

                    // Status indicator badge
                    let (badge_text, badge_color) = match &self.status {
                        StatusNotice::Saved => ("Config Saved", Color32::from_rgb(65, 195, 115)),
                        StatusNotice::Unsaved => {
                            ("Unsaved Changes", Color32::from_rgb(245, 185, 65))
                        }
                        StatusNotice::LoadedDefaults => (
                            "Default Config (Not Yet Saved)",
                            Color32::from_rgb(130, 170, 235),
                        ),
                        StatusNotice::Error(_) => ("Error", Color32::from_rgb(235, 85, 85)),
                    };

                    egui::Frame::NONE
                        .fill(badge_color.gamma_multiply(0.18))
                        .stroke(egui::Stroke::new(1.0, badge_color))
                        .corner_radius(egui::CornerRadius::same(4))
                        .inner_margin(egui::Margin::symmetric(8, 2))
                        .show(ui, |ui| {
                            ui.label(RichText::new(badge_text).small().strong().color(badge_color));
                        });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let save_btn = egui::Button::new(
                            RichText::new("Save & Apply").strong().color(Color32::WHITE),
                        )
                        .fill(Color32::from_rgb(38, 128, 86));
                        if ui.add(save_btn).clicked() {
                            self.action_save();
                        }

                        if ui.button("Reload").clicked() {
                            self.action_reload();
                        }

                        if ui.button("Reset Defaults").clicked() {
                            self.action_reset_defaults();
                        }
                    });
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Config:").small().weak());
                    let path_text = self.config_path.to_string_lossy();
                    ui.label(
                        RichText::new(path_text.as_ref())
                            .small()
                            .monospace()
                            .color(Color32::from_rgb(195, 205, 220)),
                    )
                    .on_hover_text(path_text.as_ref());

                    if ui.small_button("Browse...").clicked() {
                        self.action_pick_config_file();
                    }

                    if let StatusNotice::Error(ref msg) = self.status {
                        ui.separator();
                        ui.label(
                            RichText::new(msg)
                                .small()
                                .color(Color32::from_rgb(240, 100, 100)),
                        );
                    }
                });
            });
    }

    fn render_telemetry_panel(&mut self, ctx: &egui::Context) {
        let reading = self.telemetry.poll_if_due(&self.log_path).clone();

        egui::TopBottomPanel::bottom("live_telemetry_panel")
            .exact_height(66.0)
            .show(ctx, |ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Live Diagnostics").strong().small());
                    ui.label(
                        RichText::new("(2Hz non-blocking tail of kdmapiv3.log)")
                            .small()
                            .weak(),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (badge_label, badge_col) = if !reading.log_found {
                            ("No Telemetry Log", Color32::from_rgb(130, 138, 152))
                        } else if reading.xrun_count == 0 {
                            ("0 XRUN - OK", Color32::from_rgb(65, 200, 115))
                        } else {
                            ("XRUN DETECTED", Color32::from_rgb(235, 75, 75))
                        };

                        egui::Frame::NONE
                            .fill(badge_col.gamma_multiply(0.18))
                            .stroke(egui::Stroke::new(1.0, badge_col))
                            .corner_radius(egui::CornerRadius::same(4))
                            .inner_margin(egui::Margin::symmetric(8, 2))
                            .show(ui, |ui| {
                                ui.label(
                                    RichText::new(badge_label)
                                        .small()
                                        .strong()
                                        .color(badge_col),
                                );
                            });
                    });
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    metric_chip(
                        ui,
                        "Events/sec",
                        &format!(
                            "{} in / {} out",
                            format_with_commas(reading.events_in_per_sec),
                            format_with_commas(reading.events_out_per_sec)
                        ),
                    );
                    ui.separator();
                    metric_chip(
                        ui,
                        "Dropped Events",
                        &format_with_commas(reading.events_dropped),
                    );
                    ui.separator();
                    metric_chip(ui, "XRUN Count", &format_with_commas(reading.xrun_count));
                    ui.separator();
                    metric_chip(
                        ui,
                        "Voices",
                        &format_with_commas(reading.active_voices),
                    );
                });
            });
    }

    fn render_engine_tab(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.label(
            RichText::new("Real-Time Synth & Ring Buffer Settings")
                .strong()
                .size(14.0),
        );
        ui.label(
            RichText::new("Changes take effect on the next InitializeKDMAPIStream() call (player restart / MIDI reopen).")
                .small()
                .weak(),
        );
        ui.add_space(10.0);

        egui::Grid::new("engine_settings_grid")
            .num_columns(2)
            .spacing([18.0, 14.0])
            .striped(false)
            .show(ui, |ui| {
                // 1. Voice Limit
                ui.label(RichText::new("Voice Limit:").strong());
                ui.horizontal(|ui| {
                    let slider = egui::Slider::new(
                        &mut self.config.voice_limit,
                        VOICE_LIMIT_MIN..=VOICE_LIMIT_MAX,
                    )
                    .logarithmic(true)
                    .show_value(false);
                    ui.add_sized([220.0, 20.0], slider);

                    ui.add(
                        egui::DragValue::new(&mut self.config.voice_limit)
                            .range(VOICE_LIMIT_MIN..=VOICE_LIMIT_MAX)
                            .speed(100),
                    );
                    ui.label(RichText::new("voices (1 - 1,000,000)").small().weak());
                });
                ui.end_row();

                // 2. Interpolation Quality
                ui.label(RichText::new("Interpolation Quality:").strong());
                egui::ComboBox::from_id_salt("interpolation_combo")
                    .width(240.0)
                    .selected_text(self.config.interpolation.label())
                    .show_ui(ui, |ui| {
                        for choice in InterpolationChoice::ALL {
                            ui.selectable_value(
                                &mut self.config.interpolation,
                                choice,
                                choice.label(),
                            );
                        }
                    });
                ui.end_row();

                // 3. Audio Buffer Size (full manual entry + quick presets, zero snapping)
                ui.label(RichText::new("Audio Buffer Size:").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.config.buffer_frames)
                            .range(BUFFER_FRAMES_MIN..=BUFFER_FRAMES_MAX)
                            .speed(16)
                            .suffix(" frames"),
                    );
                    egui::ComboBox::from_id_salt("buffer_frames_combo")
                        .width(120.0)
                        .selected_text("Presets")
                        .show_ui(ui, |ui| {
                            for frames in BUFFER_FRAME_PRESETS {
                                ui.selectable_value(
                                    &mut self.config.buffer_frames,
                                    frames,
                                    format!("{} frames", frames),
                                );
                            }
                        });
                    ui.label(RichText::new("(32 - 8,192)").small().weak());
                });
                ui.end_row();

                // 4. Render Threads (full manual control, zero hidden auto-scaling)
                ui.label(RichText::new("Render Threads:").strong());
                ui.horizontal(|ui| {
                    let slider = egui::Slider::new(
                        &mut self.config.render_threads,
                        RENDER_THREADS_MIN..=RENDER_THREADS_MAX,
                    )
                    .logarithmic(true)
                    .show_value(false);
                    ui.add_sized([220.0, 20.0], slider);

                    ui.add(
                        egui::DragValue::new(&mut self.config.render_threads)
                            .range(RENDER_THREADS_MIN..=RENDER_THREADS_MAX)
                            .speed(1),
                    );
                    let detected = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    ui.label(
                        RichText::new(format!("threads (1 - 256, host CPU: {})", detected))
                            .small()
                            .weak(),
                    );
                });
                ui.end_row();

                // 5. Ring Buffer Capacity (full manual entry + quick presets, zero snapping)
                ui.label(RichText::new("Ring Buffer Capacity:").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.config.ring_buffer_capacity)
                            .range(RING_CAPACITY_MIN..=RING_CAPACITY_MAX)
                            .speed(1024),
                    );
                    egui::ComboBox::from_id_salt("ring_capacity_combo")
                        .width(165.0)
                        .selected_text("Presets")
                        .show_ui(ui, |ui| {
                            for cap in RING_CAPACITY_PRESETS {
                                ui.selectable_value(
                                    &mut self.config.ring_buffer_capacity,
                                    cap,
                                    format_ring_capacity(cap),
                                );
                            }
                        });
                });
                ui.end_row();

                // 6. Telemetry Toggle
                ui.label(RichText::new("Background Telemetry:").strong());
                ui.checkbox(
                    &mut self.config.telemetry_enabled,
                    "Write 1-second atomic counter summaries to kdmapiv3.log",
                );
                ui.end_row();
            });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        ui.label(
            RichText::new("DSP Effects (Low-Spec Toggles)")
                .strong()
                .size(13.5),
        );
        ui.label(
            RichText::new("Disabled effects are bypassed with 0% hot-path CPU overhead for low-spec CPUs or extreme Black MIDI playback.")
                .small()
                .weak(),
        );
        ui.add_space(6.0);

        ui.checkbox(
            &mut self.config.disable_all_effects,
            RichText::new("Disable All Effects (Low-Spec / Max Voices Mode)").strong(),
        );

        ui.add_space(4.0);
        ui.indent("dsp_effects_sub_toggles", |ui| {
            ui.add_enabled_ui(!self.config.disable_all_effects, |ui| {
                ui.checkbox(
                    &mut self.config.enable_lowpass_filter,
                    "SF2 Low-Pass Filter & Velocity-to-Filter",
                );
                ui.checkbox(
                    &mut self.config.enable_vibrato_lfo,
                    "Vibrato LFO (Mod Wheel / Pitch LFO)",
                );
            });
        });
    }

    fn render_soundfonts_tab(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("SoundFont Load Order (.sf2 / .sfz)")
                    .strong()
                    .size(14.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+ Add SoundFont...").clicked() {
                    self.action_add_soundfonts_dialog();
                }
            });
        });

        ui.label(
            RichText::new("Tip: Drag & drop .sf2 or .sfz files anywhere onto this window. Paths are normalized cleanly for Linux & Wine.")
                .small()
                .weak(),
        );
        ui.add_space(6.0);

        let hovering_files = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let border_color = if hovering_files {
            Color32::from_rgb(90, 195, 255)
        } else {
            Color32::from_rgb(52, 58, 70)
        };

        egui::Frame::NONE
            .fill(Color32::from_rgb(15, 17, 22))
            .stroke(egui::Stroke::new(if hovering_files { 2.0 } else { 1.0 }, border_color))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                if self.config.soundfonts.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(42.0);
                        ui.label(
                            RichText::new("No SoundFonts configured yet.")
                                .size(14.0)
                                .weak(),
                        );
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(
                                "Click \"+ Add SoundFont...\" or drop .sf2 / .sfz files here.",
                            )
                            .small()
                            .weak(),
                        );
                        ui.add_space(42.0);
                    });
                    return;
                }

                let mut move_up: Option<usize> = None;
                let mut move_down: Option<usize> = None;
                let mut remove_idx: Option<usize> = None;
                let count = self.config.soundfonts.len();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for idx in 0..count {
                            let sf = &mut self.config.soundfonts[idx];
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(format!("#{:02}", idx + 1))
                                        .monospace()
                                        .weak(),
                                );

                                ui.checkbox(&mut sf.enabled, "");

                                let filename = Path::new(&sf.path)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(&sf.path);

                                let name_color = if sf.enabled {
                                    Color32::from_rgb(230, 236, 245)
                                } else {
                                    Color32::from_rgb(120, 126, 138)
                                };

                                ui.vertical(|ui| {
                                    ui.label(RichText::new(filename).strong().color(name_color));
                                    ui.label(
                                        RichText::new(&sf.path)
                                            .small()
                                            .monospace()
                                            .weak(),
                                    );
                                });

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Remove")
                                                        .small()
                                                        .color(Color32::from_rgb(240, 120, 120)),
                                                )
                                                .min_size(Vec2::new(56.0, 22.0)),
                                            )
                                            .clicked()
                                        {
                                            remove_idx = Some(idx);
                                        }

                                        if ui
                                            .add_enabled(
                                                idx + 1 < count,
                                                egui::Button::new("Down").min_size(Vec2::new(46.0, 22.0)),
                                            )
                                            .clicked()
                                        {
                                            move_down = Some(idx);
                                        }

                                        if ui
                                            .add_enabled(
                                                idx > 0,
                                                egui::Button::new("Up").min_size(Vec2::new(36.0, 22.0)),
                                            )
                                            .clicked()
                                        {
                                            move_up = Some(idx);
                                        }
                                    },
                                );
                            });

                            if idx + 1 < count {
                                ui.separator();
                            }
                        }
                    });

                if let Some(i) = move_up {
                    if i > 0 {
                        self.config.soundfonts.swap(i, i - 1);
                    }
                } else if let Some(i) = move_down {
                    if i + 1 < self.config.soundfonts.len() {
                        self.config.soundfonts.swap(i, i + 1);
                    }
                } else if let Some(i) = remove_idx {
                    self.config.soundfonts.remove(i);
                }
            });
    }
}

impl eframe::App for ConfiguratorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_drag_and_drop(ctx);

        self.render_top_bar(ctx);
        self.render_telemetry_panel(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.active_tab,
                    ActiveTab::Engine,
                    RichText::new("Engine & Performance").strong(),
                );
                ui.selectable_value(
                    &mut self.active_tab,
                    ActiveTab::SoundFonts,
                    RichText::new(format!(
                        "SoundFonts ({})",
                        self.config.soundfonts.len()
                    ))
                    .strong(),
                );
            });
            ui.separator();

            match self.active_tab {
                ActiveTab::Engine => self.render_engine_tab(ui),
                ActiveTab::SoundFonts => self.render_soundfonts_tab(ui, ctx),
            }
        });

        self.sync_dirty_status();

        // Schedule gentle 2Hz repaints for the live telemetry panel without spinning CPU.
        ctx.request_repaint_after(Duration::from_millis(500));
    }
}

fn metric_chip(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{}:", label)).small().weak());
        ui.label(
            RichText::new(value)
                .small()
                .monospace()
                .strong()
                .color(Color32::from_rgb(225, 232, 242)),
        );
    });
}

fn format_ring_capacity(cap: usize) -> String {
    let approx_mib = (cap as f64 * 24.0) / (1024.0 * 1024.0);
    format!("{} (~{:.0} MiB)", format_with_commas(cap as u64), approx_mib)
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (idx, &b) in bytes.iter().enumerate() {
        if idx > 0 && (bytes.len() - idx) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}
