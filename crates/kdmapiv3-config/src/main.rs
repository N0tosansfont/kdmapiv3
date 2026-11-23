#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "we didn't build this for 32-bit, but if you're stubborn enough to compile it on x86 anyway, we won't stop you :) — kdmapiv3-config requires a 64-bit target (target_pointer_width = \"64\")."
);

mod app;
mod config_io;
mod telemetry_tail;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("kdmapiv3 Configurator")
            .with_inner_size([740.0, 620.0])
            .with_min_inner_size([620.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "kdmapiv3 Configurator",
        native_options,
        Box::new(|cc| Ok(Box::new(app::ConfiguratorApp::new(cc)))),
    )
}
