// Cortex-OS — AI System Control Agent
// Tauri v2 application entry point

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod hotkey;
mod tray;

use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

/// Global handle to the Python daemon process so we can kill it on exit.
struct DaemonProcess(Mutex<Option<Child>>);

fn spawn_daemon() -> Option<Child> {
    // Try to find the daemon directory relative to the executable
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();

    // Look for the daemon in several possible locations
    let possible_dirs = vec![
        exe_dir.join("daemon"),                           // bundled next to exe
        exe_dir.parent()?.join("daemon"),                 // one level up
        exe_dir.parent()?.parent()?.join("daemon"),       // two levels up (dev)
        std::path::PathBuf::from("daemon"),               // current working directory
        dirs::home_dir()?.join(".cortex-os").join("daemon"), // user install dir
    ];

    let daemon_dir = possible_dirs.into_iter().find(|d| d.join("pilot").exists());

    let child = if let Some(dir) = daemon_dir {
        // Found the daemon directory — run from there
        Command::new("python")
            .args(["-m", "pilot.server"])
            .current_dir(&dir)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()
    } else {
        // Fallback: try running globally (user may have `pip install -e .` done)
        Command::new("python")
            .args(["-m", "pilot.server"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()
    };

    if child.is_some() {
        println!("[Cortex-OS] Python daemon spawned successfully");
    } else {
        eprintln!("[Cortex-OS] Warning: Could not spawn Python daemon. Is Python installed?");
    }

    child
}

fn main() {
    // Spawn the Python daemon before building the Tauri app
    let daemon_child = spawn_daemon();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .manage(DaemonProcess(Mutex::new(daemon_child)))
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            window.hide().unwrap();

            tray::setup_tray(app)?;
            hotkey::register_hotkey(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                // Kill the daemon when the app window is destroyed
                if let Some(state) = window.try_state::<DaemonProcess>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(ref mut child) = *guard {
                            let _ = child.kill();
                            println!("[Cortex-OS] Python daemon stopped");
                        }
                    }
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::toggle_window,
            commands::get_daemon_status,
            commands::send_to_daemon,
            commands::confirm_action,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Cortex-OS");
}
