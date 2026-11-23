//! Process isolation for the two native overlay-depth regressions.
use std::time::{Duration, Instant};

pub async fn child(test: &str, phase: &str) -> bool {
    const CHILD: &str = "TC_COLLECTION_OVERLAY_CHILD";
    let selected = format!("{test}:{phase}");
    if let Some(value) = std::env::var_os(CHILD) {
        return value == selected.as_str();
    }

    assert!(std::env::var_os("RUST_MIN_STACK").is_none());
    let log = std::env::temp_dir().join(format!(
        "{}-{}-{phase}.log",
        test.replace("::", "-"),
        std::process::id()
    ));

    let output = std::fs::File::create(&log).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD, selected)
        .stdout(output.try_clone().unwrap())
        .stderr(output)
        .spawn()
        .unwrap();

    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success(),
                "{test} {phase} failed: {status}; log: {}",
                log.display()
            );

            return false;
        }

        if started.elapsed() > Duration::from_secs(60) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{test} {phase} timed out; log: {}", log.display());
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
