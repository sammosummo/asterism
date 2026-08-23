use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(manifest: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest)
        .args(arguments)
        .output()
        .expect("ASTERISM_BUILD_GIT_UNAVAILABLE");
    assert!(output.status.success(), "ASTERISM_BUILD_GIT_FAILED");
    String::from_utf8(output.stdout)
        .expect("ASTERISM_BUILD_GIT_OUTPUT_NOT_UTF8")
        .trim()
        .to_owned()
}

fn sha256(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .filter(|candidate| candidate.status.success())
        .or_else(|| {
            Command::new("shasum")
                .args(["-a", "256"])
                .arg(path)
                .output()
                .ok()
                .filter(|candidate| candidate.status.success())
        })
        .expect("ASTERISM_BUILD_SHA256_UNAVAILABLE");
    let digest = String::from_utf8(output.stdout)
        .expect("ASTERISM_BUILD_SHA256_OUTPUT_NOT_UTF8")
        .split_whitespace()
        .next()
        .expect("ASTERISM_BUILD_SHA256_OUTPUT_EMPTY")
        .to_owned();
    assert!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "ASTERISM_BUILD_SHA256_INVALID"
    );
    digest
}

fn main() {
    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"));
    let source_commit = env::var("ASTERISM_SOURCE_COMMIT")
        .unwrap_or_else(|_| git(&manifest, &["rev-parse", "HEAD"]));
    assert!(
        source_commit.len() == 40 && source_commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "ASTERISM_BUILD_SOURCE_COMMIT_INVALID"
    );

    let source_dirty = env::var("ASTERISM_SOURCE_DIRTY").unwrap_or_else(|_| {
        (!git(
            &manifest,
            &["status", "--porcelain", "--untracked-files=normal"],
        )
        .is_empty())
        .to_string()
    });
    assert!(
        matches!(source_dirty.as_str(), "true" | "false"),
        "ASTERISM_BUILD_SOURCE_DIRTY_INVALID"
    );

    let release_manifest = fs::read_to_string(manifest.join("release.toml"))
        .expect("ASTERISM_RELEASE_MANIFEST_UNREADABLE");
    let release_manifest_sha256 = sha256(&manifest.join("release.toml"));
    let cargo_lock_sha256 = sha256(&manifest.join("Cargo.lock"));
    let uv_lock_sha256 = sha256(&manifest.join("uv.lock"));
    let public_version = release_manifest
        .lines()
        .find_map(|line| {
            line.strip_prefix("version = \"")
                .and_then(|value| value.strip_suffix('"'))
        })
        .expect("ASTERISM_RELEASE_MANIFEST_VERSION_MISSING");
    assert!(
        !public_version.is_empty(),
        "ASTERISM_RELEASE_MANIFEST_VERSION_INVALID"
    );
    let release_build = release_manifest
        .lines()
        .find_map(|line| line.strip_prefix("release = "))
        .expect("ASTERISM_RELEASE_MANIFEST_RELEASE_MISSING");
    assert!(
        matches!(release_build, "true" | "false"),
        "ASTERISM_RELEASE_MANIFEST_RELEASE_INVALID"
    );

    println!("cargo:rerun-if-env-changed=ASTERISM_SOURCE_COMMIT");
    println!("cargo:rerun-if-env-changed=ASTERISM_SOURCE_DIRTY");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-changed=release.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=uv.lock");
    println!("cargo:rustc-env=ASTERISM_SOURCE_COMMIT={source_commit}");
    println!("cargo:rustc-env=ASTERISM_SOURCE_DIRTY={source_dirty}");
    println!("cargo:rustc-env=ASTERISM_PUBLIC_VERSION={public_version}");
    println!("cargo:rustc-env=ASTERISM_RELEASE_BUILD={release_build}");
    println!("cargo:rustc-env=ASTERISM_RELEASE_MANIFEST_SHA256={release_manifest_sha256}");
    println!("cargo:rustc-env=ASTERISM_CARGO_LOCK_SHA256={cargo_lock_sha256}");
    println!("cargo:rustc-env=ASTERISM_UV_LOCK_SHA256={uv_lock_sha256}");
}
