//! Opt-in ptrace attach for hang diagnosis (#353).
//!
//! With `RCSR_ATTACHABLE=1` every process that links this crate lets any
//! process of the same user attach, so `scripts/stacks.sh` works under Yama
//! `ptrace_scope=1`.

#[cfg(target_os = "linux")]
mod imp {
    use std::os::raw::{c_int, c_ulong};

    extern "C" {
        fn prctl(option: c_int, arg2: c_ulong, arg3: c_ulong, arg4: c_ulong, arg5: c_ulong)
            -> c_int;
    }

    const PR_SET_PTRACER: c_int = 0x5961_6d61;
    const PR_SET_PTRACER_ANY: c_ulong = c_ulong::MAX;

    extern "C" fn init() {
        if std::env::var_os("RCSR_ATTACHABLE").is_none_or(|v| v != "1") {
            return;
        }
        let rc = unsafe { prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) };
        if rc != 0 {
            eprintln!("rcsr: RCSR_ATTACHABLE=1 but prctl(PR_SET_PTRACER) failed");
        }
    }

    #[used]
    #[link_section = ".init_array"]
    static INIT: extern "C" fn() = init;
}

/// Forces the linker to keep this module's init entry in binaries that
/// reference the crate.
pub fn linked() -> bool {
    cfg!(target_os = "linux")
}
