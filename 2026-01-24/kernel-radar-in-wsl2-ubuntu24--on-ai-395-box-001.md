I was able to trigger an RCU stall + hang/panic while running the ReproC inside the guest, and captured the full serial evidence.
In run1 (2 vCPUs), the guest hit rcu: INFO: rcu_preempt detected stalls… and then became SSH-unresponsive; log is in mylinux/kernel_radar/repro/f8850bc3986562f79619/qemu-serial-run1.log.
In run4 (2 vCPUs, more RAM), starting /root/repro again caused SSH to time out, then serial showed the RCU stall and later Kernel panic - not syncing: softlockup: hung tasks; log is in mylinux/kernel_radar/repro/f8850bc3986562f79619/qemu-serial-run4.log.
I did not see br_handle_frame appear as a string in the captured serial output for these runs (even though bridge setup is visible right before the stall).
I wrote up the exact commands + observations in mylinux/kernel_radar/docs/2026-01-24-f8850bc3986562f79619-monitoring-and-repro-run.md (committed as 872e551).
