Here are 3 recently reported, reproducible issues with scheduler-adjacent symptoms (RCU stalls / scheduling stalls). All three are “unclaimed” by our heuristic (no patch-like [PATCH threads detected from the linked lore threads), and none show fixed/dup signals in the quick check:

INFO: rcu detected stall in inotify_add_watch

extid: aa5520f7faf8d5438034
bug: https://syzkaller.appspot.com/bug?extid=aa5520f7faf8d5438034
reported: 2026/01/19 (syz repro)
INFO: rcu detected stall in br_handle_frame (6)

extid: f8850bc3986562f79619
bug: https://syzkaller.appspot.com/bug?extid=f8850bc3986562f79619
reported: 2026/01/13 (C repro + syz repro)
INFO: rcu detected stall in cleanup_net (8)

extid: 0604401cc084920f6c3d
bug: https://syzkaller.appspot.com/bug?extid=0604401cc084920f6c3d
reported: 2026/01/13 (C repro + syz repro)
Pick one extid you want to start with (I’d lean f8850bc3986562f79619 or 0604401cc084920f6c3d since they have C repro), and I’ll immediately run:

./tools/syzbot_prepare_qemu_repro.py --extid <EXTID>
then the generated repro/<EXTID>/run_qemu.sh to reproduce and start triage.

