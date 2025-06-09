To test training in ISC mode, you can run these commands. You may need to poll the `isc experiments | tail -3` command repeatedly to observe status changes. The status is stored in the last table column.

A successful run may start in enqueued, move to running, go from running to paused any number of times, then go to completed.

A failed run will show 

```bash
(.chess-theme-classifier) root@b2d0488e99c4:~/chess-theme-classifier# isc train chessVision.isc
Using credentials file /root/credentials.isc
Success: Experiment created: 70ad60e6-fe04-43f5-91e6-1e3fb2d5c2a0
(.chess-theme-classifier) root@b2d0488e99c4:~/chess-theme-classifier# isc experiments | tail -3
├──────────────────────────────────────┼────────────────────────┼──────────────────────┼──────┼───────────────┼─────────────┼───────────┤
│ 70ad60e6-fe04-43f5-91e6-1e3fb2d5c2a0 │ chess-theme-classifier │ 2025-Jun-09 22:31:30 │ 1    │ cycle         │ 1 / 3       │ running   │
└──────────────────────────────────────┴────────────────────────┴──────────────────────┴──────┴───────────────┴─────────────┴───────────┘
```