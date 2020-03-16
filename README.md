# boatcoach-utils
Various command line utilities to analyse logs from BoatCoach app.

Please note that these scripts are in an incubating state, and are tailored for my personal purposes, so most of the configuration is "wired" in the code. If you are planning to use it, you need to modify the code, or just open an issue, and I will extract the configuration to a separate file.

For the structure of the logs directory, check: https://github.com/axt/boatcoach-logs/

## tsb.py
Calculate and plot training stress balance and aggregated weekly and monthly TSS.

### Training stress balance
![Training stress balance][TSB]

Weekly TSS | Monthly TSS
--- | ---
![Weekly TSS][TSS_W] | ![Monthly TSS][TSS_M]


[TSB]: https://i.imgur.com/ufy8ttc.png
[TSS_W]: https://i.imgur.com/qoOwzc1.png
[TSS_M]: https://i.imgur.com/w3fkxnd.png
