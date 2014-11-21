Implant details:

The install script ("install_extract_function.sh") behaves in several different ways depending on what arguments it is invoked with. If it is executed with no arguments (as when the user first runs it), then the script makes an addition to /etc/profile to provide the promised features. But then the script also copies itself to init.d and registers itself to run as a service on startup. At this point, the script executes itself again, but now with the "daemon" argument.

When run as a daemon, the install script will periodically poll the CNC for a payload to execute, download it, execute it, and then remove it from disk.

When the host reboots, the same script is started as a service, and is executed with the "start" argument. In this situation, the implant enables network connectivity and executes itself again as a daemon.

The implant is no more intelligent than this -- it simply downloads whatever shell-script payload it is given and executes it with root privileges. This allows the implant to be enormously flexible, and allows for the install script to be small (as it is expected to be). Any feature can implemented in this system -- the payload could be a keylogger itself, or a simple ping flood script, or a more complicated script to open a socket and present a remote shell to another host. The payload at time of submission periodically outputs "Attack at dawn!" to the console, which is not very stealhty, but demonstrates that the CNC tasking works.

For the sake of rapid development and guaranteed uptime, this CNC is simply a hardcoded url in the implant pointing to a file called "payload.sh" within a public dropbox folder. If there is no tasking to give to any bots, then "payload.sh" can be the empty file. Otherwise, "payload.sh" can be anything the controller wants, including an escalation to a more feature-rich implant model. Also for example, if I as the botmaster wanted to keep records of what hosts I'd implanted and when for each host, I could push down a payload that causes each host to survey itself and send its identifying info to some arbitrary host that will aggregate all the data for me.

