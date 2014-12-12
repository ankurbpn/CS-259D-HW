Two botnet features
-------------------
From part 3, we've already implemented DDoS. See bin/premade_payloads/payload_ddosGoogle.sh

We chose to add spam emails. See bin/premade_payloads/payload_sendSpam.sh. To use it, we would copy payload_sendSpam.sh to <my_dropbox>/CS259d/payload.sh and any infected machines would download and execute that script. To test it, I recommend copying payload_sendSpam.sh itself to the VM and running it directly.