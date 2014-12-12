High-level description
----------------------
The botnet uses my public Dropbox folder as its CNC. The
signature we use is the resource indicated in the HTTP
request (i.e. GET /u/62123265/CS259d/payload.sh). However,
Dropbox uses HTTPS by default, so the IDS cannot see the
fields within the encrypted HTTP request. We cannot filter
on IP or port because legitimate Dropbox traffic would be
caught in the filter. Instead, we choose to setup the IDS
as a transparent proxy that man-in-the-middle's all hosts
using it as a gateway. 

To do this, we use the mitmproxy linux tool. It inspects all
outbound requests, and drops any sessions with an HTTP 
request for a resource containing the string 
"/u/62123265/CS259d/". This is a specific enough
signature that we can assume legitimate traffic will not
be blocked. 

Normally, using a MitM proxy, we would need to install
the proxy's certificates in the host using the proxy. But
because wget on the ancient victim VMs doesn't quite work
with HTTPS anyway, the implant code ignores all
certificates and so we don't need to actually install
the correct ones.


To install and run mitmproxy (our online IDS)
---------------------------------------------
- Login to the IDS VM as user
- https://dl.dropboxusercontent.com/u/62123265/CS%20259d/mitmproxy/installAndRunMitmproxy.sh
- chmod 777 installAndRunMitmproxy.sh
- sudo ./installAndRunMitmproxy.sh

The script will install like 20 dependencies, and then mitmproxy itself. Some dependencies will take ~2 minutes to compile and it'll appear like the script is stuck. Then a couple iptables rules are added to reroute incoming HTTP and HTTPS traffic into the port mitmproxy binds to. Another script is downloaded which is used by mitmproxy to filter HTTP requests using our signature. Lastly, mitmproxy is run, and the console will remain empty until an infected VM attempts to reach the CNC, or until a clean VM attempts to download the implant. In either case, it will kill the TCP session and present an alert
on the console.


Two botnet features
-------------------
From part 3, we've already implemented DDoS. See bin/premade_payloads/payload_ddosGoogle.sh

We chose to add spam emails. See bin/premade_payloads/payload_sendSpam.sh. To use it, we would copy payload_sendSpam.sh to <my_dropbox>/CS259d/payload.sh and any infected machines would download and execute that script. To test it, I recommend copying payload_sendSpam.sh itself to the VM and running it directly.


If manually installing mitmproxy
--------------------------------
On the IDS...

login as user

sudo apt-get install python-pip python-dev libxml2-dev libxslt1-dev libssl-dev libffi-dev

sudo pip install mitmproxy

sudo iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 80 -j REDIRECT --to-port 8080

sudo iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 443 -j REDIRECT --to-port 8080

wget https://dl.dropboxusercontent.com/u/62123265/CS%20259d/mitmproxy/dropFlowsToGavilansDropbox.py

sudo mitmdump -T -s dropFlowsToGavilansDropbox.py -q