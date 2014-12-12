#! /bin/sh

# Install dependecies
apt-get install -y python-pip python-dev libxml2-dev libxslt1-dev libssl-dev libffi-dev

# Install mitmproxy
pip install mitmproxy

# Forward inbound HTTP and HTTPS traffic to the mitmproxy port
iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 80 -j REDIRECT --to-port 8080
iptables -t nat -A PREROUTING -i eth1 -p tcp --dport 443 -j REDIRECT --to-port 8080

# Download script for dropping malicious traffic
wget https://dl.dropboxusercontent.com/u/62123265/CS%20259d/mitmproxy/dropFlowsToGavilansDropbox.py

# Launch the proxy
echo "Launching the proxy. Proxy will output warning if intrusion detected"
echo "Ctrl-C to exit"

sudo mitmdump -T -s dropFlowsToGavilansDropbox.py -q