import re

CNC_httpPathSignature = r'u/62123265/CS%20259d'

def request(context, flow):
    request = flow.request

    requestPath = flow.request.path

    if (re.search(CNC_httpPathSignature, requestPath)):
    	#context.log("Dropping a request for " + requestPath, level='error')
    	flow.kill(context._master)
    	printAlertToConsole(context, flow)


def printAlertToConsole(context, flow):
    clientAddr = flow.client_conn.address
    targetHost = flow.request.host
    requestPath = flow.request.path
    output = "\nMalware detected\n"
    output += "Internal host at %s attempted to connect " % clientAddr
    output += "to host at %s " % targetHost
    output += "and requested a suspicious resource at %s\n" % requestPath
    
    #print output
    context.log(output, level='error')