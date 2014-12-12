#! /bin/sh

intervalInSecs=0.5
durationInSecs=5
ddosTarget="www.google.com"

function daemonhelper {
    ping -i $intervalInSecs -w $durationInSecs $ddosTarget #> /dev/null 2>&1
}

case "$1" in	
  daemon)
      daemonhelper
      ;;

  *)
      $0 daemon
      ;;
esac

exit 0